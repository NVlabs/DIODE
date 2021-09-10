# --------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# Official PyTorch implementation of WACV2021 paper:
# Data-Free Knowledge Distillation for Object Detection
# A Chawla, H Yin, P Molchanov, J Alvarez
# --------------------------------------------------------


import torch
import torch.nn as nn
import torch.optim as optim
import collections
from utils_di import create_folder
import random, shutil, math
import torch
import torchvision
import torchvision.utils as vutils
from apex import amp
from PIL import Image

import numpy as np
from tensorboardX import SummaryWriter
import os, sys, json, tempfile, subprocess
from models.yolo.yolostuff import flip_targets, jitter_targets, random_erase_masks, inference, convert_to_coco, predictions_to_coco, draw_targets
from models.yolo.utils import xywh2xyxy

def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        # print("new  lr", lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr


def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        # print("new  lr", epoch, warmup_length, epoch)
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)


def beta_policy(mom_fn):
    def _alr(optimizer, iteration, epoch, param, indx):
        mom = mom_fn(iteration, epoch)
        # print("new  lr", lr)
        for param_group in optimizer.param_groups:
            param_group[param][indx] = mom

    return _alr


def mom_cosine_policy(base_beta, warmup_length, epochs):
    def _beta_fn(iteration, epoch):
        # print("new  lr", epoch, warmup_length, epoch)
        if epoch < warmup_length:
            beta = base_beta * (epoch + 1) / warmup_length
        else:
            beta = base_beta
        return beta

    return beta_policy(_beta_fn)

class DeepInversionFeatureHook_fullfeatmse():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.cached_feats = 0.0 
        self.cache_batch_stats = False

    def hook_fn(self, module, input, output):
        if self.cache_batch_stats: 
            print("[HOOK] Caching full features")
            self.cached_feats = input[0].clone().detach() 
            self.cache_batch_stats = False
        r_feature = torch.norm(self.cached_feats - input[0], 2) 
        self.r_feature = r_feature
        # should not output anything

    def close(self):
        self.hook.remove()


class DeepInversionFeatureHook():
    def __init__(self, module, p_norm=1, alpha_mean=1.0, alpha_var=1.0):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.cached_var  = None
        self.cached_mean = None
        self.cache_batch_stats = False
        self.diff_mean = None
        self.diff_var  = None
        self.p_norm    = p_norm
        self.alpha_mean, self.alpha_var = alpha_mean, alpha_var
        self.ip_shape, self.op_shape = None, None

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        self.ip_shape, self.op_shape = input[0].shape, output.shape
        if 1:
            #original paper
            #forsing mean and var to match between two distributions
            
            # Cache real batch statistics
            if self.cache_batch_stats: 
                self.cached_mean = mean.clone().detach() 
                self.cached_var  = var.clone().detach()
                self.cache_batch_stats = False

            # Cache the mean/var from bnorm
            if (self.cached_mean is None) and (self.cached_var is None): 
                self.cached_var  = module.running_var.data.clone().detach()     
                self.cached_mean = module.running_mean.data.clone().detach() 

            self.diff_mean  = torch.norm(self.cached_mean.type(mean.dtype) - mean, self.p_norm)
            self.diff_var   = torch.norm(self.cached_var.type(var.dtype)   - var,  self.p_norm)
            r_feature       = self.alpha_mean*self.diff_mean + self.alpha_var*self.diff_var
            # r_feature = r_feature / nch # normalize by the number of channels

        else:
            #probably a better way via minimizing KL divergence between two Gaussians
            #use KL div loss
            #from https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
            var_gt = module.running_var.data
            var_est = var
            mean_gt = module.running_mean.data
            mean_est = mean
            #index 2 - estimated, index 1 - true
            eps = 1e-6
            # import pdb; pdb.set_trace()
            r_feature1 = 0.5*torch.log(var_est/(var_gt+eps)) + (var_gt + (mean_gt-mean_est)**2) /(2*var_est + eps) - 0.5
            r_feature2 = 0.5*torch.log(var_gt/(var_est+eps)) + (var_est + (mean_gt-mean_est)**2) /(2*var_gt + eps) - 0.5
            #KL is always positive
            r_feature = 0.5*r_feature1 + 0.5*r_feature2
            r_feature = torch.clamp(r_feature, min=0.0)
            r_feature = r_feature.sum()/30.0
            # r_feature = r_feature.mean()

        self.r_feature = r_feature
        # should not output anything

    def close(self):
        self.hook.remove()


class DeepInversionFeatureHook_features():
    def __init__(self, module, l1_reg=0.0, l2_reg=0.0):
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization

        # self.feature_loss = torch.zeros(1).half()
        self.feature_loss = torch.zeros(1)

        if self.l1_reg > 0:
            l1_loss = torch.norm(input[0], dim=1, p=1).mean()
            self.feature_loss = l1_loss * self.l1_reg

        if self.l2_reg > 0:
            l2_loss = torch.norm(input[0], dim=1, p=2).mean()
            self.feature_loss = l2_loss * self.l2_reg

        # should not output anything

    def close(self):
        self.hook.remove()

class SelfSimilarityHook():
    def __init__(self, module, threshold=0.7):
        self.ssim = None
        self.hook = module.register_forward_hook(self.hook_fn)
        self.threshold = threshold

    def hook_fn(self, module, input, output):
        features    = torch.nn.functional.adaptive_avg_pool2d(input[0], output_size=1).squeeze()
        embeddings  = features / torch.norm(features, p=2, dim=1, keepdim=True)
        ssim_matrix = torch.matmul(embeddings, embeddings.t())
        ssim_matrix[ssim_matrix<self.threshold] = 0.0
        eye         = torch.eye(ssim_matrix.shape[0], dtype=ssim_matrix.dtype, device=ssim_matrix.device)
        self.ssim   = torch.norm(ssim_matrix - eye, p=2)

    def close(self):
        self.hook.remove()


def clip(image_tensor, use_amp=False):
    '''
    adjust the input based on mean and variance
    '''
    mean = np.array([0.48853, 0.48853, 0.48853])
    std = np.array([0.08215 ** 0.5, 0.08215 ** 0.5, 0.08215 ** 0.5])
    if use_amp:
        mean, std = mean.astype(np.float16), std.astype(np.float16)

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor


def denormalize(image_tensor, use_amp=False):
    '''
    convert floats back to input
    '''
    if use_amp:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor


def get_image_prior_losses(inputs_jit):
    # COMPUTE total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1, loss_var_l2


class DeepInversionClass(object):
    def __init__(self, 
                 net_teacher=None, net_verifier=None, path="./temp/",
                 logger_big=None,
                 parameters=dict(),
                 criterion=None,
                 use_amp=False,
                 coefficients=dict(),
                 network_output_function=lambda x: x):
        '''
        :param bs: batch size per GPU for image generation
        :param use_amp: use APEX AMP for model inversion
        :parameter net_teacher: Pytorch model to be inverted
        :param path: path where to write temporal images and data
        :param data_path: path to write final images into
        :param parameters: a dictionary of control parameters:
            "resolution": input image resolution, single value, assumed to be a square, 224
            "random_label" : for classification initialize target to be random values
            "start_noise" : start from noise, def True
            "detach_student": if computing Adaptive DI, should we detach student?
        :param setting_id: predefined settings for optimization:
            0 - will run low resolution optimization for 1k and then full resolution for 1k;
            1 - will run optimization on high resolution for 2k

        :param jitter: amount of random shift applied to image at every iteration
        :param coefficients: dictionary with parameters and coefficients for optimization.
            keys:
            "r_feature" - coefficient for feature distribution regularization
            "tv_l1" - coefficient for total variation L1 loss
            "tv_l2" - coefficient for total variation L1 loss
            "wd" - weight decay applied in AdamW for optimization
            "lr" - learning rate for optimization
            "first_bn_coef" - additional regularization for the first BN in the networks, coefficient, useful if colors do not match
            "main_loss_multiplier" - coefficient for the main loss optimization
            "cig_scale" - coefficient for adaptive DeepInversion, competition, def =0 means no competition

        network_output_function: function to be applied to the output of the network to get output

        hook_for_display: function to be executed at every print/save execution, useful to check accuracy

        '''

        
        print("Deep inversion class generation")

        self.net_teacher = net_teacher
        self.net_verifier = net_verifier

        self.image_resolution = parameters["resolution"]
        self.random_label = parameters["random_label"]
        self.start_noise = parameters["start_noise"]
        self.do_flip = parameters["do_flip"]

        self.bs = parameters["bs"]  # batch size
        self.iterations = parameters["iterations"]
        self.save_every = parameters["save_every"]
        self.display_every = parameters["display_every"]
        self.beta1 = parameters["beta1"]
        self.beta2 = parameters["beta2"]
        self.nms_params = parameters["nms_params"]
        self.cosine_layer_decay = parameters['cosine_layer_decay']
        self.min_layers =  parameters['min_layers']
        self.num_layers =  parameters["num_layers"]
        self.p_norm     =  parameters['p_norm']
        self.alpha_mean =  parameters['alpha_mean']
        self.alpha_var  =  parameters['alpha_var']
        self.alpha_ssim =  parameters["alpha_ssim"]

        self.l1_reg = 0.0
        self.l2_reg = 0.0

        self.jitter = parameters["jitter"]
        self.rand_brightness = parameters["rand_brightness"]
        self.rand_contrast = parameters["rand_contrast"]
        self.random_erase  = parameters["random_erase"]
        self.mean_var_clip = parameters["mean_var_clip"]
        self.criterion = criterion

        self.network_output_function = network_output_function

        self.bn_reg_scale = coefficients["r_feature"]
        self.var_scale_l1 = coefficients["tv_l1"]
        self.var_scale_l2 = coefficients["tv_l2"]
        self.wd_coeff = coefficients["wd"]
        self.lr = coefficients["lr"]
        self.min_lr = coefficients["min_lr"]
        self.first_bn_coef = coefficients["first_bn_coef"]
        self.main_loss_multiplier = coefficients["main_loss_multiplier"]
        self.alpha_img_stats = coefficients["alpha_img_stats"]
        self.use_amp = use_amp
        self.path = path

        # Bounding Box sampler
        self.box_sampler        = parameters["box_sampler"]
        self.box_sampler_warmup = parameters["box_sampler_warmup"]
        self.box_sampler_conf   = parameters["box_sampler_conf"]
        self.box_sampler_overlap_iou = parameters["box_sampler_overlap_iou"]
        self.box_sampler_minarea= parameters["box_sampler_minarea"]
        self.box_sampler_maxarea= parameters["box_sampler_maxarea"]
        self.box_sampler_earlyexit = parameters["box_sampler_earlyexit"]

        create_folder(self.path)
        print("Results and logs will be stored at: {}".format(self.path))

        # Write parameters + coefficients to disk 
        with open(os.path.join(self.path, "parameters.json"), "wt") as fp: 
            json.dump(parameters, fp)
        with open(os.path.join(self.path, "coefficients.json"), "wt") as fp: 
            json.dump(coefficients, fp)

        # Add hooks for Batchnorm layers
        self.loss_r_feature_layers = []
        for module_group in self.net_teacher.module_list[0:74]:
            if isinstance(module_group, nn.Sequential):
                self.loss_r_feature_layers.append(DeepInversionFeatureHook(module_group.BatchNorm2d, self.p_norm, self.alpha_mean, self.alpha_var))
        print("num layers: {}".format(len(self.loss_r_feature_layers)))

        # Add hook for self-similarity
        self.ssim_hook = SelfSimilarityHook(net_teacher.module_list[53][0], threshold=0.0)
        
        # Logging 
        self.writer = SummaryWriter(os.path.join(self.path, "logs"))
        self.txtwriter = open(os.path.join(self.path, "losses.log"), "wt")
        print("tboard writer in {}".format(self.writer.logdir))
        print("Text   writer in {}".format(self.txtwriter.name))
    
    def __del__(self):
        # destructor
        self.txtwriter.close()
        self.writer.close()

    def get_images(self, targets, init):

        net_teacher = self.net_teacher
        net_teacher.eval()
        save_every = self.save_every
        best_cost = 1e4
        criterion = self.criterion

        # Setup input (which will be optimized)
        gpu_device = torch.device("cuda:0")
        inputs = init.clone().detach().to(gpu_device).requires_grad_(True)
        targets = targets.to(gpu_device)
        print("Inputs shape: {} Targets shape: {}".format(inputs.shape, targets.shape))
        

        optimizer = optim.Adam([inputs], lr=self.lr, betas=[self.beta1, self.beta2], weight_decay=self.wd_coeff)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.iterations, eta_min=self.min_lr)
        # lr_scheduler = lr_cosine_policy(self.lr, 10, self.iterations)
        # beta0_scheduler = mom_cosine_policy(0.9, 250, self.iterations)
        # beta1_scheduler = mom_cosine_policy(0.999, 250, self.iterations)
        if self.use_amp:
            net_teacher, optimizer = amp.initialize(net_teacher, optimizer, opt_level="O1")
            self.net_verifier, _   = amp.initialize(self.net_verifier, [], opt_level="O1")

        if self.box_sampler:
            print("Fp sampler is enabled, targets will be updated with high confidence non-overlapping predictions")

        layer_wise_rfeat, layer_wise_mean, layer_wise_var = [], [], []
        for iteration in range(1,self.iterations+1):
            
            scheduler.step()
            # lr_scheduler(optimizer, iteration, iteration)

            inputs_jit = torch.ones_like(inputs) * inputs
            targets_jit = targets.clone().detach() 

            # Random Jitter 
            off1, off2 = random.randint(-self.jitter, self.jitter), random.randint(-self.jitter, self.jitter)
            inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))
            if any([off1, off2]):
                height, width = inputs_jit.shape[2], inputs_jit.shape[3]
                targets_jit = jitter_targets(targets_jit, off2, off1, img_shape=(height, width))

            # Random horizontal flips
            flip = random.random() > 0.5
            if flip and self.do_flip:
                inputs_jit = torch.flip(inputs_jit, dims=(3,))
                targets_jit = flip_targets(targets_jit, horizontal=True, vertical=False)
            
            # Random brightness & contrast
            if self.rand_brightness:
                rand_brightness = torch.randn(inputs_jit.shape[0], 1, 1,1).cuda() * 0.2
                inputs_jit += rand_brightness
            if self.rand_contrast:
                rand_contrast   =  1.0 + torch.randn(inputs_jit.shape[0], 1, 1,1).cuda() * 0.1 
                inputs_jit *= rand_contrast
            
            # Random erase mask 
            if self.random_erase:
                masks = random_erase_masks(inputs_jit.shape, return_cuda=True) 
                inputs_jit *= masks

            # foward with jit images
            outputs = net_teacher(inputs_jit)

            outputs = self.network_output_function(outputs)

            # R_cross classification loss NOTE: MODIFIED FOR YOLO
            task_loss, _ = criterion(outputs, targets_jit, net_teacher)
            task_loss_copy = task_loss.clone().detach()
            task_loss = self.main_loss_multiplier * task_loss 

            # R_prior losses
            prior_loss_var_l1, prior_loss_var_l2 = get_image_prior_losses(inputs_jit)
            prior_loss_var_l1_copy = prior_loss_var_l1.clone().detach()
            prior_loss_var_l2_copy = prior_loss_var_l2.clone().detach()
            prior_loss_var_l1 = self.var_scale_l1 * prior_loss_var_l1 
            prior_loss_var_l2 = self.var_scale_l2 * prior_loss_var_l2 

            # R_feature loss (w/ cosine decay in optimizing number of layers)
            numLayers = len(self.loss_r_feature_layers) if self.num_layers==-1 else self.num_layers
            if self.cosine_layer_decay:
                _cos      = math.cos((iteration / self.iterations) * (math.pi / 2.0)) # 1.0 --> 0.0
                numLayers = int(_cos * numLayers) # numLayers --> 0
                numLayers = max(self.min_layers, numLayers)
            loss_r_feature = torch.sum(torch.stack([mod.r_feature for mod in self.loss_r_feature_layers[0:numLayers]]))
            loss_r_feature_copy = loss_r_feature.clone().detach()
            loss_r_feature = self.bn_reg_scale * loss_r_feature 

            # layer wise losses
            layer_wise_rfeat.append([mod.r_feature.item() for mod in self.loss_r_feature_layers])
            layer_wise_mean.append([mod.diff_mean.item() for mod in self.loss_r_feature_layers])
            layer_wise_var.append([mod.diff_var.item() for mod in self.loss_r_feature_layers])

            # R_feature loss layer_1
            loss_r_feature_first = sum([mod.r_feature for mod in self.loss_r_feature_layers[:1]])
            loss_r_feature_first_copy = loss_r_feature_first.clone().detach()
            loss_r_feature_first = self.first_bn_coef * loss_r_feature_first 

            # Self similarity
            ssim_loss = self.alpha_ssim * self.ssim_hook.ssim

            # # Match image stats
            # img_mean = inputs_jit.mean([2, 3])
            # img_std = inputs_jit.std([2,3])
            # # print(img_mean.mean([0]), img_std.mean([0]))
            # natural_image_mean = torch.tensor([0.51965, 0.49869, 0.44715]).cuda()
            # natural_image_std = torch.tensor([0.24281, 0.23689, 0.24182]).cuda()
            # loss_img_stats = torch.norm(img_mean - natural_image_mean, 2, dim=1).mean() + \
            #                              torch.norm(img_std - natural_image_std, 2, dim=1).mean()
            # loss_img_stats_copy = loss_img_stats.clone().detach()
            # loss_img_stats *= self.alpha_img_stats

            total_loss = task_loss + prior_loss_var_l1 + prior_loss_var_l2 + loss_r_feature + loss_r_feature_first + ssim_loss # + loss_img_stats

            # To check if weight decay is working 
            inputs_norm = torch.norm(inputs) / inputs.shape[0] 

            # do image update
            optimizer.zero_grad()
            net_teacher.zero_grad()
            if self.use_amp:
                with amp.scale_loss(total_loss, optimizer) as total_loss_scaled:
                    total_loss_scaled.backward()
            else:
                total_loss.backward()
            optimizer.step()

            with torch.no_grad(): # projected g.d. (must be separated from backprop graph)
                inputs.clamp_(min=0.0, max=1.0)
            if self.mean_var_clip:
                inputs.data = clip(inputs.data, use_amp=self.use_amp)

            # Write logs to tensorboard

            # Weighted Loss
            self.writer.add_scalar("weighted/total_loss", total_loss.item(), iteration)
            self.writer.add_scalar("weighted/task_loss", task_loss.item(), iteration)
            self.writer.add_scalar("weighted/prior_loss_var_l1", prior_loss_var_l1.item(), iteration)
            self.writer.add_scalar("weighted/prior_loss_var_l2", prior_loss_var_l2.item(), iteration)
            self.writer.add_scalar("weighted/loss_r_feature", loss_r_feature.item(), iteration)
            self.writer.add_scalar("weighted/loss_r_feature_first", loss_r_feature_first.item(), iteration)
            # self.writer.add_scalar("weighted/loss_img_stats", loss_img_stats.item(), iteration)
            # Unweighted loss
            self.writer.add_scalar("unweighted/task_loss", task_loss_copy.item() , iteration)
            self.writer.add_scalar("unweighted/prior_loss_var_l1", prior_loss_var_l1_copy.item() , iteration)
            self.writer.add_scalar("unweighted/prior_loss_var_l2", prior_loss_var_l2_copy.item() , iteration)
            self.writer.add_scalar("unweighted/loss_r_feature", loss_r_feature_copy.item() , iteration)
            self.writer.add_scalar("unweighted/loss_r_feature_first", loss_r_feature_first_copy.item() , iteration) 
            # self.writer.add_scalar("unweighted/loss_img_stats", loss_img_stats_copy.item(), iteration)
            # self.writer.add_scalar("unweighted/loss_r_feature_L0", self.loss_r_feature_layers[0].r_feature.item(), iteration)
            # self.writer.add_scalar("unweighted/loss_r_feature_L1", self.loss_r_feature_layers[1].r_feature.item(), iteration)
            # self.writer.add_scalar("unweighted/loss_r_feature_L2", self.loss_r_feature_layers[2].r_feature.item(), iteration)
            self.writer.add_scalar("unweighted/inputs_norm", inputs_norm.item(), iteration) 
            self.writer.add_scalar("learning_rate", float(optimizer.param_groups[0]["lr"]), iteration)
        
            # Write logs to txt file 
            # weighted
            self.txtwriter.write("-"*50 + "\n")
            self.txtwriter.write("ITERATION: {}\n".format(iteration))
            self.txtwriter.write("weighted/total_loss {}\n".format(total_loss.item()))
            self.txtwriter.write("weighted/task_loss {}\n".format(task_loss.item())) 
            self.txtwriter.write("weighted/prior_loss_var_l1 {}\n".format(prior_loss_var_l1.item()))
            self.txtwriter.write("weighted/prior_loss_var_l2 {}\n".format(prior_loss_var_l2.item()))
            self.txtwriter.write("weighted/loss_r_feature {}\n".format(loss_r_feature.item()))
            self.txtwriter.write("weighted/loss_r_feature_first {}\n".format(loss_r_feature_first.item()))
            # self.txtwriter.write("weighted/loss_img_stats {}\n".format(loss_img_stats.item()))
            # unweighted
            self.txtwriter.write("unweighted/task_loss {}\n".format(task_loss_copy.item() )) 
            self.txtwriter.write("unweighted/prior_loss_var_l1 {}\n".format(prior_loss_var_l1_copy.item() ))
            self.txtwriter.write("unweighted/prior_loss_var_l2 {}\n".format(prior_loss_var_l2_copy.item() ))
            self.txtwriter.write("unweighted/loss_r_feature {}\n".format(loss_r_feature_copy.item() ))
            self.txtwriter.write("unweighted/loss_r_feature_first {}\n".format(loss_r_feature_first_copy.item() )) 
            # self.txtwriter.write("unweighted/loss_img_stats {}\n".format(loss_img_stats_copy.item()))
            # self.txtwriter.write("unweighted/loss_r_feature_L0 {}\n".format( self.loss_r_feature_layers[0].r_feature.item()))
            # self.txtwriter.write("unweighted/loss_r_feature_L1 {}\n".format( self.loss_r_feature_layers[1].r_feature.item()))
            # self.txtwriter.write("unweighted/loss_r_feature_L2 {}\n".format( self.loss_r_feature_layers[2].r_feature.item()))
            self.txtwriter.write("unweighted/inputs_norm {}\n".format(inputs_norm.item()))
            self.txtwriter.write("learning_Rate {}\n".format(float(optimizer.param_groups[0]["lr"])))

            if (iteration % self.display_every) == 0:
                print("Iteration: {}".format(iteration))
                print("[WEIGHTED] total loss", total_loss.item())
                print("[WEIGHTED] task_loss", task_loss.item())
                print("[WEIGHTED] prior_loss_var_l1: ", prior_loss_var_l1.item())
                print("[WEIGHTED] prior_loss_var_l2: ", prior_loss_var_l2.item())
                print("[WEIGHTED] loss_r_feature", loss_r_feature.item())
                print("[WEIGHTED] loss_r_feature_first", loss_r_feature_first.item())
                # print("[UNWEIGHTED] loss_img_stats", loss_img_stats_copy.item())
                print("[UNWEIGHTED] inputs_norm", inputs_norm.item())

            # Save to disk
            if (iteration % self.save_every) == 0: 
                im_copy = inputs.clone().detach().cpu()

                # compute metrics (mp, mr, map, mf1) for the updated image on net_verifier
                mPrec, mRec, mAP, mF1, im_boxes_verif, _ = inference(self.net_verifier, inputs, targets, self.nms_params)
                self.writer.add_scalar("unweighted/mAP VERIFIER", float(mAP), iteration)
                self.writer.add_scalar("unweighted/mF1 VERIFIER", float(mF1), iteration)
                self.writer.add_scalar("unweighted/mPrec VERIFIER", float(mPrec), iteration)
                self.writer.add_scalar("unweighted/mRec VERIFIER", float(mRec), iteration)
                self.txtwriter.write("Verifier InvImage mPrec: {:.4} mRec: {:.4} mAP: {:.4} mF1: {:.4} \n".format(mPrec, mRec, mAP, mF1))
                print("[UNWEIGHTED] mAP VERIFIER {:.4}".format(mAP))

                # compute metrics (mp, mr, map, mf1) for the updated image on net_teacher
                mPrec, mRec, mAP, mF1, im_boxes_teach, teacher_output = inference(self.net_teacher, inputs, targets, self.nms_params)
                self.writer.add_scalar("unweighted/mAP TEACHER", float(mAP), iteration)
                self.writer.add_scalar("unweighted/mF1 TEACHER", float(mF1), iteration)
                self.writer.add_scalar("unweighted/mPrec TEACHER", float(mPrec), iteration)
                self.writer.add_scalar("unweighted/mRec TEACHER", float(mRec), iteration)
                self.txtwriter.write("Teacher InvImage mPrec: {:.4} mRec: {:.4} mAP: {:.4} mF1: {:.4} \n".format(mPrec, mRec, mAP, mF1))
                print("[UNWEIGHTED] mAP TEACHER {:.4}".format(mAP))

                # Uncomment to save batch overlayed with teacher/verifier predictions
                # self.save_image(
                #     batch_tens =im_boxes_verif,
                #     loc   = os.path.join(self.path, "iteration_verifier_{}.jpg".format(iteration)),
                #     halfsize=False
                # )
                # self.save_image(
                #     batch_tens =im_boxes_teach,
                #     loc   = os.path.join(self.path, "iteration_teacher_{}.jpg".format(iteration)),
                #     halfsize=False
                # )


                # FP sampling
                if self.box_sampler and (iteration >= self.box_sampler_warmup) and (iteration<=self.box_sampler_earlyexit):
                    new_targets = predictions_to_coco(teacher_output, im_copy)
                    new_targets = new_targets[new_targets[:,2] > self.box_sampler_conf] # filter targets by confidence threshold
                    new_targets = torch.index_select(new_targets, dim=1, index=torch.tensor([0,1,3,4,5,6])) # # remove confidence value

                    to_add = torch.zeros((len(new_targets),), dtype=torch.long).cuda()
                    batch_size = im_copy.shape[0]
                    for batchIdx in range(batch_size):
                        _targets = targets[targets[:,0]==batchIdx]
                        _new_targets = new_targets[new_targets[:,0]==batchIdx]
                        if _new_targets.shape[0] > 0:

                            ious = torchvision.ops.box_iou(
                                xywh2xyxy(_new_targets[:,2:].cuda()),
                                xywh2xyxy(_targets[:,2:])
                            )
                            max_ious, _ = torch.max(ious, dim=1)
                            _to_add     = (max_ious < self.box_sampler_overlap_iou).long() # condition: if pred has <0.2 overlap w/ any gt box, add to targets
                            to_add[new_targets[:,0]==batchIdx] += _to_add

                    new_targets = new_targets[to_add.bool()]
                    assert len(new_targets) == to_add.sum().item()
                    # filter by area
                    areas = new_targets[:,-1] * new_targets[:,-2]
                    area_limit_idcs = (areas < self.box_sampler_maxarea) * (areas > self.box_sampler_minarea)
                    new_targets = new_targets[area_limit_idcs.bool()]
                    print("Fp sampling: Adding {} new targets to batch for iteration: {} ".format(len(new_targets), iteration))
                    targets = torch.cat((targets, new_targets.cuda()), dim=0)

                # Save batch overlayed with provided and fp-sampled targets
                imgs_with_boxes_targets  = draw_targets(im_copy, targets)
                self.save_image(imgs_with_boxes_targets,os.path.join(self.path, "iteration_targets_{}.jpg".format(iteration)), halfsize=False)

                del im_copy, im_boxes_teach, im_boxes_verif, imgs_with_boxes_targets
                torch.cuda.empty_cache()

            if self.box_sampler and iteration>=self.box_sampler_earlyexit:
                print("early exit on {} iteration".format(iteration))
                break

        # Save tracked mean/std
        tracker_dict = {
            "mean"  : torch.tensor(layer_wise_mean),
            "var"   : torch.tensor(layer_wise_var),
            "rfeat" : torch.tensor(layer_wise_rfeat),
            "input_shape" : [mod.ip_shape for mod in self.loss_r_feature_layers],
            "output_shape" : [mod.op_shape for mod in self.loss_r_feature_layers]
        }
        torch.save(tracker_dict, os.path.join(self.path, "tracker.data"))

        return inputs.clone().detach().cpu(), targets.clone().detach().cpu()

    def save_image(self, batch_tens, loc, halfsize=True): 
        """
        Saves a batch_tens of images to location loc
        """ 
        print("Saving batch_tensor of shape {} to location: {}".format(batch_tens.shape, loc))
        vutils.save_image(batch_tens, loc, normalize=False)

    def generate_batch(self, targets, init):
        self.net_teacher.eval()
        generatedImages = self.get_images(targets, init)
        return generatedImages

    def cache_batch_stats(self, imgs): 
        """
        Replace the cached_mean and cached_var with statistics from real data
        """
        self.net_teacher.eval() 
        self.net_verifier.eval()
        # Enable caching of real img batch stats
        for bnorm_hook in self.loss_r_feature_layers:
            assert bnorm_hook.cache_batch_stats == False
            bnorm_hook.cache_batch_stats = True
        # Cache
        with torch.no_grad():
            _preds = self.net_teacher(imgs)
        del _preds
        # Verify 
        for bnorm_hook in self.loss_r_feature_layers:
            assert bnorm_hook.cache_batch_stats == False
            assert bnorm_hook.cached_mean is not None
            assert bnorm_hook.cached_var  is not None

            # The code below is useful for full-feature mse loss instead of batch-norm regularization loss
            # assert isinstance(bnorm_hook.cached_feats, torch.cuda.FloatTensor) \
            # , "Not a TENSOR! Found {}".format(type(bnorm_hook.cached_feats))
            # assert bnorm_hook.cached_feats.shape[0] == self.bs



        
            


