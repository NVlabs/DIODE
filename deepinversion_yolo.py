import torch
import torch.nn as nn
import torch.optim as optim
import collections
from utils_di import create_folder, Logger
import random, shutil
import torch
import torchvision.utils as vutils
from PIL import Image

import numpy as np
from tensorboardX import SummaryWriter
import os, sys, json
from models.yolo.yolostuff import run_inference, calculate_metrics, flip_targets, jitter_targets, random_erase_masks


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
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.cached_var = None 
        self.cached_mean = None
        self.cache_batch_stats = False

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        std = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        if 1:
            #original paper
            #forsing mean and variance to match between two distributions
            
            # Cache real batch statistics
            if self.cache_batch_stats: 
                self.cached_mean = mean.clone().detach() 
                self.cached_var  = std.clone().detach()
                self.cache_batch_stats = False

            # Cache the mean/var from bnorm 
            if (self.cached_mean is None) and (self.cached_var is None): 
                self.cached_var  = module.running_var.data.clone().detach()     
                self.cached_mean = module.running_mean.data.clone().detach() 

            r_feature = torch.norm(self.cached_var - std, 1) + torch.norm(
                self.cached_mean - mean, 1)
            # r_feature = torch.norm(module.running_var.data - std, 2) + torch.norm(
            #     module.running_mean.data - mean, 2)
        else:
            #probably a better way via minimizing KL divergence between two Gaussians
            #use KL div loss
            #from https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
            var_gt = module.running_var.data
            var_est = std
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


def clip(image_tensor, use_amp=False):
    '''
    adjust the input based on mean and variance
    '''
    if use_amp:
        raise NotImplementedError
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.48853, 0.48853, 0.48853])
        std = np.array([0.08215 ** 0.5, 0.08215 ** 0.5, 0.08215 ** 0.5])

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
                 setting_id=0,
                 criterion=None,
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

        # for reproducibility
        # torch.manual_seed(torch.cuda.current_device())
        self.net_teacher = net_teacher
        self.net_verifier = net_verifier
        # self.net_teacher = nn.DataParallel(net_teacher, device_ids=[0,1,2], output_device=3)

        if "resolution" in parameters.keys():
            self.image_resolution = parameters["resolution"]
            self.random_label = parameters["random_label"]
            self.start_noise = parameters["start_noise"]
            self.do_flip = parameters["do_flip"]
        else:
            self.image_resolution = (608,960)
            self.random_label = False
            self.start_noise = True
            self.do_flip = False

        self.setting_id = setting_id

        self.bs = parameters["bs"]  # batch size
        self.iterations = parameters["iterations"]
        self.save_every = parameters["save_every"]
        self.display_every = parameters["display_every"]
        self.beta1 = parameters["beta1"]
        self.beta2 = parameters["beta2"]

        self.l1_reg = 0.0
        self.l2_reg = 0.0

        self.jitter = parameters["jitter"]
        self.rand_brightness = parameters["rand_brightness"]
        self.rand_contrast = parameters["rand_contrast"]
        self.random_erase  = parameters["random_erase"]
        self.mean_var_clip = parameters["mean_var_clip"]
        self.criterion = criterion

        self.network_output_function = network_output_function

        if "r_feature" in coefficients:
            self.bn_reg_scale = coefficients["r_feature"]
            self.var_scale_l1 = coefficients["tv_l1"]
            self.var_scale_l2 = coefficients["tv_l2"]
            self.wd_coeff = coefficients["wd"]
            self.lr = coefficients["lr"]
            self.first_bn_coef = coefficients["first_bn_coef"]
            self.main_loss_multiplier = coefficients["main_loss_multiplier"]
            self.alpha_img_stats = coefficients["alpha_img_stats"]
        else:
            print("Provide a dictionary with ")

        self.num_generations = 0

        local_rank = torch.cuda.current_device()
        
        # Log to tmp if on NGC
        if "NGC_JOB_ID" in os.environ: 
            print("RUNNING IN NGC .. WRITING LOGS TO /TMP")
            basename = os.path.basename(path) 
            self.path = os.path.join("/tmp", basename)
            self.origpath = path
        else:
            print("NOT RUNNING IN NGC")
            self.path = path

        create_folder(self.path)
        print("Results and logs will be stored at: {}".format(self.path))

        # Write parameters + coefficients to disk 
        with open(os.path.join(self.path, "parameters.json"), "wt") as fp: 
            json.dump(parameters, fp)
        with open(os.path.join(self.path, "coefficients.json"), "wt") as fp: 
            json.dump(coefficients, fp)

        
        self.loss_r_feature_layers = []

        # Add hooks for Batchnorm layers
        for module in self.net_teacher.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.loss_r_feature_layers.append(DeepInversionFeatureHook(module))
        # for module in self.net_teacher.modules():
        #     if isinstance(module, nn.BatchNorm2d):
        #         self.loss_r_feature_layers.append(DeepInversionFeatureHook_fullfeatmse(module))
        
        # Logging 
        self.writer = SummaryWriter(os.path.join(self.path, "logs"))
        self.txtwriter = open(os.path.join(self.path, "losses.log"), "wt")
        print("tboard writer in {}".format(self.writer.logdir))
        print("Text   writer in {}".format(self.txtwriter.name))
    

    def get_images(self, targets, init):
        
        print("get_images call")

        net_teacher = self.net_teacher
        net_teacher.eval()
        save_every = self.save_every
        local_rank = torch.cuda.current_device()
        best_cost = 1e4
        # set up criteria
        criterion = self.criterion

        # setup target labels
        img_original = self.image_resolution
        

        # Setup input (which will be optimized)
        gpu_device = torch.device("cuda:0")
        inputs = init.clone().detach().to(gpu_device).requires_grad_(True)
        targets = targets.to(gpu_device)
        print(inputs.shape, targets.shape)
        
        poolingmod = nn.modules.pooling.AvgPool2d(kernel_size=2)

        if self.setting_id==0:
            skipfirst = False
        else:
            skipfirst = True

        iteration = 0
        for lr_it, lower_res in enumerate([2, 1]):
            if lr_it==0:
                iterations_per_layer = self.iterations
                # iterations_per_layer = 1000
            else:
                if not skipfirst:
                    iterations_per_layer = self.iterations // 2 
                else:
                    iterations_per_layer = self.iterations

            if lr_it==0 and skipfirst:
                continue

            lim_0, lim_1 = self.jitter // lower_res, self.jitter // lower_res
            # img_res = img_original // lower_res

            optimizer = optim.Adam([inputs], lr=self.lr, betas=[self.beta1, self.beta2], weight_decay=self.wd_coeff)
            # optimizer = optim.SGD([inputs], lr=self.lr)
            # optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.0, 0.0])
            ## optimization is the weakest point of algorithm. Sometimes, settings betas to zero helps optimization
            ## also there are beta schedulers latter.
            # optimizer = optim.AdamW([inputs], lr=self.lr, betas=[0.0, 0.0], weight_decay=self.wd_coeff)

            lr_scheduler = lr_cosine_policy(self.lr, 100, iterations_per_layer)

            # beta0_scheduler = mom_cosine_policy(0.9, 250, iterations_per_layer)
            # beta1_scheduler = mom_cosine_policy(0.999, 250, iterations_per_layer)

            for iteration_loc in range(iterations_per_layer):
                
                iteration += 1
                # print("iteration: {}".format(iteration))
                # apply random jitter offsets
                off1 = random.randint(-lim_0, lim_0)
                off2 = random.randint(-lim_1, lim_1)

                lr_scheduler(optimizer, iteration_loc, iteration_loc)
                # beta0_scheduler(optimizer, iteration_loc, iteration_loc, "betas", 0)
                # beta1_scheduler(optimizer, iteration_loc, iteration_loc, "betas", 1)

                if self.mean_var_clip:
                    inputs.data = clip(inputs.data, use_amp=False)

                if lower_res!=1:
                    inputs_jit = poolingmod(inputs)
                else:
                    inputs_jit = inputs 

                targets_jit = targets.clone().detach() 
                inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))
                if any([off1, off2]):
                    height, width = inputs_jit.shape[2], inputs_jit.shape[3]
                    targets_jit = jitter_targets(targets_jit, off2, off1, img_shape=(width, height))

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
                inputs_jit = torch.clamp(inputs_jit, min=0.0, max=1.0)
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

                # R_feature loss 
                loss_r_feature = sum([mod.r_feature for mod in self.loss_r_feature_layers])
                loss_r_feature_copy = loss_r_feature.clone().detach()
                loss_r_feature = self.bn_reg_scale * loss_r_feature 
                
                # R_feature loss layer_1
                loss_r_feature_first = sum([mod.r_feature for mod in self.loss_r_feature_layers[:1]])
                loss_r_feature_first_copy = loss_r_feature_first.clone().detach()
                loss_r_feature_first = self.first_bn_coef * loss_r_feature_first 

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

                total_loss = task_loss + prior_loss_var_l1 + prior_loss_var_l2 + loss_r_feature + loss_r_feature_first # + loss_img_stats

                # To check if weight decay is working 
                inputs_norm = torch.norm(inputs) / inputs.shape[0] 

                # do image update
                optimizer.zero_grad()
                net_teacher.zero_grad()
                total_loss.backward() 
                optimizer.step()
                
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
                    _, _, mean_ap, mean_f1 = calculate_metrics(self.net_verifier, inputs, targets)
                    self.writer.add_scalar("unweighted/mAP VERIFIER", float(mean_ap), iteration)
                    self.writer.add_scalar("unweighted/mF1 VERIFIER", float(mean_f1), iteration)
                    self.txtwriter.write("unweighted/mAP VERIFIER {}\n".format(float(mean_ap)))
                    self.txtwriter.write("unweighted/mF1 VERIFIER {}\n".format(float(mean_f1)))
                    print("[UNWEIGHTED] mAP VERIFIER {}".format(mean_ap))
                    print("[UNWEIGHTED] mF1 VERIFIER {}".format(mean_f1))

                    # compute metrics (mp, mr, map, mf1) for the updated image on net_teacher
                    _, _, mean_ap, mean_f1 = calculate_metrics(self.net_teacher, inputs, targets)
                    self.writer.add_scalar("unweighted/mAP TEACHER", float(mean_ap), iteration)
                    self.writer.add_scalar("unweighted/mF1 TEACHER", float(mean_f1), iteration)
                    self.txtwriter.write("unweighted/mAP TEACHER {}\n".format(float(mean_ap)))
                    self.txtwriter.write("unweighted/mF1 TEACHER {}\n".format(float(mean_f1)))
                    print("[UNWEIGHTED] mAP TEACHER {}".format(mean_ap))
                    print("[UNWEIGHTED] mF1 TEACHER {}".format(mean_f1))

                    im_boxes_teacher = run_inference(self.net_teacher, im_copy) # Add bounding boxes to generated images
                    im_boxes_verifier= run_inference(self.net_verifier, im_copy) # Add bounding boxes to generated images
                    # self.save_image(
                    #     batch_tens =im_boxes_teacher,
                    #     loc   = os.path.join(self.path, "iteration_teacher_{}.jpg".format(iteration)), 
                    #     halfsize=False
                    # )
                    self.save_image(
                        batch_tens =im_boxes_verifier,
                        loc   = os.path.join(self.path, "iteration_verifier_{}.jpg".format(iteration)), 
                        halfsize=False
                    )
                    del im_copy, im_boxes_teacher, im_boxes_verifier
                    torch.cuda.empty_cache()

                # optimizer.state = collections.defaultdict(dict)


        # to reduce memory consumption by states of the optimizer
        # optimizer.state = collections.defaultdict(dict)
        self.txtwriter.close()

    def save_image(self, batch_tens, loc, halfsize=True): 
        """
        Saves a batch_tens of images to location loc
        """ 
        print("Saving batch_tensor of shape {} to location: {}".format(batch_tens.shape, loc))
        batch_tens = torch.clamp(batch_tens, min=0.0, max=1.0)
        images = vutils.make_grid(batch_tens, nrow=8)
        images = images.numpy() 
        images = np.transpose(images, axes=(1,2,0))
        
        # Resize image to smaller size (to save memory)
        pilim = Image.fromarray((images*255).astype(np.uint8)) 
        pilim_half = pilim.resize(size=(pilim.size[0]//2, pilim.size[1]//2))
        if halfsize:
            pilim_half.save(loc)
        else: 
            pilim.save(loc)

    def generate_batch(self, targets, init):
        
        self.net_teacher.eval()
        self.get_images(targets, init)
        self.num_generations += 1

        # Copy folder from self.path to self.origpath 
        if "NGC_JOB_ID" in os.environ: 
            shutil.move(self.path, self.origpath)

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



        
            


