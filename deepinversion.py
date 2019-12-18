import torch
import torch.nn as nn
import torch.optim as optim
import collections
from apex.fp16_utils import *
from utils_di import create_folder, Logger
import random
import torch
import torchvision.utils as vutils
from PIL import Image

import numpy as np


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


class DeepInversionFeatureHook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        std = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        if 1:
            #original paper
            #forsing mean and variance to match between two distributions
            r_feature = torch.norm(module.running_var.data - std, 2) + torch.norm(
                module.running_mean.data - mean, 2)
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
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
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
    def __init__(self, bs=84,
                 use_amp=True, net_teacher=None, path="./temp/",
                 data_path="/raid/tmp/images/", logger_big=None,
                 parameters=dict(),
                 setting_id=0,
                 jitter=30,
                 criterion=None,
                 coefficients=dict(),
                 network_output_function=lambda x: x,
                 hook_for_display = None):
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
        torch.manual_seed(torch.cuda.current_device())

        self.net_teacher = net_teacher

        if "resolution" in parameters.keys():
            self.image_resolution = parameters["resolution"]
            self.random_label = parameters["random_label"]
            self.start_noise = parameters["start_noise"]
            self.detach_student = parameters["detach_student"]
            self.do_flip = parameters["do_flip"]
        else:
            self.image_resolution = 224
            self.random_label = False
            self.start_noise = True
            self.detach_student = False
            self.do_flip = True

        self.setting_id = setting_id

        self.bs = bs  # batch size
        self.use_amp = use_amp

        self.save_every = 100

        self.l1_reg = 0.0
        self.l2_reg = 0.0

        self.jitter = jitter
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
            self.cig_scale = coefficients["cig_scale"]
        else:
            print("Provide a dictionary with ")

        self.num_generations = 0

        self.data_path = data_path + "/adi/"

        if self.use_amp:
            opt_level = "O2"  # AMP doesn't work because of backprop through FP16 on BN layers, only manual FP16 conv works
            static_loss_scale = 32768.0
            dynamic_loss_scale = True
            self.static_loss_scale = static_loss_scale
            self.dynamic_loss_scale = dynamic_loss_scale


        prefix = path
        self.prefix = prefix

        local_rank = torch.cuda.current_device()
        if local_rank==0:
            create_folder(prefix)
            create_folder(prefix + "/best_images/")
            create_folder(self.data_path)
            for m in range(1000):
                create_folder(self.data_path + "/s{:03d}".format(m))

        self.loss_r_feature_layers = []

        for module in self.net_teacher.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.loss_r_feature_layers.append(DeepInversionFeatureHook(module))
            # also hook for linear layer if we will add some loss on them
            if isinstance(module, nn.Linear):
                self.loss_feature = DeepInversionFeatureHook_features(module, self.l1_reg, self.l2_reg)

        # done with hooks
        self.logger_big = None

        self.hook_for_display = None
        if hook_for_display is not None:
            self.hook_for_display = hook_for_display

    def get_images(self, net_student=None, targets=None):
        print("get_images call")

        net_teacher = self.net_teacher

        use_amp = self.use_amp
        save_every = self.save_every

        kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

        local_rank = torch.cuda.current_device()

        best_cost = 1e4

        # set up criteria
        # criterion = nn.CrossEntropyLoss()
        # criterion = nn.MSELoss()
        # criterion = nn.L1Loss()
        criterion = self.criterion

        # setup target labels
        if targets is None:
            #only works for classification now, for other tasks need to provide target vector
            targets = torch.LongTensor([random.randint(0, 999) for _ in range(self.bs)]).to('cuda')
            if not self.random_label:
                # targets = [980,]
                targets = [1, 933, 946, 980, 25, 63, 92, 94, 107, 985, 151, 154, 207, 250, 270, 277, 283, 292, 294, 309,
                           311,
                           325, 340, 360, 386, 402, 403, 409, 530, 440, 468, 417, 590, 670, 817, 762, 920, 949, 963,
                           967, 574, 487]
                targets = torch.LongTensor(targets * (int(self.bs / len(targets)))).to('cuda')

        img_original = self.image_resolution
        ##resnet18 self.lr = 0.2 self.bn_reg_scale = 0.02

        save_every = 100
        lower_res = 6

        if use_amp:
            inputs = torch.randn((self.bs, 3, img_original, img_original), requires_grad=True, device='cuda',
                                 dtype=torch.half)
        else:
            inputs = torch.randn((self.bs, 3, img_original, img_original), requires_grad=True, device='cuda',
                                 dtype=torch.float)

        poolingmod = nn.modules.pooling.AvgPool2d(kernel_size=2)

        if self.setting_id==0:
            skipfirst = False
        else:
            skipfirst = True

        iteration = 0
        for lr_it, lower_res in enumerate([2, 1]):
            if lr_it==0:
                iterations_per_layer = 2000
                # iterations_per_layer = 1000
            else:
                if not skipfirst:
                    iterations_per_layer = 1000
                else:
                    iterations_per_layer = 2000

            if lr_it==0 and skipfirst:
                continue

            lim_0, lim_1 = self.jitter // lower_res, self.jitter // lower_res
            img_res = img_original // lower_res

            # optimizer = optim.Adam([inputs], lr=self.lr)
            optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.0, 0.0])
            ## optimization is the weakest point of algorithm. Sometimes, settings betas to zero helps optimization
            ## also there are beta schedulers latter.
            #optimizer = optim.AdamW([inputs], lr=self.lr, betas=[0.0, 0.0], weight_decay=self.wd_coeff)

            if use_amp:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=self.dynamic_loss_scale,
                                           static_loss_scale=self.static_loss_scale)

            lr_scheduler = lr_cosine_policy(self.lr, 10, iterations_per_layer)

            beta0_scheduler = mom_cosine_policy(0.9, 250, iterations_per_layer)
            beta1_scheduler = mom_cosine_policy(0.999, 250, iterations_per_layer)

            for iteration_loc in range(iterations_per_layer):
                iteration += 1
                # apply random jitter offsets
                off1 = random.randint(-lim_0, lim_0)
                off2 = random.randint(-lim_1, lim_1)

                lr_scheduler(optimizer, iteration_loc, iteration_loc)
                # beta0_scheduler(optimizer, iteration_loc, iteration_loc, "betas", 0)
                # beta1_scheduler(optimizer, iteration_loc, iteration_loc, "betas", 1)

                if lower_res!=1:
                    inputs_jit = poolingmod(inputs)
                else:
                    inputs_jit = inputs

                inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))

                flip = random.random() > 0.5
                if flip and self.do_flip:
                    inputs_jit = torch.flip(inputs_jit, dims=(3,))

                # foward with jit images
                optimizer.zero_grad()
                net_teacher.zero_grad()

                outputs = net_teacher(inputs_jit)

                outputs = self.network_output_function(outputs)

                # R_cross classification loss
                loss = criterion(outputs, targets)

                # R_prior losses
                loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs_jit)

                # R_feature loss
                loss_r_feature = sum([mod.r_feature for mod in self.loss_r_feature_layers])

                # R_ADI
                loss_verifier_cig = torch.zeros(1)
                if self.cig_scale!=0.0 or 0:
                    # assume verifier needs to get the same accuracy:
                    if not self.detach_student:
                        if iteration % 100==0:
                            outputs_student = net_student(inputs_jit).detach()
                            # to save time we compute the score only once in a while and keep it in the memory
                    else:
                        outputs_student = net_student(inputs_jit)

                    T = 3.0
                    if 1:
                        T = 3.0
                        # jensen shanon divergence:
                        # another way to force KL between negative probabilities
                        P = nn.functional.softmax(outputs_student / T, dim=1)
                        Q = nn.functional.softmax(outputs / T, dim=1)
                        M = 0.5 * (P + Q)

                        P = torch.clamp(P, 0.01, 0.99)
                        Q = torch.clamp(Q, 0.01, 0.99)
                        M = torch.clamp(M, 0.01, 0.99)
                        eps = 0.0
                        loss_verifier_cig = 0.5 * kl_loss(torch.log(P + eps), M) + 0.5 * kl_loss(torch.log(Q + eps), M)
                        # JS criteria - 0 means full correlation, 1 - means completely different
                        loss_verifier_cig = 1.0 - torch.clamp(loss_verifier_cig, 0.0, 1.0)

                    if 0:
                        loss = loss + abs(self.cig_scale) * loss_verifier_cig
                        if iteration % 100==0 and (save_every > 0):
                            print("cig", (abs(self.cig_scale) * loss_verifier_cig).item())

                    if local_rank==0:
                        if iteration % (save_every)==0:
                            print('loss_verifier_cig', loss_verifier_cig.item(), iteration)

                if self.l1_reg > 0.0 or self.l2_reg > 0.0:
                    feature_loss = self.loss_feature.feature_loss
                else:
                    feature_loss = torch.zeros(1)

                # combining losses
                loss_aux = self.var_scale_l2 * loss_var_l2 + \
                           self.var_scale_l1 * loss_var_l1 + \
                           self.bn_reg_scale * loss_r_feature

                if feature_loss > 0.0:
                    loss_aux += feature_loss

                if self.cig_scale!=0.0:
                    loss_aux += self.cig_scale * loss_verifier_cig

                if self.first_bn_coef > 0.0:
                    loss_aux += self.first_bn_coef * sum([mod.r_feature for mod in self.loss_r_feature_layers[:1]])

                loss = self.main_loss_multiplier * loss + loss_aux

                if local_rank==0:
                    if iteration % save_every==0:
                        print("total loss", loss.item())
                        print("loss_r_feature", loss_r_feature.item())
                        print("main criterion", criterion(outputs, targets).item())
                        print("loss_r_feature_layers: ", [a.r_feature.item() for a in self.loss_r_feature_layers[:3]])

                        if self.hook_for_display is not None:
                            self.hook_for_display(inputs, targets)

                # do image update
                if use_amp:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                optimizer.step()

                # clip color outlayers
                inputs.data = clip(inputs.data, use_amp=use_amp)
                best_inputs = inputs.data.clone()

                # optimizer.state = collections.defaultdict(dict)

                if iteration % save_every==0 and (save_every > 0):
                    if 1 and local_rank==0:
                        # done only once
                        if 1:
                            # for landmarks we need to change the order: inputs[:,[2,1,0],:,:]
                            vutils.save_image(inputs,
                                              '{}/best_images/output_{:05d}_gpu_{}.png'.format(self.prefix,
                                                                                               iteration // save_every,
                                                                                               local_rank),
                                              normalize=True, scale_each=True, nrow=int(10))

                    ##add diusplaying acc for teacher and student models

        if 1 and local_rank==0:
            # for landmarks we need to change the order: inputs[:,[2,1,0],:,:]
            vutils.save_image(inputs,
                              '{}/best_images/fin_output_{:05d}_gpu_{}.png'.format(self.prefix,
                                                                                   self.num_generations,
                                                                                   local_rank),
                              normalize=True, scale_each=True, nrow=int(10))

        best_inputs = denormalize(best_inputs)

        self.save_images(best_inputs, targets)

        # to reduce memory consumption by states of the optimizer
        optimizer.state = collections.defaultdict(dict)

    def save_images(self, images, targets):
        # method to store generated images locally
        local_rank = torch.cuda.current_device()
        for id in range(images.shape[0]):
            class_id = targets[id].item()
            # class_id = 0
            # place_to_store = '{}/images/adi/img_class{:03d}_{:05d}_id{:03d}_gpu_{}.jpg'.format(self.prefix, class_id, self.num_generations, id, local_rank)
            place_to_store = '{}/s{:03d}/img_{:05d}_id{:03d}_gpu_{}_2.jpg'.format(self.data_path, class_id,
                                                                                      self.num_generations, id,
                                                                                      local_rank)
            image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
            pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
            pil_image.save(place_to_store)

    def generate_batch(self, net_student=None, targets=None):
        # for ADI detach student and add put to eval mode
        net_teacher = self.net_teacher

        use_amp = self.use_amp

        if use_amp:
            net_teacher.train()
        else:
            net_teacher.eval()

        if use_amp:
            for module in net_teacher.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval().half()

        # fix net_student
        if not (net_student is None):
            net_student = net_student.eval()

        if self.detach_student:
            if use_amp:
                net_student.train()
            else:
                net_student.eval()

            if use_amp:
                for module in net_student.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval().half()

        if targets is not None:
            # import pdb; pdb.set_trace()
            targets = torch.from_numpy(np.array(targets).squeeze()).cuda()
            if use_amp:
                targets = targets.half()

        self.get_images(net_student=net_student, targets=targets)

        # moev back to eval mode
        net_teacher.eval()

        # del inputs
        # self.optimizer.state = collections.defaultdict(dict)

        self.num_generations += 1
