from __future__ import division, print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.utils.data
from torchvision import datasets, transforms

import argparse
import numpy as np
import os, sys 
import functools
import random
# random.seed(0)
from PIL import Image 

from deepinversion_yolo import DeepInversionClass
from models.yolo.yolostuff import get_model_and_dataloader, run_inference, calculate_metrics, get_verifier

def create_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def run(args):
    # torch.manual_seed(args.local_rank)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    net, (imgs, targets), loss_fun = get_model_and_dataloader(batch_size=args.bs, load_pickled=True)
    net = net.to(device)
    net_verifier = get_verifier()
    net_verifier = net_verifier.to(device)

    net.eval() 
    net_verifier.eval()

    args.start_noise = True

    height = int(imgs.shape[2]) 
    width  = int(imgs.shape[3]) 
    args.resolution = (height, width)

    parameters = dict()
    parameters["resolution"] = args.resolution
    parameters["random_label"] = False
    parameters["start_noise"] = True
    parameters["do_flip"] = args.do_flip
    parameters["jitter"] = args.jitter
    parameters["bs"] = args.bs 
    parameters["iterations"] = args.iterations
    parameters["save_every"] = args.save_every
    parameters["display_every"] = args.display_every

    # criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    # criterion = functools.partial(Triterion, loss_weights={"bbox":0.0, "cov":1.0, "orient":0.0})
    criterion = loss_fun
    # criterion = Triterion(loss_weights={"bbox":1.0, "cov":1.0, "orient":0.0})  

    coefficients = dict()
    coefficients["r_feature"] = args.r_feature
    coefficients["tv_l1"] = args.tv_l1
    coefficients["tv_l2"] = args.tv_l2
    coefficients["wd"] = args.wd
    coefficients["lr"] = args.lr
    coefficients["first_bn_coef"] = args.first_bn_coef
    coefficients["main_loss_multiplier"] = args.main_loss_multiplier

    network_output_function = lambda x: x # Return output from all branches of Yolo model

    DeepInversionEngine = DeepInversionClass(net_teacher=net,
                                             net_verifier=net_verifier,
                                             path=args.path,
                                             logger_big=None,
                                             parameters=parameters,
                                             setting_id=1, # no low-resolutio optim, only original resolution optimization
                                             criterion=criterion,
                                             coefficients = coefficients,
                                             network_output_function = network_output_function)

    # Save the input image to disk
    imgs = imgs.float() / 255.0
    imgs_with_boxes = run_inference(net, imgs)
    imgs_with_boxes_verifier = run_inference(net_verifier, imgs)
    DeepInversionEngine.save_image(imgs_with_boxes, os.path.join(DeepInversionEngine.path, "input_image_teacher.jpg"), halfsize=False)
    DeepInversionEngine.save_image(imgs_with_boxes_verifier, os.path.join(DeepInversionEngine.path, "input_image_verifier.jpg"), halfsize=False)

    # Calculate metrics 
    _, _, mean_ap_verifier, mean_f1_verifier = calculate_metrics(net_verifier, imgs, targets)
    DeepInversionEngine.txtwriter.write("[VERIFIER] Real image mAP: {} \n".format(float(mean_ap_verifier)))
    DeepInversionEngine.txtwriter.write("[VERIFIER] Real image mF1: {} \n".format(float(mean_f1_verifier)))
    _, _, mean_ap_net, mean_f1_net = calculate_metrics(net, imgs, targets)
    DeepInversionEngine.txtwriter.write("[NET] Real image mAP: {} \n".format(float(mean_ap_verifier)))
    DeepInversionEngine.txtwriter.write("[NET] Real image mF1: {} \n".format(float(mean_f1_verifier)))

    # TV value for real data 
    from deepinversion_yolo import get_image_prior_losses 
    with torch.no_grad():
        real_loss_var_l1, real_loss_var_l2 = get_image_prior_losses(imgs) 
    print("Real image l1: {} | l2: {}".format(real_loss_var_l1.item(), real_loss_var_l2.item()))
    DeepInversionEngine.txtwriter.write("Real image l1: {} | l2: {}\n".format(real_loss_var_l1.item(), real_loss_var_l2.item()))
    del real_loss_var_l2, real_loss_var_l1

    # l2 Norm for real data 
    with torch.no_grad():
        real_inputs_norm = torch.norm(imgs) / imgs.shape[0]
    DeepInversionEngine.txtwriter.write("Real image norm: {}\n".format(real_inputs_norm.item()))

    assert imgs.shape[0] == parameters["bs"], "Batchsize of data {} doesn't match batchsize specified in cli {}".format(imgs.shape[0], parameters["bs"])

    # Used cached stats
    if args.cache_batch_stats:
        DeepInversionEngine.cache_batch_stats(imgs.clone().detach().cuda())
        print("Overwriting cached_mean and cached_var with batch stats of real data") 
        DeepInversionEngine.txtwriter.write("[CACHE_BATCH_STATS] Overwriting cached_mean and cached_var with batch stats of real data\n")

    # random initialized inputs 
    init = torch.rand((DeepInversionEngine.bs, 3, DeepInversionEngine.image_resolution[0], DeepInversionEngine.image_resolution[1]), dtype=torch.float)
    init = (args.real_mixin_alpha)*imgs + (1.0-args.real_mixin_alpha)*init

    DeepInversionEngine.generate_batch(targets, init)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', '--rank', type=int, default=0, help='Rank of the current process.')
    parser.add_argument('--no-cuda', action='store_true')

    parser.add_argument('--epochs', default=20000, type=int, help='number of epochs?')
    parser.add_argument('--iterations', default=2000, type=int, help='number of iterations for DI optim')
    parser.add_argument('--bs', default=1, type=int, help='batch size')
    parser.add_argument('--jitter', default=30, type=int, help='jitter')
    parser.add_argument('--comment', default='', type=str, help='batch size')
    parser.add_argument('--arch_name', default='resnet50', type=str, help='model name from torchvision or resnet50v15')

    parser.add_argument('--save_every', type=int, default=100, help='save an image every x iterations')
    parser.add_argument('--display_every', type=int, default=2, help='display the lsses every x iterations')
    parser.add_argument('--path', type=str, default='test', help='where to store experimental data NOT: MUST BE A FOLDER')

    parser.add_argument('--do_flip', action='store_true', help='apply flip for model inversion')
    parser.add_argument('--r_feature', type=float, default=0.05, help='coefficient for feature distribution regularization')
    parser.add_argument('--tv_l1', type=float, default=0.0, help='coefficient for total variation L1 loss')
    parser.add_argument('--tv_l2', type=float, default=0.0001, help='coefficient for total variation L2 loss')
    parser.add_argument('--lr', type=float, default=0.2, help='learning rate for optimization')
    parser.add_argument('--wd', type=float, default=0.01, help='weight decay for optimization')
    parser.add_argument('--first_bn_coef', type=float, default=0.0, help='additional regularization for the first BN in the networks, coefficient, useful if colors do not match')
    parser.add_argument('--main_loss_multiplier', type=float, default=1.0, help=' coefficient for the main loss optimization')
    parser.add_argument("--cache_batch_stats", action="store_true", help="use real image stats instead of bnorm mean/var")
    parser.add_argument("--real_mixin_alpha", type=float, default=0.0, help="how much of real image to mix in with the random initialization")

    args = parser.parse_args()

    print(args)
    torch.backends.cudnn.benchmark = True
    run(args)


if __name__ == '__main__':
    main()

