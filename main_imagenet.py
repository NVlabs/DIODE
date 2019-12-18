from __future__ import division, print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import argparse

import torch
from torch import distributed, nn

import random
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data

from torchvision import datasets, transforms

import numpy as np

from apex.fp16_utils import *

import os

import torchvision.models as models

def load_model_pytorch(model, load_model, model_name='resnet',gpu_n=0):


    print("=> loading checkpoint '{}'".format(load_model))

    SHOULD_I_PRINT = not torch.distributed.is_initialized() or torch.distributed.get_rank()==0

    if 1:
        checkpoint = torch.load(load_model, map_location = lambda storage, loc: storage.cuda(gpu_n))

    if 1:
        if 'state_dict' in checkpoint.keys():
            load_from = checkpoint['state_dict']
        else:
            load_from = checkpoint

    # match_dictionaries, useful if loading model without gate:
    if 1:
        if 'module.' in list(model.state_dict().keys())[0]:
            if 'module.' not in list(load_from.keys())[0]:
                from collections import OrderedDict

                load_from = OrderedDict([("module.{}".format(k), v) for k, v in load_from.items()])

        if 'module.' not in list(model.state_dict().keys())[0]:
            if 'module.' in list(load_from.keys())[0]:
                from collections import OrderedDict

                load_from = OrderedDict([(k.replace("module.", ""), v) for k, v in load_from.items()])

    # just for vgg
    if 1:
        if model_name == "vgg":
            from collections import OrderedDict

            load_from = OrderedDict([(k.replace("features.", "features"), v) for k, v in load_from.items()])
            load_from = OrderedDict([(k.replace("classifier.", "classifier"), v) for k, v in load_from.items()])

    if 1:
        if list(load_from.items())[0][0][:2] == "1." and list(model.state_dict().items())[0][0][:2] != "1.":
            load_from = OrderedDict([(k[2:], v) for k, v in load_from.items()])


    if SHOULD_I_PRINT and 0:
        for ind, (key, item) in enumerate(model.state_dict().items()):
            if ind > 10:
                continue
            print(key, model.state_dict()[key].shape)

        print("*********")

        for ind, (key, item) in enumerate(load_from.items()):
            if ind > 10:
                continue
            print(key, load_from[key].shape)

    if 1:
        for key, item in model.state_dict().items():
            # if we add gate that is not in the saved file
            if key not in load_from:
                load_from[key] = item
            # if load pretrined model
            if load_from[key].shape != item.shape:
                load_from[key] = item

    model.load_state_dict(load_from, strict=False)

    # del checkpoint

    epoch_from = -1
    if 'epoch' in checkpoint.keys():
        epoch_from = checkpoint['epoch']
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(load_model, epoch_from))

def create_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


random.seed(0)

def distributed_is_initialized():
    if distributed.is_available():
        if distributed.is_initialized():
            return True
    return False

def validate_one(input, target, model):
    """Perform validation on the validation set"""

    def accuracy(output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    with torch.no_grad():
        output = model(input)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

    print("Verifier accuracy: ", prec1.item())


def run(args):
    torch.manual_seed(args.local_rank)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    # net = resnet50(skip_gate=False)

    if args.arch_name == "resnet50v15":
        from models.resnetv15 import build_resnet
        net = build_resnet("resnet50", "classic")
    else:
        print("loading another model")
        net = models.__dict__[args.arch_name](pretrained=True)

    net = net.to(device)

    use_amp = args.fp16
    if use_amp:
        net = network_to_half(net)

    print('==> Resuming from checkpoint..')

    ### load code
    if args.arch_name=="resnet50v15":
        path_to_model = "./models/resnet50v15/model_best.pth.tar"
        load_model_pytorch(net, path_to_model, model_name='resnet', gpu_n=torch.cuda.current_device())

    ### load code
    print('==> Getting BN params')
    layer_index = 0
    noise_params_layers = list()
    for module in net.modules():
        if isinstance(module, nn.BatchNorm2d):
            mean, std = module.running_mean.data, module.running_var.data
            if use_amp:
                noise_params_layers.append((std.type(torch.float16), mean.type(torch.float16)))
            else:
                noise_params_layers.append((std, mean))
            layer_index += 1


    if distributed_is_initialized():
        net.to(device)
    else:
        net.to(device)


    if use_amp:
        net.train()
    else:
        net.eval()

    if use_amp:
        for module in net.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval().half()

    # reserved to compute test accuracy on generated images by different networks
    net_verifier = None
    if args.verifier and args.cig_scale == 0:
        # if args.local_rank == 1:
            # args.verifier_arch = "resnet18"
            # net_verifier = models.__dict__[args.verifier_arch](pretrained=True).to(device)
            # net_verifier.eval()

        if args.local_rank == 0:
            # args.verifier_arch = "inception_v3"
            print("loading verifier: ", args.verifier_arch)
            net_verifier = models.__dict__[args.verifier_arch](pretrained=True).to(device)
            net_verifier.eval()

            if use_amp:
                net_verifier = net_verifier.half()

    if args.cig_scale != 0.0:
        # net_verifier = resnet50(skip_gate=True)
        # load_model_rn50(net_verifier, filename='./checkpoint/trained_imagenet_58.weights', use_checkpoint = True)
        student_arch = "resnet18"
        net_verifier = models.__dict__[student_arch](pretrained=True).to(device)
        net_verifier.eval()

        if use_amp:
            net_verifier = network_to_half(net_verifier)

        net_verifier = net_verifier.to(device)
        net_verifier.train()

        if use_amp:
            # net_verifier = net_verifier.half()
            for module in net_verifier.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval().half()

    from deepinversion import DeepInversionClass

    exp_name = args.exp_name
    adi_data_path = "./temp/images/%s"%exp_name
    exp_name = "generations/%s"%exp_name

    args.iterations = 2000
    args.start_noise = True
    args.detach_student = False

    args.resolution = 224
    bs = args.bs
    jitter = 30

    parameters = dict()
    parameters["resolution"] = 224
    parameters["random_label"] = False
    parameters["start_noise"] = True
    parameters["detach_student"] = False
    parameters["do_flip"] = True

    parameters["do_flip"] = args.do_flip

    criterion = nn.CrossEntropyLoss()

    coefficients = dict()
    coefficients["r_feature"] = args.r_feature
    coefficients["tv_l1"] = args.tv_l1
    coefficients["tv_l2"] = args.tv_l2
    coefficients["wd"] = args.wd
    coefficients["lr"] = args.lr
    coefficients["first_bn_coef"] = args.first_bn_coef
    coefficients["main_loss_multiplier"] = args.main_loss_multiplier
    coefficients["cig_scale"] = args.cig_scale

    network_output_function = lambda x: x

    # check accuracy of verifier
    if args.verifier:
        hook_for_display = lambda x,y: validate_one(x, y, net_verifier)
    else:
        hook_for_display = None

    DeepInversionEngine = DeepInversionClass(net_teacher=net,
                                             data_path=adi_data_path,
                                             path=exp_name,
                                             logger_big=None,
                                             parameters=parameters,
                                             setting_id=0,
                                             bs = bs,
                                             use_amp = args.fp16,
                                             jitter = jitter,
                                             criterion=criterion,
                                             coefficients = coefficients,
                                             network_output_function = network_output_function,
                                             hook_for_display = hook_for_display)

    DeepInversionEngine.generate_batch(net_student=None)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--worldsize', type=int, default=1, help='Number of processes participating in the job.')
    parser.add_argument('--local_rank', '--rank', type=int, default=0, help='Rank of the current process.')
    parser.add_argument('--cig_scale', type=float, default=0.0, help='Coefficient for Adaptive Deep Inversion')
    parser.add_argument('--no-cuda', action='store_true')

    parser.add_argument('--epochs', default=20000, type=int, help='batch size')
    parser.add_argument('--bs', default=64, type=int, help='batch size')
    parser.add_argument('--jitter', default=30, type=int, help='batch size')
    parser.add_argument('--comment', default='', type=str, help='batch size')
    parser.add_argument('--arch_name', default='resnet50', type=str, help='model name from torchvision or resnet50v15')

    parser.add_argument('--fp16', action='store_true', help='use FP16 for optimization')
    parser.add_argument('--exp_descr', type=str, help='')
    parser.add_argument('--exp_name', type=str, default='test', help='where to store experimental data')

    parser.add_argument('--verifier', action='store_true', help='run test with resnet18')
    parser.add_argument('--verifier_arch', type=str, default='mobilenet_v2', help = "arch name from torchvision models")

    parser.add_argument('--do_flip', action='store_true', help='apply flip for model inversion')
    parser.add_argument('--r_feature', type=float, default=0.05, help='coefficient for feature distribution regularization')
    parser.add_argument('--tv_l1', type=float, default=0.0, help='coefficient for total variation L1 loss')
    parser.add_argument('--tv_l2', type=float, default=0.0001, help='coefficient for total variation L2 loss')
    parser.add_argument('--lr', type=float, default=0.2, help='learning rate for optimization')
    parser.add_argument('--wd', type=float, default=0.01, help='weight decay for optimization')
    parser.add_argument('--first_bn_coef', type=float, default=0.0, help='additional regularization for the first BN in the networks, coefficient, useful if colors do not match')
    parser.add_argument('--main_loss_multiplier', type=float, default=1.0, help=' coefficient for the main loss optimization')


    parser.add_argument('--distributed', action='store_true', help='distributed')
    parser.add_argument('--dispatch_ngc', action='store_true', help='dispatch_ngc')

    args = parser.parse_args()

    # if args.dispatch_ngc:
    #     args.exp_descr = args.exp_descr + "_tvl1_{:3.1e}_tvl2_{:3.1e}_fp16_{}_cos{:3.1e}_dist{:3.1e}_l2norm{:3.1e}_bn{:3.1e}_l1norm{:3.1e}_lr{:3.1e}".format(args.var_scale_l1, args.var_scale, int(args.fp16), args.cos_loss_weight, args.dist_cost_scale, args.l2norm_scale, args.bn_reg_scale, args.l1norm_scale, args.lr)
    #     if args.no_flip:
    #         args.exp_descr += "_noflip"
    #     args.exp_descr = args.exp_descr.replace("+","")

    print(args)

    # if args.dispatch_ngc:
    #     _dispatch_ngc(args)

    if args.distributed:
        # FOR DISTRIBUTED:  Set the device according to local_rank.
        print("args.local_rank", args.local_rank)
        torch.cuda.set_device(args.local_rank)

        # FOR DISTRIBUTED:  Initialize the backend.  torch.distributed.launch will provide
        # environment variables, and requires that you use init_method=`env://`.
        torch.distributed.init_process_group(backend='nccl',
                                             init_method="env://")
                                             # init_method="file://./temp/distributed_test_%s"%datetime.now().strftime("%d.%H.%M.%S"), rank=args.local_rank, world_size=1)
                                             # init_method='tcp://%s'%datetime.now().strftime("%d.%H.%M.%S"))

    torch.backends.cudnn.benchmark = True

    run(args)


if __name__ == '__main__':
    main()

