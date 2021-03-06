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
from PIL import Image 

from deepinversion_yolo import DeepInversionClass
from models.yolo.yolostuff import load_model, load_batch, inference, convert_to_coco, draw_targets
from models.yolo.utils import compute_loss as criterion

def create_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def set_all_seeds(seeds): 
    random.seed(seeds[0])
    np.random.seed(seeds[1])
    torch.manual_seed(seeds[3])

def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    net = load_model(cfg='./models/yolo/cfg/yolov3-spp.cfg', weights='./models/yolo/yolov3-spp-ultralytics.pt').to(device)
    net_verifier = load_model(cfg='./models/yolo/cfg/yolov3-tiny.cfg', weights='./models/yolo/yolov3-tiny.pt').to(device)
    imgs, targets, imgspaths = load_batch(args.train_txt_path, args.bs, args.resolution[0], args.shuffle)
    net.eval() 
    net_verifier.eval()

    args.start_noise = True

    parameters = dict()
    # Data augmentation params
    parameters["random_label"] = False
    parameters["start_noise"] = True
    parameters["do_flip"] = args.do_flip
    parameters["jitter"] = args.jitter
    parameters["rand_brightness"] = args.rand_brightness 
    parameters["rand_contrast"]   = args.rand_contrast
    parameters["random_erase"]    = args.random_erase
    parameters["mean_var_clip"] = args.mean_var_clip
    # Other params
    parameters["resolution"] = args.resolution
    parameters["bs"] = args.bs 
    parameters["iterations"] = args.iterations
    parameters["save_every"] = args.save_every
    parameters["display_every"] = args.display_every
    parameters["beta1"] = args.beta1
    parameters["beta2"] = args.beta2
    parameters["nms_params"] = args.nms_params
    parameters["cosine_layer_decay"] = args.cosine_layer_decay
    parameters["min_layers"] = args.min_layers
    parameters["num_layers"] = args.num_layers
    parameters["p_norm"] = args.p_norm
    parameters["alpha_mean"] = args.alpha_mean
    parameters["alpha_var"]  = args.alpha_var
    parameters["alpha_ssim"] = args.alpha_ssim

    # Bounding box samper
    parameters["box_sampler"]        = args.box_sampler
    parameters["box_sampler_warmup"] = args.box_sampler_warmup
    parameters["box_sampler_conf"]   = args.box_sampler_conf
    parameters["box_sampler_overlap_iou"] = args.box_sampler_overlap_iou
    parameters["box_sampler_minarea"]= args.box_sampler_minarea
    parameters["box_sampler_maxarea"]= args.box_sampler_maxarea
    parameters["box_sampler_earlyexit"] = args.box_sampler_earlyexit

    # criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    # criterion = functools.partial(Triterion, loss_weights={"bbox":0.0, "cov":1.0, "orient":0.0})
    # criterion = Triterion(loss_weights={"bbox":1.0, "cov":1.0, "orient":0.0})  

    coefficients = dict()
    coefficients["r_feature"] = args.r_feature
    coefficients["tv_l1"] = args.tv_l1
    coefficients["tv_l2"] = args.tv_l2
    coefficients["wd"] = args.wd
    coefficients["lr"] = args.lr
    coefficients["min_lr"] = args.min_lr
    coefficients["first_bn_coef"] = args.first_bn_coef
    coefficients["main_loss_multiplier"] = args.main_loss_multiplier
    coefficients["alpha_img_stats"] = args.alpha_img_stats

    network_output_function = lambda x: x[1] # When in .eval() mode, DarkNet returns (inference_output, training_output). 

    DeepInversionEngine = DeepInversionClass(net_teacher=net,
                                             net_verifier=net_verifier,
                                             path=args.path,
                                             logger_big=None,
                                             parameters=parameters,
                                             criterion=criterion,
                                             use_amp=args.fp16,
                                             coefficients = coefficients,
                                             network_output_function = network_output_function)

    # initialize inputs
    if args.init_chkpt.endswith(".pt"):
        initchkpt = torch.load(args.init_chkpt, map_location=torch.device("cpu"))
        init = initchkpt["images"]
        imgs = initchkpt["origimages"]
        targets = initchkpt["targets"]
        imgspaths = initchkpt["imgspaths"]
        init, imgs, imgspaths = init[0:args.bs], imgs[0:args.bs], imgspaths[0:args.bs]
        targets = targets[targets[:,0]<args.bs]
        if init.shape[2] != args.resolution[0]:
            init = F.interpolate(init, size=(args.resolution[0], args.resolution[1]))
            imgs = F.interpolate(imgs, size=(args.resolution[0], args.resolution[1]))
    else:
        init = torch.randn((args.bs, 3, args.resolution[0], args.resolution[1]), dtype=torch.float)
        init = torch.clamp(init, min=0.0, max=1.0)
        init = (args.init_scale * init) + args.init_bias
        init = (args.real_mixin_alpha)*imgs + (1.0-args.real_mixin_alpha)*init
    DeepInversionEngine.save_image(init, os.path.join(DeepInversionEngine.path, "initialization.jpg"), halfsize=True)

    init_with_boxes = draw_targets(init, targets)
    DeepInversionEngine.save_image(init_with_boxes, os.path.join(DeepInversionEngine.path, "init_with_boxes.jpg"))
    mPrec, mRec, mAP, mF1, init_with_boxes_verif, _ = inference(net_verifier, init, targets, args.nms_params)
    DeepInversionEngine.save_image(init_with_boxes_verif, os.path.join(DeepInversionEngine.path, "init_with_boxes_verifier.jpg"))
    _init_metrics_str = "Initialization mAP: {} | mF1: {} | mPrec: {} | mRec: {}".format(mAP, mF1, mPrec, mRec)
    DeepInversionEngine.txtwriter.write(_init_metrics_str+"\n")
    print(_init_metrics_str)

    # Save the input image to disk
    imgs_with_boxes_targets  = draw_targets(imgs, targets)
    DeepInversionEngine.save_image(imgs_with_boxes_targets, os.path.join(DeepInversionEngine.path, "real_image_targets.jpg"), halfsize=False)

    # Inference on real image
    mPrec, mRec, mAP, mF1, imgs_with_boxes_verif, _ = inference(net_verifier, imgs, targets, args.nms_params)
    DeepInversionEngine.txtwriter.write("Verifier RealImage mPrec: {:.4} mRec: {:.4} mAP: {:.4} mF1: {:.4} \n".format(mPrec, mRec, mAP, mF1))
    mPrec, mRec, mAP, mF1, imgs_with_boxes_teach, _ = inference(net, imgs, targets, args.nms_params)
    DeepInversionEngine.txtwriter.write("Teacher RealImage mPrec: {:.4} mRec: {:.4} mAP: {:.4} mF1: {:.4} \n".format(mPrec, mRec, mAP, mF1))
    DeepInversionEngine.save_image(imgs_with_boxes_verif, os.path.join(DeepInversionEngine.path, "real_image_verifier.jpg"), halfsize=False)
    DeepInversionEngine.save_image(imgs_with_boxes_teach, os.path.join(DeepInversionEngine.path, "real_image_teacher.jpg"), halfsize=False)

    assert imgs.shape[0] == parameters["bs"], "Batchsize of data {} doesn't match batchsize specified in cli {}".format(imgs.shape[0], parameters["bs"])

    # Used cached stats
    if args.cache_batch_stats:
        DeepInversionEngine.cache_batch_stats(imgs.clone().detach().cuda())
        print("Overwriting cached_mean and cached_var with batch stats of real data") 
        DeepInversionEngine.txtwriter.write("[CACHE_BATCH_STATS] Overwriting cached_mean and cached_var with batch stats of real data\n")

    # Losses on real data batch
    from deepinversion_yolo import get_image_prior_losses
    with torch.no_grad():
        _real_tv_l1, _real_tv_l2 = get_image_prior_losses(imgs.cuda())
        DeepInversionEngine.net_teacher.eval()
        _real_outputs = DeepInversionEngine.net_teacher(imgs.cuda())
        _real_outputs = DeepInversionEngine.network_output_function(_real_outputs)
        _real_task_loss, _ = DeepInversionEngine.criterion(_real_outputs, targets.cuda(), DeepInversionEngine.net_teacher)
        numLayers = len(DeepInversionEngine.loss_r_feature_layers) if args.num_layers==-1 else args.num_layers
        _real_di_loss = sum([mod.r_feature for mod in DeepInversionEngine.loss_r_feature_layers[0:numLayers]])

    _real_loss_str = "Real batch losses: tv L1: {:.4f} tv L2: {:.4f} task: {:.4f} di: {:.4f}".format(_real_tv_l1.item(), _real_tv_l2.item(), _real_task_loss.item(), _real_di_loss.item())
    print(_real_loss_str)
    DeepInversionEngine.txtwriter.write(_real_loss_str+"\n")
    del _real_tv_l1, _real_tv_l2, _real_outputs, _real_task_loss, _real_di_loss

    generatedImages, targets = DeepInversionEngine.generate_batch(targets, init)
    generatedImages_with_targets = draw_targets(generatedImages, targets)
    DeepInversionEngine.save_image(generatedImages_with_targets, os.path.join(DeepInversionEngine.path, "inverted_with_targets.jpg"), halfsize=False)
    mPrec, mRec, mAP, mF1, generatedImages_with_boxes_verif, _ = inference(net_verifier, generatedImages, targets, args.nms_params)
    DeepInversionEngine.save_image(generatedImages_with_boxes_verif, os.path.join(DeepInversionEngine.path, "inverted_with_preds.jpg"), halfsize=False)
    DeepInversionEngine.txtwriter.write("Verifier GeneratedImage mPrec: {:.4} mRec: {:.4} mAP: {:.4} mF1: {:.4} \n".format(mPrec, mRec, mAP, mF1))

    # Store image checkpoint (useful for multi-scale generation)
    chkpt = {"images":generatedImages, "targets":targets, "origimages": imgs, "imgspaths":imgspaths}
    torch.save(chkpt, os.path.join(args.path, "chkpt.pt"))

    # Store generatedImages in coco format
    if args.save_coco:
        os.makedirs(os.path.join(args.path, "coco", "images", "train2014"))
        os.makedirs(os.path.join(args.path, "coco", "labels", "train2014"))
        pilImages, cocoTargets = convert_to_coco(generatedImages, targets)
        for pilim, cocotarget, imgpath in zip(pilImages, cocoTargets, imgspaths):
            imgname = os.path.basename(imgpath) # get filename
            imgname = os.path.splitext(imgname)[0] # remove .jpg/.png extension
            pilim.save(os.path.join(args.path, "coco", "images", "train2014", imgname+".png"))
            with open(os.path.join(args.path,"coco","labels","train2014",imgname+".txt"),"wt") as f:
                if len(cocotarget)>0:
                    f.write(''.join(cocotarget).rstrip('\n'))
    # Save the args
    with open(os.path.join(args.path, "args.txt"), "wt") as f:
        f.write(str(args)+"\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', '--rank', type=int, default=0, help='Rank of the current process.')
    parser.add_argument('--seeds', type=str, default="0,0,0", help="seeds for built_in random, numpy random and torch.manual_seed")
    parser.add_argument('--shuffle', action='store_true', help='use shuffle in dataloader')
    parser.add_argument('--no-cuda', action='store_true')

    parser.add_argument('--resolution', default=320, type=int, help="image optimization resolution")
    parser.add_argument('--epochs', default=20000, type=int, help='number of epochs?')
    parser.add_argument('--iterations', default=2000, type=int, help='number of iterations for DI optim')
    parser.add_argument('--bs', default=1, type=int, help='batch size')
    parser.add_argument('--jitter', default=30, type=int, help='jitter')
    parser.add_argument('--mean_var_clip', action='store_true', help='clip the optimized image to the mean/var sampled from real data')
    parser.add_argument('--arch_name', default='resnet50', type=str, help='model name from torchvision or resnet50v15')

    parser.add_argument('--save_every', type=int, default=100, help='save an image every x iterations')
    parser.add_argument('--display_every', type=int, default=2, help='display the lsses every x iterations')
    parser.add_argument("--nms_conf_thres", type=float, default=0.01, help="NMS confidence 0.01 for speed, 0.001 for max mAP (default 0.01)")
    parser.add_argument("--nms_iou_thres", type=float, default=0.5, help="NMS iou threshold default: 0.5")
    parser.add_argument('--path', type=str, default='test', help='where to store experimental data NOT: MUST BE A FOLDER')
    parser.add_argument('--train_txt_path', type=str, default='/home/achawla/akshayws/lpr_deep_inversion/models/yolo/5k_fullpath.txt', help='Path to .txt file containing location of images for Dataset')
    parser.add_argument('--fp16', action="store_true", help="Enabled Mixed Precision Training")
    parser.add_argument('--save-coco', action="store_true", help="save generated batch in coco format")

    parser.add_argument('--do_flip', action='store_true', help='DA:apply flip for model inversion')
    parser.add_argument("--rand_brightness", action="store_true", help="DA: randomly adjust brightness during optizn")
    parser.add_argument("--rand_contrast", action="store_true", help="DA: randomly adjust contrast during optizn")
    parser.add_argument("--random_erase", action="store_true", help="DA: randomly set rectangular regions to 0 during optizn")

    parser.add_argument('--r_feature', type=float, default=0.05, help='coefficient for feature distribution regularization')
    parser.add_argument('--p_norm', type=int, default=1, help='p for the Lp norm used to calculate r_feature')
    parser.add_argument('--alpha-mean', type=float, default=1.0, help='weight for mean norm in r_feature')
    parser.add_argument('--alpha-var', type=float, default=1.0, help='weight for var norm in r_feature')
    parser.add_argument('--alpha-ssim', type=float, default=0.0, help='weight for ssim')
    parser.add_argument('--cosine_layer_decay', action='store_true', help='use cosine decay for number of layers used to calculate r_feature')
    parser.add_argument('--min_layers', type=int, default=1, help='minimum number of layers used to calculate r_feature when using cosine decay')
    parser.add_argument('--num_layers', type=int, default=-1, help='number of layers used to calculate r_feature')
    parser.add_argument('--tv_l1', type=float, default=0.0, help='coefficient for total variation L1 loss')
    parser.add_argument('--tv_l2', type=float, default=0.0001, help='coefficient for total variation L2 loss')
    parser.add_argument('--lr', type=float, default=0.2, help='learning rate for optimization')
    parser.add_argument('--min_lr', type=float, default=0.0, help='minimum learning rate for scheduler')
    parser.add_argument('--wd', type=float, default=0.01, help='weight decay for optimization')
    parser.add_argument('--first_bn_coef', type=float, default=0.0, help='additional regularization for the first BN in the networks, coefficient, useful if colors do not match')
    parser.add_argument('--main_loss_multiplier', type=float, default=1.0, help=' coefficient for the main loss optimization')
    parser.add_argument('--alpha_img_stats', type=float, default=1.0, help='coefficient for loss_img_stats')
    parser.add_argument("--cache_batch_stats", action="store_true", help="use real image stats instead of bnorm mean/var")
    parser.add_argument("--real_mixin_alpha", type=float, default=0.0, help="how much of real image to mix in with the random initialization")
    parser.add_argument("--init_scale", type=float, default=1.0, help="for scaling the initialization, useful to start with a 'closer to black' kinda image") 
    parser.add_argument("--init_bias", type=float, default=0.0, help="for biasing the initialization")
    parser.add_argument('--init_chkpt', type=str, default="", help="chkpt containing initialization image (will up upsampled to args.resolution)")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 for adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.999, help="beta1 for adam optimizer")

    # Bounding box sampler technique
    parser.add_argument("--box-sampler", action="store_true", help="Enable False positive (Fp) sampling")
    parser.add_argument("--box-sampler-warmup", type=int, default=1000, help="warmup iterations before we start adding predictions to targets")
    parser.add_argument("--box-sampler-conf", type=float, default=0.5, help="confidence threshold for a prediction to become targets")
    parser.add_argument("--box-sampler-overlap-iou", type=float, default=0.2, help="a prediction must be below this overlap threshold with targets to become a target") # Increasing box overlap leads to more overlapped objects appearing
    parser.add_argument("--box-sampler-minarea", type=float, default=0.0, help="new targets must be larger than this minarea")
    parser.add_argument("--box-sampler-maxarea", type=float, default=1.0, help="new targets must be smaller than this maxarea")
    parser.add_argument("--box-sampler-earlyexit", type=int, default=1000000, help='early exit at this iteration')

    args = parser.parse_args()
    args.resolution = (args.resolution, args.resolution) # int -> (height,width)
    args.nms_params = { "iou_thres":args.nms_iou_thres, "conf_thres":args.nms_conf_thres }

    print(args)
    torch.backends.cudnn.benchmark = True
    run(args)


if __name__ == '__main__':
    main()

