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


from .datasets import LoadImagesAndLabels
from .models import Darknet, compute_loss
from .utils import labels_to_class_weights, non_max_suppression, plot_one_box, clip_coords, xywh2xyxy, xyxy2xywh, box_iou, ap_per_class
import torch
import glob, pickle, os
import numpy as np
import torchvision
from PIL import Image

hyp = {'giou': 3.54,  # giou loss gain
       'cls': 37.4,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.225,  # iou training threshold
       'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
       'lrf': -4.,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
       'momentum': 0.937,  # SGD momentum
       'weight_decay': 0.000484,  # optimizer weight decay
       'fl_gamma': 1.5,  # focal loss gamma
       'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 1.98 * 0,  # image rotation (+/- deg)
       'translate': 0.05 * 0,  # image translation (+/- fraction)
       'scale': 0.05 * 0,  # image scale (+/- gain)
       'shear': 0.641 * 0}  # image shear (+/- deg)

def load_model(cfg, weights, img_size=320):
    """Load a YoloV3 model matching architecture `cfg` with parameters `weights`."""
    net = Darknet(cfg, img_size)
    net.load_state_dict(torch.load(weights, map_location=torch.device('cpu'))['model'])
    net.nc = 80
    net.arc = 'default'
    net.hyp = hyp
    net.gr = 0.0  # giou loss ratio (obj_loss = 1.0 or giou)
    # model.class_weights = labels_to_class_weights(labels, nc).to(gpu_device)  # attach class weights, look at note
    # NOTE: class weights isn't used in compute_loss, so I guess its safe to remove it, plus removes dependence on labels
    return net


def load_batch(train_txt_path, batch_size=64, img_size=320, shuffle=False):
    """Loads a batch of data with `batch_size` items from dataset `train_txt_path`."""
    dataset = LoadImagesAndLabels(train_txt_path, img_size, batch_size,
                augment=False, hyp=hyp, rect=False, cache_images=False,
                cache_labels=False, single_cls=False)
    dataloader = torch.utils.data.DataLoader(dataset,
                    batch_size=batch_size, num_workers=0,
                    shuffle=shuffle, pin_memory=False,
                    collate_fn=dataset.collate_fn)
    imgs, targets, imgspaths, _ = next(iter(dataloader))
    return imgs.float()/255.0, targets, imgspaths


def inference(net, imgs, targets, nms_params={"iou_thres":0.5, "conf_thres":0.01}):
    """Calculate iou metrics on network using `imgs` and corresponding `targets`."""
    imgs, targets = imgs.clone().detach().cuda(), targets.clone().detach().cuda()

    # Enable inference
    net.eval()

    # Forward + nms
    with torch.no_grad():
        preds = net(imgs)[0] # (batchsize, bboxes, 85)
        # Apply NMS
        # Confidence threshold: 0.01 for speed, 0.001 for best mAP
        output = non_max_suppression(preds, nms_params["conf_thres"], nms_params["iou_thres"], classes=None, agnostic=False)

    # Get colors + names of classes
    with open("./models/yolo/names.pkl", "rb") as f:
        names = pickle.load(f)
    with open("./models/yolo/colors.pkl", "rb") as f:
        colors = pickle.load(f)

    # Plot bounding boxes on each image
    imgs_with_boxes = []
    for idx, det in enumerate(output):

        img_np = imgs[idx].clone().detach().cpu().numpy()
        img_np = np.transpose(img_np, axes=(1,2,0))
        img_np = (img_np * 255).astype(np.uint8)
        img_np = np.ascontiguousarray(img_np, dtype=np.uint8)

        # Plot boundingboxes for this image
        if det is not None:
            for *xyxy, conf, cls in det:
                label = '%s %.2f' % (names[int(cls)], conf)
                plot_one_box(xyxy, img_np, label=label, color=colors[int(cls)])
        else:
            # print("[INFERENCE] NoneType found in Prediction skipping drawing boxes on this image idx {}".format(idx))
            pass

        imgs_with_boxes.append(np.transpose(img_np, axes=(2,0,1)))
    imgs_with_boxes = np.array(imgs_with_boxes).astype(np.float32) / 255.0
    imgs_with_boxes = torch.from_numpy(imgs_with_boxes)

    # hyperparameters
    batch_size, _, height, width = imgs.shape
    iouv = torch.tensor([0.5], dtype=torch.float32).cuda()
    niou = iouv.numel()
    whwh = torch.tensor([width, height, width, height], dtype=torch.float32).cuda()
    stats = []

    # Compute metrics per image
    for img_idx, pred in enumerate(output):
        labels = targets[targets[:, 0] == img_idx , 1:]
        nl = len(labels)
        tcls = labels[:, 0].tolist()  # target class

        if pred is None:
            if nl:
                stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
            continue

        # Clip boxes to image bounds
        clip_coords(pred, (height, width))

        # Assign all predictions as incorrect
        correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool).cuda()

        detected = []  # target indices
        tcls_tensor = labels[:, 0]

        # target boxes
        tbox = xywh2xyxy(labels[:, 1:5]) * whwh

        # Per target class
        for cls in torch.unique(tcls_tensor):
            ti = (cls == tcls_tensor).nonzero().view(-1)  # prediction indices
            pi = (cls == pred[:, 5]).nonzero().view(-1)  # target indices

            # Search for detections
            if pi.shape[0]:
                # Prediction to target ious
                ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                # Append detections
                for j in (ious > iouv[0]).nonzero():
                    d = ti[i[j]]  # detected target
                    if d not in detected:
                        detected.append(d)
                        correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                        if len(detected) == nl:  # all targets already located in image
                            break

        # Append statistics (correct, conf, pcls, tcls)
        stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    p, r, ap, f1, ap_class = ap_per_class(*stats)
    mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
    nt = np.bincount(stats[3].astype(np.int64), minlength=net.nc)  # number of targets per class

    # save memory
    del preds
    torch.cuda.empty_cache()
    return float(mp), float(mr), float(map), float(mf1), imgs_with_boxes, output


def flip_targets(targets, horizontal=True, vertical=False):
    """horizontal and vertical flipping for `targets`."""
    assert targets.shape[1] == 6
    targets_flipped = targets.clone().detach()
    if horizontal:
        targets_flipped[:,2] = 0.5 - (targets_flipped[:,2] - 0.5)
    if vertical:
        targets_flipped[:,3] = 0.5 - (targets_flipped[:,3] - 0.5)
    return targets_flipped

def jitter_targets(targets, xshift=0, yshift=0, img_shape=(320,320)):
    """
    Apply horizontal & vertical jittering to the targets for given img_shape
    note: img_shape is in real world parameters, but targets are still between 0-1
    img_shape = (height, width)
    targets shape = [batch_idx, cls, center x, center y, w, h]
    """
    assert targets.shape[1] == 6
    targets_jittered = targets.clone().detach().cpu()
    height, width = img_shape
    xywh = targets_jittered[:,2:]
    whwh = torch.tensor([width, height, width, height], dtype=torch.float32)
    xyxy = xywh2xyxy(xywh) * whwh

    # adjust the tbox
    xyxy[:,0] += xshift
    xyxy[:,2] += xshift
    xyxy[:,1] += yshift
    xyxy[:,3] += yshift

    # Limit co-ords
    xyxy[:,0] = torch.clamp(xyxy[:,0], min=0, max=width)
    xyxy[:,2] = torch.clamp(xyxy[:,2], min=0, max=width)
    xyxy[:,1] = torch.clamp(xyxy[:,1], min=0, max=height)
    xyxy[:,3] = torch.clamp(xyxy[:,3], min=0, max=height)

    # xyxy --> xywh
    xywh = xyxy2xywh(xyxy / whwh)
    targets_jittered[:,2:] = xywh

    # remove boxes that have 0 area
    oof = (targets_jittered[:,-1] * targets_jittered[:,-2] * width * height) < 1
    # print("Jittering Dropping {} boxes".format(oof.sum()))
    targets_jittered = targets_jittered[~oof]

    return targets_jittered.to(targets.device)


def plot_all_boxes(imgs, targets):
    """
    Useful as a debugging tool for def: jitter_box and def: flip_box
    """

    # targets shape: #boxes x 6  [batch_idx, cls, x, y, w, h]
    targets = targets.clone().detach().cpu()
    imgs_np = imgs.clone().detach().cpu().numpy()
    imgs_np = np.transpose(imgs_np, axes=(0, 2, 3, 1))
    imgs_np = (imgs_np * 255).astype(np.uint8)
    imgs_np = np.ascontiguousarray(imgs_np, dtype=np.uint8)

    # plot box-by-box
    height, width = imgs_np.shape[1], imgs_np.shape[2]
    whwh = torch.tensor([width, height, width, height], dtype=torch.float32)
    for box_idx, box in enumerate(targets):
        box_xyxy = torch.squeeze( xywh2xyxy(box[None, 2:]) * whwh )
        img_idx  = int(box[0])
        cls      = int(box[1])
        label="cls:{}".format(str(cls))
        plot_one_box(box_xyxy, imgs_np[img_idx], label=label, color=(153, 51, 255))

    imgs_np = np.transpose(imgs_np, axes=(0, 3, 1, 2))
    imgs_with_boxes = torch.from_numpy(imgs_np.astype(np.float32) / 255.0)

    return imgs_with_boxes

def random_erase_masks(inputs_shape, return_cuda=True):
    """
    return a 1/0 mask with random rectangles marked as 0.
    shape should match inputs_shape
    """
    bs = inputs_shape[0]
    height = inputs_shape[2]
    width  = inputs_shape[3]
    masks = []
    _rand_erase = torchvision.transforms.RandomErasing(
        p=0.5,
        scale=(0.02, 0.2),
        ratio=(0.3, 3.3),
        value=0
    )
    for idx in range(bs):
        mask = torch.ones(3,height,width,dtype=torch.float32)
        mask = _rand_erase(mask)
        masks.append(mask)
    masks = torch.stack(masks)
    if return_cuda:
        masks = masks.cuda()
    return masks

def predictions_to_coco(output, inputs):
    """
    NMS predictions --> coco targets
    inputs: input image, only required for shape
    output: list w/ size = batchsize
            each element of list is a #predsx6 tensor, each prediction is of dims: xyxy, conf, cls.
            xyxy is in pixel space and CAN have negative vales, so int and clamp
    targets: [bidx, cls, x, y, w, h] where xywh is in 0-1 format
    """
    bsize, channels, height, width = inputs.shape
    assert height==width, "height and width are different"
    targets = []
    for batchIdx, preds in enumerate(output):
        if preds is not None:
            for box in preds:
                xyxy, conf, cls = box[0:4], box[4].item(), int(box[5].item())
                xyxy = torch.clamp(xyxy, min=0.0)
                xywh = xyxy2xywh(xyxy.view(1,-1))[0]
                xywh = xywh / height
                targets.append(torch.tensor([batchIdx, cls, conf, xywh[0].item(), xywh[1].item(), xywh[2].item(), xywh[3].item()]))
    targets = torch.stack(targets, dim=0)
    return targets


def convert_to_coco(inputs_tensor, targets=None):
    """
    Convert an inputs_tensor (bs x 3 x height x width) to #batch-size images in PIL
    format.
    Convert targets loaded by a dataloader to plaintxt annotations in coco format
    """
    images = inputs_tensor.clone().detach().cpu()
    images = torch.clamp(images, min=0.0, max=1.0) * 255.0
    images = images.to(torch.uint8)
    images = images.numpy()
    images = np.transpose(images, axes=(0,2,3,1)) # (32, h, w, 3)
    pil_images = []
    for batch_idx in range(len(images)):
        pil_images.append(Image.fromarray(images[batch_idx]))

    coco_targets = [None] * len(images)
    if targets is not None:
        targets = targets.clone().detach().cpu()
        coco_targets = []
        for batch_idx in range(len(images)):
            imboxes = targets[targets[:,0]==batch_idx]
            boxlist = []
            for box in imboxes:
                _box_str = "{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(int(box[1].item()), box[2].item(), box[3].item(), box[4].item(), box[5].item())
                boxlist.append(_box_str)
            coco_targets.append(boxlist)

    assert len(coco_targets) == len(pil_images)

    return pil_images, coco_targets

def draw_targets(imgs, targets):
    """Draw `targets` bboxes on `imgs`."""
    batch_size = len(imgs)
    # Get colors + names of classes
    with open("./models/yolo/names.pkl", "rb") as f:
        names = pickle.load(f)
    with open("./models/yolo/colors.pkl", "rb") as f:
        colors = pickle.load(f)

    # Draw boxes
    imgs_with_boxes = []
    for idx in range(batch_size):
        img_np = imgs[idx].clone().detach().cpu().numpy()
        img_np = np.transpose(img_np, axes=(1,2,0))
        img_np = (img_np * 255).astype(np.uint8)
        img_np = np.ascontiguousarray(img_np, dtype=np.uint8)
        height,width,_ = img_np.shape

        targets_batch = targets[targets[:,0]==idx]
        if len(targets_batch):
            for box in targets_batch:
                cls = int(box[1].item())
                xywh = box[2:].view(1,-1)
                xyxy = xywh2xyxy(xywh)
                xyxy[:,0] *= width
                xyxy[:,1] *= height
                xyxy[:,2] *= width
                xyxy[:,3] *= height
                label="{}".format(names[cls])
                plot_one_box(xyxy[0], img_np, label=label, color=colors[cls])
        imgs_with_boxes.append(np.transpose(img_np, axes=(2,0,1)))

    imgs_with_boxes = np.array(imgs_with_boxes).astype(np.float32) / 255.0
    imgs_with_boxes = torch.from_numpy(imgs_with_boxes)

    torch.cuda.empty_cache()
    return imgs_with_boxes
