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

def run_inference(net, batch_tens): 

    imgs = batch_tens.clone().detach().cuda()  
    net.eval() 

    # Get colors + names of classes 
    with open("./models/yolo/names.pkl", "rb") as f: 
        names = pickle.load(f) 
    with open("./models/yolo/colors.pkl", "rb") as f: 
        colors = pickle.load(f) 

    # Enable inference 
    net.inference_enabled = True
    for mdef, module in zip(net.module_defs, net.module_list): 
        if mdef["type"] == "yolo": 
            module.inference_enabled = True

    # Inference
    with torch.no_grad():
        pred = net(imgs)[0] # (batchsize, bboxes, 85) 

        # Apply NMS
        conf_thres = 0.3 
        iou_thres  = 0.6 
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)

    # Plot bounding boxes on each image
    imgs_with_boxes = []
    for idx, det in enumerate(pred): 

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
    
    # Disable inference 
    net.inference_enabled = False
    for mdef, module in zip(net.module_defs, net.module_list): 
        if mdef["type"] == "yolo": 
            module.inference_enabled = False
    
    del pred 
    torch.cuda.empty_cache()

    return imgs_with_boxes


def get_model_and_targets(
        batch_size=64, load_pickled=True, shuffle=False, 
        train_txt_path="/home/achawla/akshayws/lpr_deep_inversion/models/yolo/5k_fullpath.txt"):
    
    # params
    cfg = "./models/yolo/cfg/yolov3.cfg" 
    data = "data/coco2014.data" 
    img_size = 320
    weights = "./models/yolo/yolov3.pt"
    cpu_device = torch.device("cpu")
    gpu_device = torch.device("cuda:0")
    num_workers = 0
    nc = 80 
    arc = "default"

    # models
    model = Darknet(cfg, img_size)
    model.load_state_dict(torch.load(weights, map_location=cpu_device)['model'])

    # Datasets + DataLoaders
    if (len(glob.glob("./serialized_dict.pkl")) == 1) and load_pickled:  
        print("Loading seed data from serialized_dict.pkl")
        with open("serialized_dict.pkl", "rb") as f: 
            serialized_dict = pickle.load(f) 
        imgs = torch.from_numpy(serialized_dict["imgs"]) 
        targets = torch.from_numpy(serialized_dict["targets"]) 
        labels = serialized_dict["labels"]
        imgspaths = None
        assert batch_size <= imgs.shape[0], "ERROR: pickled data bsize: {} , required bsize: {}, Insufficient data".format(imgs.shape[0], batch_size)
        if batch_size < imgs.shape[0]: 
            import warnings 
            excess_data_warning_str = "loaded pickled data has bsize: {} , required bsize:  {} , So clipping out extra data".format(imgs.shape[0], batch_size)
            warnings.warn(excess_data_warning_str)
            imgs = imgs[0:batch_size]
            targets = targets[ targets[:,0] < batch_size ] # targets[0,:] = (batch_idx, class, x, y, w, h)
    else: 
        print("Loading seed data from Full Dataloader")
        assert os.path.isfile(train_txt_path)
        dataset = LoadImagesAndLabels(train_txt_path, img_size, batch_size,
                                    augment=False,
                                    hyp=hyp,  # augmentation hyperparameters
                                    rect=False,  # rectangular training
                                    cache_images=False,
                                    cache_labels=False,
                                    single_cls=False)

        dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                shuffle=shuffle,
                                                pin_memory=True,
                                                collate_fn=dataset.collate_fn)
        # Get the data as well
        _iter = iter(dataloader)
        imgs, targets, imgspaths, _ = next(_iter)
        labels = dataset.labels
        serialized_dict = {
            "imgs": imgs.detach().cpu().numpy(), 
            "targets": targets.detach().cpu().numpy(), 
            "labels": labels
            }
        # with open("serialized_dict.pkl", "wb") as f: 
        #     pickle.dump(serialized_dict, f)

    # Attach extras to model
    model.nc = nc  # attach number of classes to model
    model.arc = arc  # attach yolo architecture
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 0.0  # giou loss ratio (obj_loss = 1.0 or giou)
    # model.class_weights = labels_to_class_weights(labels, nc).to(gpu_device)  # attach class weights
    # NOTE: class weights isn't used in compute_loss, so I guess its safe to remove it, plus removes dependence on labels

    return model, (imgs, targets, imgspaths), compute_loss

def calculate_metrics(net, imgs, targets):
    """
    Calculate iou metrics on network using imgs and corresponding targets
    """
    imgs, targets = imgs.clone().detach().cuda(), targets.clone().detach().cuda()

    # Enable inference
    net.eval()
    net.inference_enabled = True
    for mdef, module in zip(net.module_defs, net.module_list):
        if mdef["type"] == "yolo":
            module.inference_enabled = True

    # Inference
    with torch.no_grad():
        preds = net(imgs)[0] # (batchsize, bboxes, 85)
        # Apply NMS
        conf_thres = 0.01 # 0.01 for speed, 0.001 for best mAP
        iou_thres  = 0.6
        output = non_max_suppression(preds, conf_thres, iou_thres, classes=None, agnostic=False)

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

    # print("MP: {} MR: {} MAP: {} MF1: {}".format(mp, mr, map, mf1))
    # print("Number of targets per class: {}".format(nt))

    # Disable inference
    net.inference_enabled = False
    for mdef, module in zip(net.module_defs, net.module_list):
        if mdef["type"] == "yolo":
            module.inference_enabled = False

    # save memory 
    del output, preds
    torch.cuda.empty_cache()
    return mp, mr, map, mf1

def get_verifier():
    # params
    verifier_cfg = "./models/yolo/cfg/yolov3-tiny.cfg" 
    img_size = 320
    weights = "./models/yolo/yolov3-tiny.pt"
    cpu_device = torch.device("cpu")
    gpu_device = torch.device("cuda:0")
    num_workers = 0
    nc = 80 
    arc = "default"

    # verifiers
    verifier = Darknet(verifier_cfg, img_size)
    verifier.load_state_dict(torch.load(weights, map_location=cpu_device)['model'])

    verifier.nc = nc  # attach number of classes to verifier
    verifier.arc = arc  # attach yolo architecture
    verifier.hyp = hyp  # attach hyperparameters to verifier
    verifier.gr = 0.0  # giou loss ratio (obj_loss = 1.0 or giou)
    # verifier.class_weights = labels_to_class_weights(labels, nc).to(gpu_device)  # attach class weights

    return verifier

def flip_targets(targets, horizontal=True, vertical=False): 
    """
    horizontal and vertical flipping for targets
    """
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
    img_shape = (width, height)
    targets shape = [batch_idx, cls, center x, center y, w, h]
    """ 
    assert targets.shape[1] == 6
    targets_jittered = targets.clone().detach().cpu() 
    width, height = img_shape 
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

def convert_to_coco(inputs_tensor, targets):
    """
    Convert an inputs_tensor (bsx3x320x320) to #bs images in PIL format
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
