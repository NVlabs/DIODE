from .datasets import LoadImagesAndLabels
from .models import Darknet, compute_loss 
from .utils import labels_to_class_weights, non_max_suppression, plot_one_box
import torch 
import glob, pickle
import numpy as np 

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
            print("[INFERENCE] NoneType found in Prediction skipping drawing boxes on this image idx {}".format(idx))
        
        imgs_with_boxes.append(np.transpose(img_np, axes=(2,0,1)))
    imgs_with_boxes = np.array(imgs_with_boxes).astype(np.float32) / 255.0 
    imgs_with_boxes = torch.from_numpy(imgs_with_boxes) 
    
    # Disable inference 
    net.inference_enabled = False
    for mdef, module in zip(net.module_defs, net.module_list): 
        if mdef["type"] == "yolo": 
            module.inference_enabled = False

    return imgs_with_boxes


def get_model_and_dataloader(batch_size=64, load_pickled=True): 
    
    # params
    cfg = "./models/yolo/cfg/yolov3-tiny.cfg" 
    data = "data/coco2014.data" 
    img_size = 320
    weights = "./models/yolo/yolov3-tiny.pt"
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
        assert imgs.shape[0] == batch_size , "ERROR: pickled data bsize: {} , required bsize: {}".format(imgs.shape[0], batch_size)
    else: 
        print("Loading seed data from Full Dataloader")
        test_path = "/home/achawla/akshayws/lpr_deep_inversion/models/yolo/5k_fullpath.txt"
        dataset = LoadImagesAndLabels(test_path, img_size, batch_size,
                                    augment=False,
                                    hyp=hyp,  # augmentation hyperparameters
                                    rect=False,  # rectangular training
                                    cache_images=False,
                                    single_cls=False)

        dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                shuffle=False,
                                                pin_memory=True,
                                                collate_fn=dataset.collate_fn)
        # Get the data as well
        _iter = iter(dataloader)
        imgs, targets, paths, _ = next(_iter)
        labels = dataset.labels
        serialized_dict = {
            "imgs": imgs.detach().cpu().numpy(), 
            "targets": targets.detach().cpu().numpy(), 
            "labels": labels
            }
        with open("serialized_dict.pkl", "wb") as f: 
            pickle.dump(serialized_dict, f)

    # Attach extras to model
    model.nc = nc  # attach number of classes to model
    model.arc = arc  # attach yolo architecture
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 0.0  # giou loss ratio (obj_loss = 1.0 or giou)
    model.class_weights = labels_to_class_weights(labels, nc).to(gpu_device)  # attach class weights

    return model, (imgs, targets), compute_loss

    
