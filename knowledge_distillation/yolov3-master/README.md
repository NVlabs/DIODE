## YoLo-V3 Knowledge Distillation

Original code from: https://github.com/ultralytics/yolov3 


This repository performs knowledge distillation between two yolo-v3 models: pre-trained teacher and student initialized from scratch using proxy datasets. 

### Environment

Install python 3.8 environment with following packages:

```
$ pip install -r requirements.txt
```

or use provided Dockerfile to create an image. 


### How to run?

1. Get access to `diode_yolo` directory as in top level repository. 
2. Extract a proxy dataset from `diode_yolo` directory to `/tmp` as follows:
   ``` 
   $ tar xzf /path/to/diode_yolo/hallucinate/hallucinate_320_normed.tgz -C /tmp
   ```
3. Extract coco dataset from `diode_yolo` directory to `/tmp` as follows: (for evaluation during training)
   ```
   $ tar xzf /path/to/diode_yolo/coco/coco.tgz -C /tmp
   ```
3. Copy yolo-v3 teacher weights file from `diode_yolo` to `weights` directory.
   ```
   cp /path/to/diode_yolo/pretrained/yolov3-spp-ultralytics.pt /path/to/lpr_deep_inversion/yolov3/weights/
   ```
3. Perform knowledge distillation on proxy dataset as follows:
   ```
   python distill.py --data NGC_hallucinate.data --weights '' --batch-size 64 --cfg yolov3-spp.cfg --device='0,1,2,3' --nw=20 --cfg-teacher yolov3-spp.cfg --weights-teacher './weights/yolov3-spp-ultralytics.pt' --alpha-yolo=0.0 --alpha-distill=1.0 --distill-method='mse'
   ```
4. Evaluate:
   ```
   python test.py --cfg yolov3-spp.cfg --weights='weights/best.pt' --img 640 --data='data/NGC_coco2014.data' --device='0'
   ```

Distillation and training logs are available at `diode_yolo/logs/yolov3_spp/`. e.g for onebox dataset distillation:
```
$ ls -1 /path/to/diode_yolo/logs/yolov3_spp/distill.onebox

best.pt (best checkpoint)
bestresults (evaluation results from best checkpoint)
info.txt (distillation command, evaluation command, time taken etc)
last.pt (last checkpoint)
lastresults (evaluation results from last checkpoint)
results.txt (eval results of every epoch)
runs (tensorboard logs)
test_batch0_gt.jpg
test_batch0_pred.jpg
train_batch0.jpg

```

Knowledge distillation can be performed with different proxy datasets. The available proxy dataset and their corresponding locations and `--data` flag for `distill.py` are:

```
# Real/Rendered proxy datasets
coco  /path/to/diode_yolo/coco/coco.tgz  --data NGC_coco2014.data
GTA5  /path/to/diode_yolo/gta5/gta5.tgz  --data NGC_gta5.data
bdd100k  /path/to/diode_yolo/bdd100k/bdd100k.tar.gz  --data NGC_bdd100k.data
voc  /path/to/diode_yolo/voc/voc.tgz  --data NGC_voc.data
imagenet  /path/to/diode_yolo/imagenet/imagenet.tgz  --data NGC_imagenet.data

# DIODE generated proxy datasets
diode-coco  /path/to/diode_yolo/fakecoco/fakecocov3.tgz  --data NGC_fakecoco.data
diode-onebox  /path/to/diode_yolo/onebox/onebox.tgz  --data NGC_onebox.data
diode-onebox w/ fp sampling  /path/to/diode_yolo/hallucinate/hallucinate_320_normed.tgz  --data NGC_hallucinate.data
diode-onebox w/ tiles  /path/to/diode_yolo/onebox_tiles_coco/tiles.tgz  --data NGC_tiles.data
```

### LICENSE
This code is built on original Yolo-V3 code written by https://github.com/ultralytics/yolov3 and the following files are covered under its original licence https://github.com/NVlabs/DIODE/blob/master/knowledge_distillation/yolov3-master/LICENSE 
```
yolov3-master/cfg/*
yolov3-master/data/samples/*
yolov3-master/data/coco*
yolov3-master/data/get_coco*
yolov3-master/utils/*
yolov3-master/weights/*
yolov3-master/detect.py
yolov3-master/Dockerfile
yolov3-master/models.py
yolov3-master/requirements.txt
yolov3-master/test.py
yolov3-master/train.py
yolov3-master/tutorial.ipynb
```

The following files have been added by this repository and are made available under the Nvidia Source Code License (1-Way Commercial). To view a copy of this license, visit https://github.com/NVlabs/DIODE/blob/master/LICENSE
```
yolov3-master/data/NGC*
yolov3-master/distill.py
yolov3-master/run.sh
```