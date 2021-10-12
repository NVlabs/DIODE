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


# Train
# python train.py --weights '' --cfg yolov3-spp.cfg --data 'data/NGC_coco2014.data' --batch-size 64 --device='0,1,2,3' --nw=28

# Distill 
# python distill_baseline.py --data NGC_coco2014.data --weights './weights/temp/last.pt' --batch-size 64 --cfg yolov3-tiny.cfg --device='0' --nw=8 \
# --cfg-teacher yolov3-tiny.cfg --weights-teacher './weights/yolov3-tiny.pt' \
# --alpha-yolo=0.0 --alpha-distill=1.0 --distill-method='osd' --epochs=100 --device='0' --adam-lr=0.01

# python distill.py --data NGC_coco2014.data --weights '' --batch-size 64 --cfg yolov3-tiny.cfg --device='0' --nw=8 \
# --cfg-teacher yolov3-tiny.cfg --weights-teacher './weights/yolov3-tiny.pt' \
# --alpha-yolo=0.0 --alpha-distill=1.0 --distill-method='mse' --epochs=100 --device='0' --adam-lr=0.001

# Training command for master branch
# python distill.py --data NGC_hallucinate.data --weights '' --batch-size 64 --cfg yolov3-spp.cfg --device='0,1,2,3' --nw=20 \
# --cfg-teacher yolov3-spp.cfg --weights-teacher './weights/yolov3-spp.pt' --alpha-yolo=0.0 --alpha-distill=1.0 --distill-method='mse'

# Training command for debugging branch 
# python -u distill.py --data NGC_tiles.data --weights '' --batch-size 64 --cfg yolov3-spp.cfg --device='0,1,2,3' --nw=20 \
# --cfg-teacher yolov3-spp.cfg --weights-teacher './weights/yolov3-spp.pt' \
# --alpha-yolo=0.0 --alpha-distill=1.0 --distill-method='mse' --epochs=300 --adam-lr=0.001 | tee rawlogs.txt
