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


""" 
script to find corrupted images in dataset 
""" 
import cv2 
import os, sys 
import numpy as np 
from tqdm import tqdm

manifest = sys.argv[1] # path to trainvalno5k.txt/manifest.txt file

with open(manifest, "rt") as f:
    images = f.readlines()
    images = [img.strip() for img in images]

damaged = []
for image in tqdm(images):
    try:
        img = cv2.imread(image)
        # assert img.shape[0] == 256 
        # assert img.shape[1] == 256 
    except:
        damaged.append(image)
        print("found damaged image! {}".format(image))

alright = set(images) - set(damaged)
alright = list(alright) 

alright = "\n".join(alright) 
damaged = "\n".join(damaged) 

with open(manifest+".damaged.txt", "wt") as f:
    f.write(damaged)
with open(manifest+".alright.txt", "wt") as f:
    f.write(alright)
