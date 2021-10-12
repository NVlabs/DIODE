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
Create randomized labels for COCO images
COCO labels are structured as: 
[0-79] x y w h 
where x,y,w,h are normalized to 0-1 and have 6 places after decimal and 1 place before decimal (0/1) 
e.g: 
1 0.128828 0.375258 0.249063 0.733333
0 0.476187 0.289613 0.028781 0.138099

To randomize: 
First generate width and height dimensions 
Then jitter the x/y labels
Then fix using max/min clipping
"""
import numpy as np
import argparse
import os
from PIL import Image
from tqdm import tqdm

MINDIM=0.2
MAXDIM=0.8

def populate(args):

    # folder
    os.makedirs(os.path.join(args.outdir, "images", "train2014"))
    os.makedirs(os.path.join(args.outdir, "labels", "train2014")) 

    for imgIdx in tqdm(range(args.numImages)):

        # box: w,h,x,y
        width   = MINDIM + (MAXDIM-MINDIM) * np.random.rand()
        height  = MINDIM + (MAXDIM-MINDIM) * np.random.rand()
        x       = 0.5 + (0.5-width/2.0)  * np.random.rand() * np.random.choice([1,-1])
        y       = 0.5 + (0.5-height/2.0) * np.random.rand() * np.random.choice([1,-1])
        assert x+width/2.0 <= 1.0, "overflow width, x+width/2.0={}".format(x+width/2.0)
        assert y+height/2.0<= 1.0, "overflow height, y+height/2.0={}".format(y+height/2.0)

        # class
        cls     = np.random.choice(np.arange(args.numClasses))

        _label_str = "{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
            int(cls), x, y, width, height
        )

        im = Image.new(mode="RGB", size=(256,256), color=(127,127,127))

        # save
        outfile = "COCO_train2014_{:012d}".format(imgIdx+1)
        im.save(os.path.join(args.outdir, "images", "train2014", outfile+".jpg"))
        with open(os.path.join(args.outdir, "labels", "train2014", outfile+".txt"), "wt") as f:
            f.write(_label_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='populate single box per image labels') 
    parser.add_argument('--numImages', type=int, default=120000, help='number of images to generate') 
    parser.add_argument('--numClasses', type=int, default=80, help='number of classes')
    parser.add_argument('--outdir', type=str, required=True, help='output directory') 
    args = parser.parse_args()

    populate(args)

