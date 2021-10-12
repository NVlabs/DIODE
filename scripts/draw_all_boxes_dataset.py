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


import os, sys
from PIL import Image, ImageDraw
import argparse
import numpy as np
from tqdm import tqdm
import pickle

def draw(args):
    
    with open(args.names, "rb") as f:
        names = pickle.load(f)

    os.makedirs(args.outdir)
    with open(args.manifest, "rt") as f:
        images = f.readlines()
        images = [img.strip() for img in images]
    
    labels = [image.replace("images","labels") for image in images]
    labels = [lbl.replace(os.path.splitext(lbl)[1], '.txt') for lbl in labels]

    for image,label in tqdm(zip(images, labels)):
        
        pilimage = Image.open(image).convert(mode='RGB')
        boxes = np.loadtxt(label).reshape(-1,5)
        draw = ImageDraw.Draw(pilimage)
        width, height = pilimage.size
        for box in boxes:
            cls,x,y,w,h = box
            x1 = (x - w/2.0) * width
            x2 = (x + w/2.0) * width  
            y1 = (y - h/2.0) * height
            y2 = (y + h/2.0) * height 
            pilbox = [x1,y1,x2,y2]
            pilbox = [int(atom) for atom in pilbox]
            try:
                draw.rectangle(xy=pilbox, outline=(254,0,0), width=2)
                draw.text(xy=pilbox[0:2], text="cls:{} {}".format(int(cls), names[int(cls)]))
            except:
                import pdb; pdb.set_trace()
        
        outfile = os.path.join(args.outdir, os.path.basename(image))
        pilimage.save(outfile) 
    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='draw boxes on images of dataset')
    parser.add_argument('--manifest', type=str, required=True, help='txt file containing list of images') 
    parser.add_argument('--outdir', type=str, required=True, help='dir where labelled images will be stored')
    parser.add_argument('--names', type=str, required=True, help='path to names.pkl file')
    args = parser.parse_args()

    draw(args)

    
