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
