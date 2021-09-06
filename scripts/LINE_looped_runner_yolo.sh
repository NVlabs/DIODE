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


##############################################################################
# This script runs the main_yolo.py file multiple times with different inits 
# to generate multiple different batches of inverted images. 
# e.g. $ ./LINE_looped_runner_yolo.sh 1 512 CUDA_GPU_ID
# How: this will split lines [1-511] in provided manifest file into 4 subsets of 128 
# lines each and will generate images with labels initialized from each of those
# subsets.
# To generate large number of images, run this script on multiple GPUs as follows:
# $ ./LINE_looped_runner_yolo.sh 1 512 0
# $ ./LINE_looped_runner_yolo.sh 512 1024 1
# $ ./LINE_looped_runner_yolo.sh 1024 1536 2
# $ .... 

# To generate our datasets we ran this script on 28 gpus to generate a dataset in 48
# hours. 

# How to use this script: 
# 1. use LINE_looped_runner_yolo.sh to generate 938 batches of 128 images
#    of 160x160 resolution each.
# 2. Then coalesce images from 938 batches into a single dataset of 120064 images
# 3. Upsample 120064 images from 160x160 to 320x320 using imagemagick or any other tool
# 3. Then use this newly generated dataset as initialization for this script
#    with resolution=320 and batchsize=96 to fine-tune 320x320 images using DIODE. 
##############################################################################

STARTLINE=$1 
ENDLINE=$2 

export CUDA_VISIBLE_DEVICES=$3
echo "Running on GPU: $CUDA_VISIBLE_DEVICES from [ $STARTLINE , $ENDLINE )"

CURLINE=$STARTLINE
CURENDLINE=0

resolution=160
batchsize=128

##############################################################################
# Uncomment below to use res=320 with bs=96
##############################################################################
# resolution=320
# batchsize=96


while [ $CURLINE -lt  $ENDLINE ]
do

    # CURLINE, CURENDLINE
    CURENDLINE=$( expr $CURLINE + $batchsize )
    if [ $CURENDLINE -gt $ENDLINE ]
    then
        CURENDLINE=$ENDLINE
        batchsize=$( expr $CURENDLINE - $CURLINE )
    fi 
    echo "lines: [$CURLINE - $CURENDLINE ) | batchsize: $batchsize | resolution: $resolution"
    
    # extract subset trainvalno5k lines [$CURLINE - $CURENDLINE) 
    # randstring=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 32 | head -n 1)
    SUBSETFILE="subset_${CURLINE}_${CURENDLINE}_bs${batchsize}_res${resolution}.txt"
    OUTDIR="subset_${CURLINE}_${CURENDLINE}_bs${batchsize}_res${resolution}"

    #######################################################################
    # Modify path to dataset below
    # e.g /tmp/onebox/manifest.txt to get DIODE initialization labels from one-box dataset
    # e.g /tmp/coco/trainvalno5k.txt to get DIODE initialization labels from coco dataset 
    #######################################################################
    cat /tmp/onebox/manifest.txt | head -n $( expr $CURENDLINE - 1 ) | tail -n $batchsize > /tmp/$SUBSETFILE 

    # Check that number of lines in file == batchsize 
    nlines=$( cat /tmp/$SUBSETFILE | wc -l )
    if [ $nlines -ne $batchsize ]
    then
        echo "Note: bs:${batchsize} doesn't match nlines:$nlines" 
    fi

    echo $(date)

    # Resolution = 160
    python -u main_yolo.py --resolution=${resolution} --bs=${batchsize} \
    --jitter=20 --do_flip --rand_brightness --rand_contrast --random_erase \
    --path="/tmp/${OUTDIR}" \
    --train_txt_path="/tmp/$SUBSETFILE" \
    --iterations=2500 \
    --r_feature=0.1 --p_norm=2 --alpha-mean=1.0 --alpha-var=1.0 --num_layers=-1 \
    --first_bn_coef=2.0 \
    --main_loss_multiplier=1.0 \
    --alpha_img_stats=0.0 \
    --tv_l1=75.0 \
    --tv_l2=0.0 \
    --lr=0.2 --min_lr=0.0 --beta1=0.0 --beta2=0.0 \
    --wd=0.0 \
    --save_every=1000 \
    --seeds="0,0,23460" \
    --display_every=100 --init_scale=1.0 --init_bias=0.0 --nms_conf_thres=0.05 --alpha-ssim=0.00 --save-coco > /dev/null
    
    # # Resolution = 320
    # python main_yolo.py --resolution=${resolution} --bs=${batchsize} \
    # --jitter=40 --do_flip --rand_brightness --rand_contrast --random_erase \
    # --path="/tmp/${OUTDIR}" \
    # --train_txt_path="/tmp/$SUBSETFILE" \
    # --iterations=1500 \
    # --r_feature=0.1 --p_norm=2 --alpha-mean=1.0 --alpha-var=1.0 --num_layers=51 \
    # --first_bn_coef=0.0 \
    # --main_loss_multiplier=1.0 \
    # --alpha_img_stats=0.0 \
    # --tv_l1=75.0 \
    # --tv_l2=0.0 \
    # --lr=0.002 --min_lr=0.0005 \
    # --wd=0.0 \
    # --save_every=1000 \
    # --seeds="0,0,23460" \
    # --display_every=100 --init_scale=1.0 --init_bias=0.0 --nms_conf_thres=0.1 --alpha-ssim=0.0 --save-coco --real_mixin_alpha=1.0 \
    # --box-sampler-warmup=4000 --box-sampler-conf=0.2 --box-sampler-overlap-iou=0.35 --box-sampler-minarea=0.01 --box-sampler-maxarea=0.85 --box-sampler-earlyexit=4000 > /dev/null

    # Clean up large unusable files
    rm /tmp/$OUTDIR/chkpt.pt
    rm /tmp/$OUTDIR/iteration_targets*
    rm /tmp/$OUTDIR/tracker.data
    mv /tmp/$SUBSETFILE /tmp/$OUTDIR 
    cat /tmp/$OUTDIR/losses.log | grep "Initialization"
    cat /tmp/$OUTDIR/losses.log | grep "Verifier RealImage"
    cat /tmp/$OUTDIR/losses.log | grep "Verifier GeneratedImage" | tail -n1

    # tar this folder 
    tar czf /tmp/${OUTDIR}.tgz -C /tmp $OUTDIR
    rm -r /tmp/$OUTDIR
    mv /tmp/${OUTDIR}.tgz /result/

    # loop increment
    CURLINE=$CURENDLINE
done

echo "Finished"
