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


SEED=$1 
ITERS=$2 
export CUDA_VISIBLE_DEVICES=$3
echo $CUDA_VISIBLE_DEVICES
echo "Starting with seed: ${SEED} for ${ITERS} iterations"

TERMINALCONDITION=$( expr $SEED + $ITERS )
echo "terminal condition: $TERMCONDITION"
while [ $SEED -lt $TERMINALCONDITION ] 
do
    echo "current seed: $SEED"
    OUTDIR="seed_"
    OUTDIR+="$SEED"
    echo "output directory: ${OUTDIR}"

    python main_yolo.py --resolution=320 --bs=64 \
    --jitter=30 --do_flip --mean_var_clip --rand_brightness --rand_contrast --random_erase \
    --path="/result/$OUTDIR" \
    --train_txt_path="/akshayws/ngc_train_82k.txt" \
    --iterations=4000 \
    --r_feature=0.002 \
    --first_bn_coef=10.0 \
    --main_loss_multiplier=1.0 \
    --alpha_img_stats=0.0 \
    --tv_l1=100.0 \
    --tv_l2=0.0 \
    --lr=0.015 \
    --wd=0 \
    --save_every=500 \
    --seeds="0,0,${SEED}" --shuffle \
    --display_every=100 --init_scale=0.28661 --init_bias=0.48853 --fp16 > /dev/null

    cat /result/$OUTDIR/losses.log | grep "\[VERIFIER\] Real image mAP"
    cat /result/$OUTDIR/losses.log | grep "mAP VERIFIER" | tail -n1

    SEED=$( expr $SEED + 1 )

done 

