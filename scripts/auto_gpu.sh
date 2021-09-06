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


if [ "${CUDA_VISIBLE_DEVICES}" = "auto" ]
then
    # number of gpus 
    NUMGPUS=`nvidia-smi -q -d MEMORY | grep "Attached GPU" | grep -P -o "\d"`
    echo "NUMGPUS: $NUMGPUS"
    
    # extract free-memory for each gpu
    MEMLIST="ID FREEMEM"
    for (( DEVICE=0; DEVICE<${NUMGPUS}; DEVICE++ ))
    do
        echo "RUNNING for GPU: ${DEVICE}"
        FREEMEM=`nvidia-smi -q -d MEMORY -i ${DEVICE} | grep "Free" | head -n1 | grep -E -o "[0-9]+"`
        MEMLIST="${MEMLIST}\n${DEVICE} ${FREEMEM}"
    done
    echo "####################"
    echo -e $MEMLIST
    echo "####################"

    # MEMLIST --> remove first line --> sort on gpumem --> pick first line --> pick first GPU device-id 
    export CUDA_VISIBLE_DEVICES=`echo -e ${MEMLIST} | tail -n +2 | sort -n -r -k2 | head -n1 | grep -E -o "^[0-9]"`

fi 
echo "CUDA_VISIBLE_DEVICES set to: ${CUDA_VISIBLE_DEVICES}"
