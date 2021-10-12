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


import torch 
from torch import nn
import torch.nn.functional as nnfunc

class Distillation(object):
    
    """
    Distillation loss class 
    supports :param: method--> 
        1. mse      : match yolo layer outputs  
    """

    def __init__(self,method="mse"): 
        if method=="mse":
            self.loss_fn = self.mse
        # elif method=="cfmse":
        #    self.loss_fn = self.cfmse
        # elif method=="cfmse2":
        #     self.loss_fn = self.cfmse2
        else:
            raise NotImplementedError 

    def mse(self, predS, predT): 
        """
        mse between predT & predS
        only works when Stu & Tea are same architecture 
        """ 
        assert len(predT) == len(predS) 
        dLoss = []
        for branchS, branchT in zip(predS, predT):
            dLoss.append(torch.mean((branchS - branchT)**2))
        dLoss = sum(dLoss)
        dLoss_items = torch.tensor((0.0, 0.0, 0.0, dLoss.item())).to(dLoss.device)
        return dLoss, dLoss_items.detach()
