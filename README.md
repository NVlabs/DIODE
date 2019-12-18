![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# Intro

Code for the paper:

Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion<br>
Hongxu Yin, Pavlo Molchanov, Zhizhong Li, Jose M. Alvarez, Arun Mallya, Derek Hoiem, Niraj K. Jha and Jan Kautz<br>

Abstract: We introduce DeepInversion, a new method for synthesizing images from the image distribution used to train a deep neural network. We 'invert' a trained network (teacher) to synthesize class-conditional input images starting from random noise, without using any additional information about the training dataset. Keeping the teacher fixed, our method optimizes the input while regularizing the distribution of intermediate feature maps using information stored in the batch normalization layers of the teacher. Further, we improve the diversity of synthesized images using Adaptive DeepInversion, which maximizes the Jensen-Shannon divergence between the teacher and student network logits. The resulting synthesized images from networks trained on the CIFAR-10 and ImageNet datasets demonstrate high fidelity and degree of realism, and help enable a new breed of data-free applications - ones that do not require any real images or labeled data. We demonstrate the applicability of our proposed method to three tasks of immense practical importance -- (i) data-free network pruning, (ii) data-free knowledge transfer, and (iii) data-free continual learning.

## How to run:
This snippet will generate 84 images by inverting resnet50 model from torchvision package.

`python main_imagenet.py --bs=84 --do_flip --exp_name="test_rn50_3" --r_feature=0.01 --arch_name="resnet50" --fp16 --verifier`
