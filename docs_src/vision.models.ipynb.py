
# coding: utf-8

# ## Computer Vision models zoo

from fastai.gen_doc.nbdoc import *
from fastai.vision.models.darknet import Darknet
from fastai.vision.models.wrn import wrn_22, WideResNet


# On top of the models offered by [torchvision](https://pytorch.org/docs/stable/torchvision/models.html), the fastai library has implementations for the following models:
#
# - Darknet architecture, which is the base of [Yolo v3](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
# - Unet architecture based on a pretrained model. The original unet is described [here](https://arxiv.org/abs/1505.04597), the model implementation is detailed in [`models.unet`](/vision.models.unet.html#vision.models.unet)
# - Wide resnets architectures, as introduced in [this article](https://arxiv.org/abs/1605.07146).

show_doc(Darknet)


# Create a Darknet with blocks of sizes given in `num_blocks`, ending with `num_classes` and using `nf` initial features. Darknet53 uses `num_blocks = [1,2,8,8,4]`.

show_doc(WideResNet)


# Each group contains `N` blocks. `start_nf` the initial number of features. Dropout of `drop_p` is applied in between the two convolutions in each block. The expected input channel size is fixed at 3.

# Structure: initial convolution  ->  `num_groups` x `N` blocks -> final layers of regularization and pooling

#  The first block of each group joins a path containing 2 convolutions with filter size 3x3 (and various regularizations) with another path containing a single convolution with a filter size of 1x1. All other blocks in each group follow the more traditional res_block style, i.e., the input of the path with two convs is added to the output of that path.
#
#  In the first group the stride is 1 for all convolutions. In all subsequent groups the stride in the first convolution of the first block is 2 and then all following convolutions have a stride of 1. Padding is always 1.

show_doc(wrn_22)


# This is a [`WideResNet`](/vision.models.wrn.html#WideResNet) with `num_groups=3`, `N=3`, `k=6` and `drop_p=0.`.
