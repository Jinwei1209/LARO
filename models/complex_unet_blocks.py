import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from models.initialization import init_weights
from models.BCRNN import Conv2dFT
from models.cplx import *


class ComplexConvBlock(nn.Module):

    def __init__(
        self,
        input_dim, 
        output_dim, 
        kernel_size=3,
        stride=1,
        padding=1, 
        use_bn=2,  # use instance norm 2, use complex instance norm 3
        slim=0,  # flag to use only one convolution
        convFT=0  # flag to use Conv2dFT
    ):
        super(ComplexConvBlock, self).__init__()
        self.use_bn = use_bn
        if convFT:
            self.conv = Conv2dFT(input_dim, output_dim, kernel_size)
        else:
            self.conv = ComplexConv2d(input_dim, output_dim, kernel_size)
        if use_bn == 1:
            self.norm = nn.BatchNorm2d(output_dim*2)
        elif use_bn == 2:
            self.norm = nn.GroupNorm(output_dim*2, output_dim*2)
        elif use_bn == 3:
            self.norm = ComplexInstanceNorm2d(output_dim)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.conv(input)
        (N, _, C, W, H) = x.size()
        if self.use_bn == 1 or self.use_bn == 2:
            x = self.norm(x.view(N, 2*C, W, H)).view(N, 2, C, W, H)
        elif self.use_bn == 3:
            x = self.norm(x)
        x = self.relu(x)
        return x


class ComplexDownConvBlock(nn.Module):

    def __init__(
        self, 
        input_dim, 
        output_dim, 
        kernel_size=3,
        stride=1,
        padding=1, 
        use_bn=1, 
        pool=True,
        poolType=0,  # 0: real/imag pooling; 1: comlpex pooling
        slim=False,
        convFT=0
    ):

        super(ComplexDownConvBlock, self).__init__()

        self.pool = pool
        self.poolType = poolType
        if pool:
            if poolType == 0:
                self.pooling = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            else:
                self.pooling = ComplexMaxPool2d(kernel_size=2, stride=2, padding=0)

        self.convBlock = ComplexConvBlock(input_dim, output_dim, kernel_size, stride, padding, use_bn, slim, convFT)

    def forward(self, input):
        x = input
        (N, _, C, W, H) = x.size()
        if self.pool:
            if self.poolType == 0:
                x = self.pooling(x.view(N, 2*C, W, H)).view(N, 2, C, W//2, H//2)
            else:
                x = self.pooling(x)
        x = self.convBlock(x)
        return x


class ComplexUpConvBlock(nn.Module):

    def __init__(
        self, 
        input_dim, 
        output_dim,
        kernel_size=3,
        stride=1,
        padding=1, 
        use_bn=1,
        use_deconv=1,
        slim=False,
        convFT=0
    ):
        
        super(ComplexUpConvBlock, self).__init__()
        self.use_deconv = use_deconv

        if self.use_deconv:
            self.upconv_layer = ComplexConv2dTrans(input_dim, output_dim, kernel_size=2)

        self.conv_block = ComplexDownConvBlock(input_dim, output_dim, kernel_size, stride, padding, use_bn, pool=False, slim=slim, convFT=convFT)

    def forward(self, right, left):
        
        if self.use_deconv:  # always use deconvolution
            right = self.upconv_layer(right)
        
        left_shape = left.size()
        right_shape = right.size()
        padding = (left_shape[4] - right_shape[4], 0, left_shape[3] - right_shape[3], 0)

        right_pad = nn.ConstantPad2d(padding, 0)
        right = right_pad(right.view(right_shape[0], right_shape[1]*right_shape[2], right_shape[3], right_shape[4]))
        right = right.view(right_shape[0], right_shape[1], right_shape[2], left_shape[3], left_shape[4])
        out = torch.cat([right, left], 2)
        out =  self.conv_block(out)

        return out

