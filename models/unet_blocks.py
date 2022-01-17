import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from models.initialization import init_weights
from models.BCRNN import Conv2dFT


def ConvBlock(
    input_dim, 
    output_dim, 
    kernel_size=3,
    stride=1,
    padding=1, 
    use_bn=1,
    slim=0,  # flag to use only one convolution
    convFT=0  # flag to use Conv2dFT
):

    layers = []

    if convFT:
        layers.append(Conv2dFT(input_dim, output_dim, kernel_size))
    else:
        layers.append(nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding))
    if use_bn == 1:
        layers.append(nn.BatchNorm2d(output_dim))
    elif use_bn == 2:
        layers.append(nn.GroupNorm(output_dim, output_dim))
    layers.append(nn.ReLU(inplace=True))

    if not slim:
        if convFT:
            layers.append(Conv2dFT(input_dim, output_dim, kernel_size))
        else:
            layers.append(nn.Conv2d(output_dim, output_dim, kernel_size, stride, padding))
        if use_bn == 1:
            layers.append(nn.BatchNorm2d(output_dim))
        elif use_bn == 2:
            layers.append(nn.GroupNorm(output_dim, output_dim))
        layers.append(nn.ReLU(inplace=False))
    return layers


class DownConvBlock(nn.Module):

    def __init__(
        self, 
        input_dim, 
        output_dim, 
        kernel_size=3,
        stride=1,
        padding=1, 
        use_bn=1, 
        pool=True,
        slim=False,
        convFT=0
    ):

        super(DownConvBlock, self).__init__()
        self.layers = []

        if pool:
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            # self.layers.append(nn.Conv2d(input_dim, input_dim, kernel_size=2, stride=2, padding=0))

        convBlock = ConvBlock(input_dim, output_dim, kernel_size, stride, padding, use_bn, slim, convFT)

        for layer in convBlock:
            self.layers.append(layer)

        self.layers = nn.Sequential(*self.layers)

        self.layers.apply(init_weights)

    def forward(self, inputs):

        return self.layers(inputs)


class DownConvBlock2(nn.Module):
    '''
        DownConvBlock with additional concatenated input features from MultiLevelBCRNNlayer
    '''
    def __init__(
        self, 
        input_dim, 
        output_dim, 
        kernel_size=3,
        stride=1,
        padding=1, 
        use_bn=1, 
        pool=True,
        slim=False,
        convFT=0
    ):

        super(DownConvBlock2, self).__init__()
        self.conv_layers = []
        if pool:
            self.downsampling = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        convBlock = ConvBlock(output_dim, output_dim, kernel_size, stride, padding, use_bn, slim, convFT)
        for layer in convBlock:
            self.conv_layers.append(layer)
        self.conv_layers = nn.Sequential(*self.conv_layers)

    def forward(self, inputs, features):
        inputs = self.downsampling(inputs)
        inputs = torch.cat([inputs, features], 1)
        outputs = self.conv_layers(inputs)
        return outputs


class UpConvBlock(nn.Module):

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
        
        super(UpConvBlock, self).__init__()
        self.use_deconv = use_deconv

        if self.use_deconv:
            self.upconv_layer = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=2, stride=2)
        else:
            self.upconv_layer = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding)
        self.upconv_layer.apply(init_weights)

        self.conv_block = DownConvBlock(input_dim, output_dim, kernel_size, stride, padding, use_bn, pool=False, slim=slim, convFT=convFT)

    def forward(self, right, left):
        
        if self.use_deconv:
            right = self.upconv_layer(right)
        else:
            right = nn.functional.interpolate(right, mode='nearest', scale_factor=2)
            right = self.upconv_layer(right)
        
        left_shape = left.size()
        right_shape = right.size()
        padding = (left_shape[3] - right_shape[3], 0, left_shape[2] - right_shape[2], 0)

        right_pad = nn.ConstantPad2d(padding, 0)
        right = right_pad(right)
        out = torch.cat([right, left], 1)
        out =  self.conv_block(out)

        return out

