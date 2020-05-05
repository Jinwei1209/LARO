import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from models.initialization import init_weights


def ConvBlock(
    input_dim, 
    output_dim, 
    kernel_size=3,
    stride=1,
    padding=1, 
    use_bn=1
):

    layers = []

    layers.append(nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding))
    if use_bn == 1:
        layers.append(nn.BatchNorm2d(output_dim))
    elif use_bn == 2:
        layers.append(nn.GroupNorm(output_dim, output_dim))
    layers.append(nn.LeakyReLU(inplace=False))

    layers.append(nn.Conv2d(output_dim, output_dim, kernel_size, stride, padding))
    if use_bn == 1:
        layers.append(nn.BatchNorm2d(output_dim))
    elif use_bn == 2:
        layers.append(nn.GroupNorm(output_dim, output_dim))
    layers.append(nn.LeakyReLU(inplace=False))

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
        pool=True
    ):

        super(DownConvBlock, self).__init__()
        self.layers = []

        if pool:
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            # self.layers.append(nn.Conv2d(input_dim, input_dim, kernel_size=2, stride=2, padding=0))

        convBlock = ConvBlock(input_dim, output_dim, kernel_size, stride, padding, use_bn)

        for layer in convBlock:
            self.layers.append(layer)

        self.layers = nn.Sequential(*self.layers)

        self.layers.apply(init_weights)

    def forward(self, inputs):

        return self.layers(inputs)


class UpConvBlock(nn.Module):

    def __init__(
        self, 
        input_dim, 
        output_dim,
        kernel_size=3,
        stride=1,
        padding=1, 
        use_bn=1,
        use_deconv=1
    ):
        
        super(UpConvBlock, self).__init__()
        self.use_deconv = use_deconv

        if self.use_deconv:
            self.upconv_layer = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=2, stride=2)
        else:
            self.upconv_layer = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding)
        self.upconv_layer.apply(init_weights)

        self.conv_block = DownConvBlock(input_dim, output_dim, kernel_size, stride, padding, use_bn, pool=False)

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

