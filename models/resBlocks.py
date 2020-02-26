import torch
import torch.nn as nn
import torch.nn.functional as F
from models.initialization import *


def ResBlock(
    input_dim, 
    filter_dim, 
    kernel_size=3,
    stride=1,
    padding=1, 
    use_norm=1,  # 0 for no, 1 for batchnorm, 2 for instant norm
    N=5,
    unc_map=False
):  
    layers = []

    layers.append(nn.Conv2d(input_dim, filter_dim, kernel_size, stride, padding))
    if use_norm == 1:
        layers.append(nn.BatchNorm2d(filter_dim))
    elif use_norm == 2:
        layers.append(nn.GroupNorm(filter_dim, filter_dim))
    layers.append(nn.ReLU(inplace=True))

    for i in range(N-1):
        layers.append(nn.Conv2d(filter_dim, filter_dim, kernel_size, stride, padding))
        if use_norm == 1:
            layers.append(nn.BatchNorm2d(filter_dim))
        elif use_norm == 2:
            layers.append(nn.GroupNorm(filter_dim, filter_dim))
        layers.append(nn.ReLU(inplace=True))
    if unc_map:
        layers.append(nn.Conv2d(filter_dim, input_dim+2, 1))
    else:
        layers.append(nn.Conv2d(filter_dim, input_dim, 1))

    return layers


class ResBlock2(nn.Module):

    def __init__(
        self,
        input_dim, 
        filter_dim, 
        kernel_size=3,
        stride=1,
        padding=1, 
        use_norm=1  # 0 for no, 1 for batchnorm, 2 for instant norm
    ):

        super(ResBlock2, self).__init__()
        self.layers = []
        self.input_dim = input_dim
        self.filter_dim = filter_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_norm = use_norm

        self.basicBlock1 = self._basicBlock(self.input_dim, self.filter_dim)
        # self.img2latent = self._conv2d(self.input_dim, self.filter_dim)

        self.basicBlock2 = self._basicBlock(self.filter_dim, self.filter_dim)
        self.basicBlock3 = self._basicBlock(self.filter_dim, self.filter_dim)
        self.basicBlock4 = self._basicBlock(self.filter_dim, self.filter_dim)

        self.basicBlock5 = self._basicBlock(self.filter_dim, self.input_dim)
        # self.latent2img = self._conv2d(self.filter_dim, self.input_dim)


    def _basicBlock(self, input_dim, output_dim):
        layers = []
        if input_dim <= output_dim:
            layers.append(nn.Conv2d(
                input_dim, 
                output_dim, 
                self.kernel_size, 
                self.stride, 
                self.padding)
            )
            if self.use_norm == 1:
                layers.append(nn.BatchNorm2d(output_dim))
            elif self.use_norm == 2:
                layers.append(nn.GroupNorm(output_dim, output_dim))
            layers.append(nn.ReLU(inplace=True))
        else:
            layers.append(nn.Conv2d(input_dim, output_dim, 1))
        basicBlock = nn.Sequential(*layers)
        basicBlock.apply(init_weights)
        return basicBlock

    def _conv2d(self, input_dim, output_dim):
        layers = []
        layers.append(nn.Conv2d(input_dim, output_dim, 1))
        layer_conv2d = nn.Sequential(*layers)
        layer_conv2d.apply(init_weights)
        return layer_conv2d


    def forward(self, x):
        # identity = self.img2latent(x)
        x = self.basicBlock1(x)
        # x = out + identity

        identity = x
        out = self.basicBlock2(x)
        x = out + identity

        identity = x
        out = self.basicBlock3(x)
        x = out + identity

        identity = x
        out = self.basicBlock4(x)
        x = out + identity

        # identity = self.latent2img(x)
        x = self.basicBlock5(x)
        # x = out + identity
        return x

