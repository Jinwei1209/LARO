import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dc_blocks import *
from models.unet_blocks import *
from models.initialization import *


def ResBlock(
    input_dim, 
    filter_dim, 
    kernel_size=3,
    stride=1,
    padding=1, 
    use_bn=1,
    N = 5
):  
    layers = []

    layers.append(nn.Conv2d(input_dim, filter_dim, kernel_size, stride, padding))
    if use_bn:
        layers.append(nn.BatchNorm2d(filter_dim))
    layers.append(nn.ReLU(inplace=True))

    for i in range(N-1):
        layers.append(nn.Conv2d(filter_dim, filter_dim, kernel_size, stride, padding))
        if use_bn:
            layers.append(nn.BatchNorm2d(filter_dim))
        layers.append(nn.ReLU(inplace=True))

    layers.append(nn.Conv2d(filter_dim, input_dim, kernel_size, stride, padding))
    if use_bn:
        layers.append(nn.BatchNorm2d(input_dim))

    return layers


class Resnet_with_DC(nn.Module):


    def __init__(
        self,
        input_channels,
        filter_channels,
        lambda_dll2, # initializing lambda_dll2
        K=1
    ):
        super(Resnet_with_DC, self).__init__()
        self.resnet_block = []
        layers = ResBlock(input_channels, filter_channels)
        for layer in layers:
            self.resnet_block.append(layer)
        self.resnet_block = nn.Sequential(*self.resnet_block)
        self.resnet_block.apply(init_weights)

        self.K = K
        self.lambda_dll2 = nn.Parameter(torch.ones(1)*lambda_dll2, requires_grad=True)
        # self.lambda_dll2 = torch.tensor(lambda_dll2)


    def forward(self, x, csms, masks):

        device = x.get_device()
        x_start = x
        self.lambda_dll2 = self.lambda_dll2.to(device)
        A = Back_forward(csms, masks, self.lambda_dll2)
        for i in range(self.K):
            x_block = self.resnet_block(x)
            x_block = x - x_block
            rhs = x_start + self.lambda_dll2*x_block
            dc_layer = DC_layer(A, rhs)
            x = dc_layer.CG_iter()
            
        return x
