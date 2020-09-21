"""
    MoDL for Cardiac QSM data and multi_echo GRE brain data (kspace)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dc_blocks import *
from models.unet_blocks import *
from models.initialization import *
from models.resBlocks import *
from utils.operators import *

'''
    For Cardiac QSM data
'''
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
        layers = ResBlock(input_channels, filter_channels, use_norm=2)
        for layer in layers:
            self.resnet_block.append(layer)
        self.resnet_block = nn.Sequential(*self.resnet_block)
        self.resnet_block.apply(init_weights)
        self.K = K
        self.lambda_dll2 = nn.Parameter(torch.ones(1)*lambda_dll2, requires_grad=True)

    def forward(self, x, csms, masks):
        device = x.get_device()
        x_start = x
        # self.lambda_dll2 = self.lambda_dll2.to(device)
        A = backward_forward_CardiacQSM(csms, masks, self.lambda_dll2)
        Xs = []
        for i in range(self.K):
            x_block = self.resnet_block(x)
            x_block1 = x - x_block[:, 0:2, ...]
            rhs = x_start + self.lambda_dll2*x_block1
            dc_layer = DC_layer(A, rhs)
            x = dc_layer.CG_iter()
            Xs.append(x)
        return Xs[-1]


'''
    For multi_echo GRE brain data
'''
class Resnet_with_DC2(nn.Module):

    def __init__(
        self,
        input_channels,
        filter_channels,
        lambda_dll2, # initializing lambda_dll2
        K=1,  # number of unrolls
        echo_cat=1, # flag to concatenate echo dimension into channel
    ):
        super(Resnet_with_DC2, self).__init__()
        self.resnet_block = []
        self.echo_cat = echo_cat
        if self.echo_cat == 1:
            layers = ResBlock(input_channels, filter_channels, \
                            output_dim=input_channels, use_norm=2)
        elif self.echo_cat == 0:
            layers = ResBlock_3D(input_channels, filter_channels, \
                            output_dim=input_channels, use_norm=2)
        for layer in layers:
            self.resnet_block.append(layer)
        self.resnet_block = nn.Sequential(*self.resnet_block)
        self.resnet_block.apply(init_weights)
        self.K = K
        self.lambda_dll2 = nn.Parameter(torch.ones(1)*lambda_dll2, requires_grad=True)
    
    def forward(self, x, csms, masks, flip):
        device = x.get_device()
        x_start = x
        # self.lambda_dll2 = self.lambda_dll2.to(device)
        A = Back_forward_multiEcho(csms, masks, flip, 
                                self.lambda_dll2, self.echo_cat)
        Xs = []
        for i in range(self.K):
            x_block = self.resnet_block(x)
            x_block1 = x - x_block
            rhs = x_start + self.lambda_dll2*x_block1
            dc_layer = DC_layer_multiEcho(A, rhs, self.echo_cat)
            x = dc_layer.CG_iter()
            if self.echo_cat:
                x = torch_channel_concate(x)
            Xs.append(x)
        return Xs

