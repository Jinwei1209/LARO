import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dc_blocks import *
from models.unet_blocks import *
from models.initialization import *
from models.resBlocks import *


class Resnet_with_DC(nn.Module):


    def __init__(
        self,
        input_channels,
        filter_channels,
        lambda_dll2, # initializing lambda_dll2
        K=1,
        pre_dc_map=False,
        unc_map=False
    ):
        super(Resnet_with_DC, self).__init__()
        self.resnet_block = []
        layers = ResBlock(input_channels, filter_channels, unc_map=unc_map)
        for layer in layers:
            self.resnet_block.append(layer)
        self.resnet_block = nn.Sequential(*self.resnet_block)
        self.resnet_block.apply(init_weights)
        self.K = K
        self.unc_map = unc_map
        self.pre_dc_map = pre_dc_map
        self.lambda_dll2 = nn.Parameter(torch.ones(1)*lambda_dll2, requires_grad=True)
        # self.lambda_dll2 = torch.tensor(lambda_dll2)

    def forward(self, x, csms, masks):

        device = x.get_device()
        x_start = x
        self.lambda_dll2 = self.lambda_dll2.to(device)
        A = Back_forward(csms, masks, self.lambda_dll2)
        Xs = []
        Unc_maps = []
        X_refs = []
        for i in range(self.K):
            x_block = self.resnet_block(x)
            x_block1 = x - x_block[:, 0:2, ...]
            rhs = x_start + self.lambda_dll2*x_block1
            dc_layer = DC_layer(A, rhs)
            x = dc_layer.CG_iter()
            Xs.append(x)
            X_refs.append(x_block1)
            if self.unc_map:
                Unc_maps.append(x_block[:, 2, ...])
        if self.unc_map:
            if self.pre_dc_map:
                return Xs, Unc_maps, X_refs
            else:
                return Xs, Unc_maps
        else:
            if self.pre_dc_map:
                return Xs, X_refs
            else:
                return Xs
