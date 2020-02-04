'''
    straight strough estimation of pmask for T2w images (2D cartesian along two phase encoding direction)
    start on 02/03/2020
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.dc_blocks import *
from models.unet_blocks import *
from models.initialization import *
from models.resBlocks import *
from models.straight_through_layers import *
        

class DC_ST_Pmask(nn.Module):

    def __init__(
        self,
        input_channels,
        filter_channels,
        lambda_dll2, # initializing lambda_dll2
        ncoil=32,
        nrow=256,
        ncol=192,
        flag_ND=2, # 0 for 1D Cartesian along row direciton, 1 for 1D Cartesian along column direction, 
                   # 2 for 2D Cartesian, 3 for variable density random
        K=1,
        unc_map=False,
        slope=0.25,
        passSigmoid=False,
        stochasticSampling=True,
        rescale=False,
        samplingRatio = 0.1, # sparsity level of the sampling mask
    ):
        super(DC_ST_Pmask, self).__init__()
        self.resnet_block = []
        layers = ResBlock(input_channels, filter_channels, use_norm=2, unc_map=unc_map)
        for layer in layers:
            self.resnet_block.append(layer)
        self.resnet_block = nn.Sequential(*self.resnet_block)
        self.resnet_block.apply(init_weights)
        self.K = K
        self.unc_map = unc_map
        self.lambda_dll2 = nn.Parameter(torch.ones(1)*lambda_dll2, requires_grad=True)
        self.slope = slope
        self.passSigmoid = passSigmoid
        self.stochasticSampling = stochasticSampling
        self.ncoil = ncoil
        self.nrow = nrow
        self.ncol = ncol
        self.flag_ND = flag_ND
        self.rescale = rescale
        self.samplingRatio = samplingRatio
        if flag_ND == 0:
            temp = (torch.rand(nrow)-0.5)*30
            temp[nrow//2-13 : nrow//2+12] = 15
        elif flag_ND == 1:
            temp = (torch.rand(ncol)-0.5)*30
            temp[ncol//2-13 : ncol//2+12] = 15
        elif flag_ND == 2:
            temp1 = (torch.rand(nrow)-0.5)*30
            temp2 = (torch.rand(ncol)-0.5)*30
            temp1[nrow//2-13 : nrow//2+12] = 15
            temp2[ncol//2-13 : ncol//2+12] = 15
        elif flag_ND == 3:
            temp = (torch.rand(nrow, ncol)-0.5)*30
            temp[nrow//2-13 : nrow//2+12, ncol//2-13 : ncol//2+12] = 15
        if flag_ND != 2:
            self.weight_parameters = nn.Parameter(temp, requires_grad=True)
        else:
            self.weight_parameters1 = nn.Parameter(temp1, requires_grad=True)
            self.weight_parameters2 = nn.Parameter(temp2, requires_grad=True)

    def rescalePmask(self, Pmask, samplingRatio):
        device = Pmask.get_device()
        xbar = torch.mean(Pmask)
        r = samplingRatio/xbar
        beta = (1-samplingRatio) / (1-xbar)
        le = (r<=1).to(device, dtype=torch.float32)
        return le * Pmask * r + (1-le) * (1 - (1-Pmask) * beta)

    def samplingPmask(self, Pmask_rescaled, flag_ND):
        if flag_ND == 0:
            Mask1D = bernoulliSample.apply(Pmask_rescaled)
            Mask = Mask1D.repeat(self.ncol, 1).transpose(0,1)
        elif flag_ND == 1:
            Mask1D = bernoulliSample.apply(Pmask_rescaled)
            Mask = Mask1D.repeat(self.nrow, 1)
        elif flag_ND == 3:
            Mask = bernoulliSample.apply(Pmask_rescaled)
        return Mask

    def generateMask(self):

        if self.flag_ND != 2:
            if self.passSigmoid:
                self.Pmask = passThroughSigmoid.apply(self.slope * self.weight_parameters)
            else:
                self.Pmask = 1 / (1 + torch.exp(-self.slope * self.weight_parameters))
            if self.rescale:
                self.Pmask_rescaled = self.rescalePmask(self.Pmask, self.samplingRatio)
            else:
                self.Pmask_rescaled = self.Pmask
            self.Mask = self.samplingPmask(self.Pmask_rescaled, self.flag_ND)
            return self.Mask

        elif self.flag_ND == 2:
            if self.passSigmoid:
                self.Pmask1 = passThroughSigmoid.apply(self.slope * self.weight_parameters1)
                self.Pmask2 = passThroughSigmoid.apply(self.slope * self.weight_parameters2)
            else:
                self.Pmask1 = 1 / (1 + torch.exp(-self.slope * self.weight_parameters1))
                self.Pmask2 = 1 / (1 + torch.exp(-self.slope * self.weight_parameters2))
            self.Pmask = self.Pmask1.repeat(self.ncol, 1).transpose(0,1) + self.Pmask2.repeat(self.nrow, 1)
            if self.rescale:
                self.Pmask_rescaled1 = self.rescalePmask(self.Pmask1, np.sqrt(self.samplingRatio))
                self.Pmask_rescaled2 = self.rescalePmask(self.Pmask2, np.sqrt(self.samplingRatio))
            else:
                self.Pmask_rescaled1 = self.Pmask1
                self.Pmask_rescaled2 = self.Pmask2
            Mask1 = self.samplingPmask(self.Pmask_rescaled1, flag_ND=0)
            Mask2 = self.samplingPmask(self.Pmask_rescaled2, flag_ND=1)
            self.Mask = Mask1 * Mask2
            return self.Mask
        
    def At(self, kdata, mask, csm):
        device = kdata.get_device()
        self.ncoil = csm.shape[1]
        self.nrow = csm.shape[2] 
        self.ncol = csm.shape[3]
        self.flip = torch.ones([self.nrow, self.ncol, 1]) 
        self.flip = torch.cat((self.flip, torch.zeros(self.flip.shape)), -1).to(device)
        self.flip[::2, ...] = - self.flip[::2, ...] 
        self.flip[:, ::2, ...] = - self.flip[:, ::2, ...]
        temp = cplx_mlpy(kdata, mask)
        coilImgs = torch.ifft(temp, 2)
        coilImgs = fft_shift_row(coilImgs, self.nrow) # for GE kdata
        coilImgs = cplx_mlpy(coilImgs, self.flip) # for GE kdata
        coilComb = torch.sum(
            cplx_mlpy(coilImgs, cplx_conj(csm)),
            dim=1,
            keepdim=False
        )
        coilComb = coilComb.permute(0, 3, 1, 2)
        return coilComb

    def forward(self, kdata, csms):
        device = kdata.get_device()
        self.lambda_dll2 = self.lambda_dll2.to(device)
        if self.flag_ND != 2:
            self.weight_parameters = self.weight_parameters.to(device)
        else:
            self.weight_parameters1 = self.weight_parameters1.to(device)
            self.weight_parameters2 = self.weight_parameters2.to(device)
        masks = self.generateMask()[None, :, :, None]
        self.Pmask = self.Pmask
        self.masks = self.Mask

        # keep the calibration region
        masks[:, masks.size()[1]//2-13:masks.size()[1]//2+12,  masks.size()[2]//2-13:masks.size()[2]//2+12, :] = 1
        
        # to complex data
        masks = torch.cat((masks, torch.zeros(masks.shape).to(device)),-1)
        # add coil dimension
        masks = torch.cat(self.ncoil*[masks])
        x = self.At(kdata, masks, csms)    
        x_start = x
        self.lambda_dll2 = self.lambda_dll2.to(device)
        A = Back_forward(csms, masks, self.lambda_dll2)
        Xs = []
        Unc_maps = []
        for i in range(self.K):
            x_block = self.resnet_block(x)
            x_block1 = x - x_block[:, 0:2, ...]
            rhs = x_start + self.lambda_dll2*x_block1
            dc_layer = DC_layer(A, rhs)
            x = dc_layer.CG_iter()
            Xs.append(x)
            if self.unc_map:
                Unc_maps.append(x_block[:, 2:4, ...])
        if self.unc_map:
            return Xs, Unc_maps
        else:
            return Xs