'''
    straight strough estimation of pmask for T2w images (2D cartesian along two phase encoding direction)
    start on 02/03/2020
'''
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

from models.dc_blocks import *
from models.unet_blocks import *
from models.initialization import init_weights
from models.resBlocks import ResBlock2, ResBlock
from models.unet import *
from models.straight_through_layers import *
from utils.data import cplx_mlpy, cplx_conj, fft_shift_row


class Para2Weight(nn.Module):
    def __init__(self, inC, outC, kernel_size=3):
        super(Para2Weight, self).__init__()
        self.inC = inC
        self.kernel_size = kernel_size
        self.outC = outC
        self.meta_block=nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.kernel_size*self.kernel_size*self.inC*self.outC + self.outC)
        )
        self.meta_block.apply(init_weights)

    def forward(self,x):
        output = self.meta_block(x)
        return output

class Meta_Res_DC(nn.Module):

    def __init__(
        self,
        input_channels,
        filter_channels,
        lambda_dll2,
        lambda_tv=0.01,
        rho_penalty=0.01,
        flag_solver=0, # 0: Classic ADMM solver; 1: deep ADMM with resnet denoiser; 
                       # 2: meta deep ADMM with meta-resnet denoiser
        ncoil=32,
        nrow=256,
        ncol=192,
        K=1,
        contrast='T2',
        samplingRatio=0.1
    ):
        super(Meta_Res_DC, self).__init__()
        self.input_channels = input_channels
        self.filter_channels = filter_channels
        self.K = K
        self.ncoil = ncoil
        self.nrow = nrow
        self.ncol = ncol
        self.contrast = contrast
        self.samplingRatio = samplingRatio
        self.flag_solver = flag_solver
        # flags for different solvers
        if flag_solver == 0:
            self.lambda_tv = nn.Parameter(torch.ones(1)*lambda_tv, requires_grad=True)
            self.rho_penalty = nn.Parameter(torch.ones(1)*rho_penalty, requires_grad=True)
        elif flag_solver == 1:
            self.lambda_dll2 = nn.Parameter(torch.ones(1)*lambda_dll2, requires_grad=True)
            self.resnet_block = []
            layers = ResBlock(input_channels, filter_channels, use_norm=2)
            for layer in layers:
                self.resnet_block.append(layer)
            self.resnet_block = nn.Sequential(*self.resnet_block)
            self.resnet_block.apply(init_weights)
        elif flag_solver == 2:
            self.lambda_dll2 = nn.Parameter(torch.ones(1)*lambda_dll2, requires_grad=True)

            self.P2W_1 = Para2Weight(input_channels, filter_channels)
            self.P2W_2 = Para2Weight(filter_channels, filter_channels)
            self.P2W_3 = Para2Weight(filter_channels, filter_channels)
            self.P2W_4 = Para2Weight(filter_channels, filter_channels)
            self.P2W_5 = Para2Weight(filter_channels, filter_channels)
            self.P2W_6 = Para2Weight(filter_channels, input_channels, kernel_size=1)

            self.norm1 = nn.GroupNorm(filter_channels, filter_channels)
            self.relu1 = nn.ReLU(inplace=True)

            self.norm2 = nn.GroupNorm(filter_channels, filter_channels)
            self.relu2 = nn.ReLU(inplace=True)

            self.norm3 = nn.GroupNorm(filter_channels, filter_channels)
            self.relu3 = nn.ReLU(inplace=True)

            self.norm4 = nn.GroupNorm(filter_channels, filter_channels)
            self.relu4 = nn.ReLU(inplace=True)

            self.norm5 = nn.GroupNorm(filter_channels, filter_channels)
            self.relu5 = nn.ReLU(inplace=True)

    def rescalePmask(self, Pmask, samplingRatio):
        xbar = np.mean(Pmask)
        r = samplingRatio/xbar
        beta = (1-samplingRatio) / (1-xbar)
        le = (r<=1)
        return le * Pmask * r + (1-le) * (1 - (1-Pmask) * beta)
        
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

    def meta_resnet(self, x, Pb):
        # compute weights and bias for each layer
        P2W_1_out = self.P2W_1(Pb)
        res_weights1 = P2W_1_out[:-self.P2W_1.outC].view(self.filter_channels, self.input_channels, 3, 3)
        res_bias1 = P2W_1_out[-self.P2W_1.outC:].view(self.filter_channels)
        
        P2W_2_out = self.P2W_2(Pb)
        res_weights2 = P2W_2_out[:-self.P2W_2.outC].view(self.filter_channels, self.filter_channels, 3, 3)
        res_bias2 = P2W_2_out[-self.P2W_2.outC:].view(self.filter_channels)

        P2W_3_out = self.P2W_3(Pb)
        res_weights3 = P2W_3_out[:-self.P2W_3.outC].view(self.filter_channels, self.filter_channels, 3, 3)
        res_bias3 = P2W_3_out[-self.P2W_3.outC:].view(self.filter_channels)

        P2W_4_out = self.P2W_4(Pb)
        res_weights4 = P2W_4_out[:-self.P2W_4.outC].view(self.filter_channels, self.filter_channels, 3, 3)
        res_bias4 = P2W_4_out[-self.P2W_4.outC:].view(self.filter_channels)

        P2W_5_out = self.P2W_5(Pb)
        res_weights5 = P2W_5_out[:-self.P2W_5.outC].view(self.filter_channels, self.filter_channels, 3, 3)
        res_bias5 = P2W_5_out[-self.P2W_5.outC:].view(self.filter_channels)

        P2W_6_out = self.P2W_6(Pb)
        res_weights6 = P2W_6_out[:-self.P2W_6.outC].view(self.input_channels, self.filter_channels, 1, 1)
        res_bias6 = P2W_6_out[-self.P2W_6.outC:].view(self.input_channels)

        x = nn.functional.conv2d(x, res_weights1, res_bias1, 1, 1)
        x = self.norm1(x)
        x = self.relu1(x)

        x = nn.functional.conv2d(x, res_weights2, res_bias2, 1, 1)
        x = self.norm2(x)
        x = self.relu2(x)

        x = nn.functional.conv2d(x, res_weights3, res_bias3, 1, 1)
        x = self.norm3(x)
        x = self.relu3(x)

        x = nn.functional.conv2d(x, res_weights4, res_bias4, 1, 1)
        x = self.norm4(x)
        x = self.relu4(x)

        x = nn.functional.conv2d(x, res_weights5, res_bias5, 1, 1)
        x = self.norm5(x)
        x = self.relu5(x)

        x = nn.functional.conv2d(x, res_weights6, res_bias6, 1, 0)
        return x

    def forward(self, kdata, csms, pmask_BO, Pb): # pb: input value to Para2Weight
        device = kdata.get_device()
        self.pmask_BO = self.rescalePmask(pmask_BO, self.samplingRatio)
        u = np.random.uniform(0, 1, size=(256, 192))
        masks = np.float32(self.pmask_BO > u)
        masks[128-13:128+12, 96-13:96+12] = 1
        masks = torch.tensor(masks[np.newaxis, ..., np.newaxis]).to(device)
        self.masks = masks
        # to complex data
        masks = torch.cat((masks, torch.zeros(masks.shape).to(device)),-1)
        # add coil dimension
        masks = torch.cat(self.ncoil*[masks])[None, ...]
        x = self.At(kdata, masks, csms)    
        # input
        x_start = x

        # ADMM
        if self.flag_solver == 0:
            self.lambda_tv = self.lambda_tv.to(device)
            self.rho_penalty = self.rho_penalty.to(device)
            A = Back_forward(csms, masks, self.rho_penalty)
            Xs = []
            wk = torch.zeros(x_start.size()+(2,)).to(device)
            etak = torch.zeros(x_start.size()+(2,)).to(device)
            for i in range(self.K):
                # update auxiliary variable wk through threshold
                ek = gradient(x) + etak/self.rho_penalty
                wk = ek.sign() * torch.max(torch.abs(ek) - self.lambda_tv/self.rho_penalty, torch.zeros(ek.size()).to(device))
                # update x using CG block
                rhs = x_start + self.rho_penalty*divergence(wk) - divergence(etak)
                dc_layer = DC_layer(A, rhs, use_dll2=2)
                x = dc_layer.CG_iter(max_iter=20)  # 20/30 for test (30 only for uniform mask)
                Xs.append(x)
                # update dual variable etak
                etak = etak + self.rho_penalty * (gradient(x) - wk)
            return Xs

        # Deep ADMM
        elif self.flag_solver == 1:
            self.lambda_dll2 = self.lambda_dll2.to(device)
            A = Back_forward(csms, masks, self.lambda_dll2)
            Xs = []
            uk = torch.zeros(x_start.size()).to(device)
            for i in range(self.K):
                # update auxiliary variable v
                v_block = self.resnet_block(x+uk/self.lambda_dll2)
                v_block1 = x + uk/self.lambda_dll2 - v_block[:, 0:2, ...]
                # update x using CG block
                x0 = v_block1 - uk/self.lambda_dll2
                rhs = x_start + self.lambda_dll2*x0
                dc_layer = DC_layer(A, rhs, use_dll2=1)
                x = dc_layer.CG_iter()
                Xs.append(x)
                # update dual variable uk
                uk = uk + self.lambda_dll2*(x - v_block1)
            return Xs

        # Meta deep ADMM
        elif self.flag_solver == 2:
            self.lambda_dll2 = self.lambda_dll2.to(device)
            A = Back_forward(csms, masks, self.lambda_dll2)
            Xs = []
            uk = torch.zeros(x_start.size()).to(device)

            for i in range(self.K):
                # update auxiliary variable v
                v_block = self.meta_resnet(x+uk/self.lambda_dll2, Pb)
                v_block1 = x + uk/self.lambda_dll2 - v_block[:, 0:2, ...]
                # update x using CG block
                x0 = v_block1 - uk/self.lambda_dll2
                rhs = x_start + self.lambda_dll2*x0
                dc_layer = DC_layer(A, rhs, use_dll2=1)
                x = dc_layer.CG_iter()
                Xs.append(x)
                # update dual variable uk
                uk = uk + self.lambda_dll2*(x - v_block1)
            return Xs

