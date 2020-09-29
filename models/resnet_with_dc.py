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
from models.danet import daBlock  
from models.fa import faBlockNew
from models.unet import *
from models.straight_through_layers import *
from utils.data import *
from utils.operators import *


class Resnet_with_DC(nn.Module):
    '''
        For Cardiac QSM data
    '''

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


class Resnet_with_DC2(nn.Module):
    '''
        For multi_echo GRE brain data
    '''
    def __init__(
        self,
        input_channels,
        filter_channels,
        lambda_dll2, # initializing lambda_dll2
        lambda_tv=1e-3,
        rho_penalty=1e-2,
        necho=10, # number of echos in the data
        nrow=206,
        ncol=80,
        ncoil=12,
        K=1,  # number of unrolls
        echo_cat=1, # flag to concatenate echo dimension into channel
        flag_solver=0,  # 0 for deep Quasi-newton, 1 for deep ADMM,
                        # 2 for TV Quasi-newton, 3 for TV ADMM.
        flag_precond=0, # flag to use the preconditioner in the CG layer
        flag_loupe=0, # 1: same mask across echos, 2: mask for each echo
        slope=0.25,
        passSigmoid=0,
        stochasticSampling=1,
        rescale=1,
        samplingRatio=0.2, # sparsity level of the sampling mask
        # att=0, # flag to use attention-based denoiser
        # random=0, # flag to multiply the input data with a random complex number
    ):
        super(Resnet_with_DC2, self).__init__()
        self.resnet_block = []
        self.necho = necho
        self.nrow = nrow
        self.ncol = ncol
        self.ncoil = ncoil
        self.echo_cat = echo_cat
        self.flag_solver = flag_solver
        self.flag_precond = flag_precond
        self.flag_loupe = flag_loupe
        self.slope = slope
        self.passSigmoid = passSigmoid
        self.stochasticSampling = stochasticSampling
        self.rescale = rescale
        self.samplingRatio = samplingRatio
        # self.att = att
        # self.random = random

        if self.flag_solver <= 1:
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
            self.lambda_dll2 = nn.Parameter(torch.ones(1)*lambda_dll2, requires_grad=True)
        
        elif self.flag_solver == 2:
            self.lambda_dll2 = nn.Parameter(torch.ones(1)*lambda_dll2, requires_grad=False)
        
        elif self.flag_solver == 3:
            self.rho_penalty = nn.Parameter(torch.ones(1)*rho_penalty, requires_grad=False)
            self.lambda_tv = nn.Parameter(torch.ones(1)*lambda_tv, requires_grad=False)
        
        # if self.att == 1:
        #     self.attBlock = daBlock(input_channels, filter_channels//8, \
        #                             out_channels=input_channels, use_norm=2)

        self.K = K

        # flag for using preconditioning:
        if self.flag_precond == 1:
            print('Apply preconditioning in the CG block')
            self.preconditioner = Unet(2*self.necho, 2*self.necho, num_filters=[2**i for i in range(4, 8)])

        # flag for mask learning strategy
        if self.flag_loupe == 1:
            temp = (torch.rand(self.nrow, self.ncol)-0.5)*30
            temp[self.nrow//2-13 : self.nrow//2+12, self.ncol//2-13 : self.ncol//2+12] = 15
            self.weight_parameters = nn.Parameter(temp, requires_grad=True)
        elif self.flag_loupe == 2:
            temp = (torch.rand(self.necho, self.nrow, self.ncol)-0.5)*30
            temp[:, self.nrow//2-13 : self.nrow//2+12, self.ncol//2-13 : self.ncol//2+12] = 15
            self.weight_parameters = nn.Parameter(temp, requires_grad=True)

    def generateMask(self, weight_parameters):
        if self.passSigmoid:
            Pmask = passThroughSigmoid.apply(self.slope * weight_parameters)
        else:
            Pmask = 1 / (1 + torch.exp(-self.slope * weight_parameters))
        if self.rescale:
            Pmask_rescaled = self.rescalePmask(Pmask, self.samplingRatio)
        else:
            Pmask_rescaled = Pmask

        masks = self.samplingPmask(Pmask_rescaled)[:, :, None] # (nrow, ncol, 1)
        # keep central calibration region to 1
        masks[self.nrow//2-13:self.nrow//2+12, self.ncol//2-13:self.ncol//2+12, :] = 1
        # to complex data
        masks = torch.cat((masks, torch.zeros(masks.shape).to('cuda')),-1) # (nrow, ncol, 2)
        # add echo dimension
        masks = masks[None, ...] # (1, nrow, ncol, 2)
        masks = torch.cat(self.necho*[masks]) # (necho, nrow, ncol, 2)
        # add coil dimension
        masks = masks[None, ...] # (1, necho, nrow, ncol, 2)
        masks = torch.cat(self.ncoil*[masks]) # (ncoil, necho, nrow, ncol, 2)
        # add batch dimension
        masks = masks[None, ...] # (1, ncoil, necho, nrow, ncol, 2)
        return masks

    def rescalePmask(self, Pmask, samplingRatio):
        xbar = torch.mean(Pmask)
        r = samplingRatio / xbar
        beta = (1-samplingRatio) / (1-xbar)
        # le = (r<=1).to('cuda', dtype=torch.float32)
        le = (r<=1).float()
        return le * Pmask * r + (1-le) * (1 - (1-Pmask) * beta)

    def samplingPmask(self, Pmask_rescaled):
        if self.stochasticSampling:
            Mask = bernoulliSample.apply(Pmask_rescaled)
        else:
            thresh = torch.rand(Pmask_rescaled.shape).to('cuda')
            Mask = 1/(1+torch.exp(-12*(Pmask_rescaled-thresh)))
        return Mask

    def forward(self, kdatas, csms, masks, flip):
        # generate sampling mask
        if self.flag_loupe == 1:
            masks = self.generateMask(self.weight_parameters)
            self.Pmask = 1 / (1 + torch.exp(-self.slope * self.weight_parameters))
            self.Mask = masks[0, 0, 0, :, :, 0]
        elif self.flag_loupe == 2:
            masks = torch.zeros(1, self.ncoil, self.necho, self.nrow, self.ncol, 2).to('cuda')
            for echo in range(self.necho):
                masks[:, :, echo, ...] = self.generateMask(self.weight_parameters[echo, ...])[:, :, echo, ...]
            self.Pmask = 1 / (1 + torch.exp(-self.slope * self.weight_parameters)).permute(1, 2, 0)
            self.Mask = masks[0, 0, :, :, :, 0].permute(1, 2, 0)

        # input
        x = backward_multiEcho(kdatas, csms, masks, flip, self.echo_cat)
        x_start = x
        if self.echo_cat == 0:
            x_start_ = torch_channel_concate(x_start)
        else:
            x_start_ = x_start

        # generate preconditioner
        if self.flag_precond == 1:
            precond = 3 / (1 + torch.exp(-0.1 * self.preconditioner(x_start_))) + 1
            precond = torch_channel_deconcate(precond)
            # precond[:, 1, ...] = 0
            self.precond = precond
        else:
            self.precond = 0

        # Deep Quasi-newton
        if self.flag_solver == 0:
            A = Back_forward_multiEcho(csms, masks, flip, 
                                    self.lambda_dll2, self.echo_cat)
            Xs = []
            for i in range(self.K):
                # if self.random:
                #     mag = (1 + torch.randn(1)/3).to(device)
                #     phase = (torch.rand(1) * 3.14/2 - 3.14/4).to(device)
                #     factor = torch.cat((mag*torch.cos(phase), mag*torch.sin(phase)), 0)[None, :, None, None, None]

                #     if self.echo_cat == 0:
                #         x = mlpy_in_cg(x, factor)  # for echo_cat=0
                #     elif self.echo_cat == 1:
                #         x = torch_channel_concate(mlpy_in_cg(torch_channel_deconcate(x), factor))  # for echo_cat=1

                # if i != self.K // 2:
                #     x_block = self.resnet_block(x)
                # else:
                #     if self.att == 1:
                #         x_block = self.attBlock(x)
                #     else:
                #         x_block = self.resnet_block(x)

                x_block = self.resnet_block(x)
                x_block1 = x - x_block

                # if self.random:
                #     factor = torch.cat((1/mag*torch.cos(phase), -1/mag*torch.sin(phase)), 0)[None, :, None, None, None]
                #     if self.echo_cat == 0:
                #         x = mlpy_in_cg(x, factor)  # for echo_cat=0
                #     elif self.echo_cat == 1:
                #         x = torch_channel_concate(mlpy_in_cg(torch_channel_deconcate(x), factor))  # for echo_cat=1

                rhs = x_start + self.lambda_dll2*x_block1
                dc_layer = DC_layer_multiEcho(A, rhs, echo_cat=self.echo_cat,
                        flag_precond=self.flag_precond, precond=self.precond)
                x = dc_layer.CG_iter()

                if self.echo_cat:
                    x = torch_channel_concate(x)
                Xs.append(x)
            return Xs

        # Deep ADMM
        elif self.flag_solver == 1:
            A = Back_forward_multiEcho(csms, masks, flip, 
                                    self.lambda_dll2, self.echo_cat)
            Xs = []
            uk = torch.zeros(x_start.size()).to('cuda')
            for i in range(self.K):
                # update auxiliary variable v
                v_block = self.resnet_block(x+uk/self.lambda_dll2)
                v_block1 = x + uk/self.lambda_dll2 - v_block
                # update x using CG block
                x0 = v_block1 - uk/self.lambda_dll2
                rhs = x_start + self.lambda_dll2*x0
                dc_layer = DC_layer_multiEcho(A, rhs, echo_cat=self.echo_cat,
                        flag_precond=self.flag_precond, precond=self.precond)
                x = dc_layer.CG_iter()
                if self.echo_cat:
                    x = torch_channel_concate(x)
                Xs.append(x)
                # update dual variable uk
                uk = uk + self.lambda_dll2*(x - v_block1)
            return Xs

        # TV Quasi-newton
        elif self.flag_solver == 2:
            A = Back_forward_multiEcho(csms, masks, flip, 
                                    self.lambda_dll2, self.echo_cat)
            Xs = []
            for i in range(self.K):
                rhs = x_start - A.AtA(x, use_dll2=3)
                dc_layer = DC_layer_multiEcho(A, rhs, echo_cat=self.echo_cat,
                    flag_precond=self.flag_precond, precond=self.precond, use_dll2=3)
                delta_x = dc_layer.CG_iter(max_iter=10)
                if self.echo_cat:
                    delta_x = torch_channel_concate(delta_x)
                x = x + delta_x
                Xs.append(x)
            return Xs

        # TV ADMM
        elif self.flag_solver == 3:
            A = Back_forward_multiEcho(csms, masks, flip, 
                                    self.rho_penalty, self.echo_cat)
            Xs = []
            wk = torch.zeros(x_start.size()+(2,)).to('cuda')
            etak = torch.zeros(x_start.size()+(2,)).to('cuda')
            zeros_ = torch.zeros(x_start.size()+(2,)).to('cuda')
            for i in range(self.K):
                # update auxiliary variable wk through threshold
                ek = gradient(x) + etak/self.rho_penalty
                wk = ek.sign() * torch.max(torch.abs(ek) - self.lambda_tv/self.rho_penalty, zeros_)

                x_old = x
                # update x using CG block
                rhs = x_start + self.rho_penalty*divergence(wk) - divergence(etak)
                dc_layer = DC_layer_multiEcho(A, rhs, echo_cat=self.echo_cat,
                            flag_precond=self.flag_precond, precond=self.precond, use_dll2=2)
                x = dc_layer.CG_iter(max_iter=10)
                if self.echo_cat:
                    x = torch_channel_concate(x)
                Xs.append(x)
                # update dual variable etak
                etak = etak + self.rho_penalty * (gradient(x) - wk)
            return Xs