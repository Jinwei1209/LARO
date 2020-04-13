"""
    Unrolled network for multi-echo MR parameter estimation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dc_blocks import *
from models.unet_blocks import *
from models.initialization import *
from models.resBlocks import *
from utils.operators import *
from utils.data import *


class MultiEchoDC(nn.Module):


    def __init__(
        self,
        input1_channels,
        filter1_channels,
        output1_channels,  # equivalent to input2_channels
        filter2_channels,
        lambda_dll2, # initializing lambda_dll2
        gd_stepsize,
        K=1
    ):
        super(MultiEchoDC, self).__init__()
        # wrnup resnet to provide initialization of optimization
        self.resnet_init = []
        layers = ResBlock(input1_channels, filter1_channels, output_dim=output1_channels, use_norm=2)
        for layer in layers:
            self.resnet_init.append(layer)
        self.resnet_init = nn.Sequential(*self.resnet_init)
        self.resnet_init.apply(init_weights)
        # prior resnet to do l2-regularized optimization
        self.resnet_prior = []
        layers = ResBlock(output1_channels, filter2_channels, output_dim=output1_channels, use_norm=2)
        for layer in layers:
            self.resnet_prior.append(layer)
        self.resnet_prior = nn.Sequential(*self.resnet_prior)
        self.resnet_prior.apply(init_weights)

        self.K = K
        self.lambda_dll2 = nn.Parameter(torch.tensor(lambda_dll2), requires_grad=True).float()
        self.gd_stepsize = nn.Parameter(torch.tensor(gd_stepsize), requires_grad=True).float()

    def forward(self, mask, csm, kdata):
        device = kdata.get_device()
        zero_filled = torch.ifft(kdata, signal_ndim=2)
        init_start = torch.sum(
            cplx_mlpy(zero_filled, cplx_conj(csm)),
            dim=1,
            keepdim=False
        )
        init_start = torch.cat((init_start[..., 0], init_start[..., 1]), dim=1)
        para_start = self.resnet_init(init_start)
        
        lambda_dll2 = self.lambda_dll2.to(device)
        lambda_dll2 = lambda_dll2[None, :, None, None]
        lambda_dll2 = lambda_dll2.repeat(csm.shape[0], 1, csm.shape[3], csm.shape[4])

        gd_stepsize = self.gd_stepsize.to(device)
        gd_stepsize = gd_stepsize[None, :, None, None]
        gd_stepsize = gd_stepsize.repeat(csm.shape[0], 4, csm.shape[3], csm.shape[4])

        paras = []
        paras_prior = []
        para = para_start
        for i in range(self.K):
            M_0 = para[:, 0:1, ...]
            R_2 = para[:, 1:2, ...]
            phi_0 = para[:, 2:3, ...]
            f = para[:, 3:4, ...]
            para_prior = self.resnet_prior(para)
            my_isnan(para_prior, i)
            # generate operator
            operators = OperatorsMultiEcho(mask, csm, M_0, R_2, phi_0, f)
            kdata_generated = operators.forward_operator()
            my_isnan(kdata_generated, i)
            kdata_diff = kdata_generated - kdata
            # calculate gradient
            gradient_fidelity = operators.jacobian_conj(kdata_diff)
            gradient_prior = lambda_dll2  * (para - para_prior)
            gradient_total = gradient_fidelity + gradient_prior
            # gradient descent step
            para = para - gd_stepsize * gradient_total
            paras.append(para)
            paras_prior.append(para_prior)
        return para_start, paras[-1]
