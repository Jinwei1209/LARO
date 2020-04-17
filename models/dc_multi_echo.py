"""
    Unrolled network for multi-echo MR parameter estimation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dc_blocks import *
from models.unet import *
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
        filter2_channels,
        lambda_dll2,
        gd_stepsize,
        K=1
    ):
        super(MultiEchoDC, self).__init__()
        # two warmup resnets to provide initialization of optimization
        # self.resnet_init_mag = []
        # layers = ResBlock(input1_channels, filter1_channels, output_dim=2, use_norm=2)
        # for layer in layers:
        #     self.resnet_init_mag.append(layer)
        # self.resnet_init_mag = nn.Sequential(*self.resnet_init_mag)
        # self.resnet_init_mag.apply(init_weights)

        # self.resnet_init_phase = []
        # layers = ResBlock(input1_channels, filter1_channels, output_dim=2, use_norm=2)
        # for layer in layers:
        #     self.resnet_init_phase.append(layer)
        # self.resnet_init_phase = nn.Sequential(*self.resnet_init_phase)
        # self.resnet_init_phase.apply(init_weights)

        self.resnet_init = Unet(
            input_channels=input1_channels*2, 
            output_channels=4, 
            num_filters=[2**i for i in range(5, 10)],
            use_bn=2,
            use_deconv=0
        )

        # # prior resnet to do l2-regularized optimization
        # self.resnet_prior_M_0 = []
        # layers = ResBlock(1, filter2_channels, output_dim=1, use_norm=2)
        # for layer in layers:
        #     self.resnet_prior_M_0.append(layer)
        # self.resnet_prior_M_0 = nn.Sequential(*self.resnet_prior_M_0)
        # self.resnet_prior_M_0.apply(init_weights)

        # self.resnet_prior_R_2 = []
        # layers = ResBlock(1, filter2_channels, output_dim=1, use_norm=2)
        # for layer in layers:
        #     self.resnet_prior_R_2.append(layer)
        # self.resnet_prior_R_2 = nn.Sequential(*self.resnet_prior_R_2)
        # self.resnet_prior_R_2.apply(init_weights)

        # self.resnet_prior_phi_0 = []
        # layers = ResBlock(1, filter2_channels, output_dim=1, use_norm=2)
        # for layer in layers:
        #     self.resnet_prior_phi_0.append(layer)
        # self.resnet_prior_phi_0 = nn.Sequential(*self.resnet_prior_phi_0)
        # self.resnet_prior_phi_0.apply(init_weights)

        # self.resnet_prior_f = []
        # layers = ResBlock(1, filter2_channels, output_dim=1, use_norm=2)
        # for layer in layers:
        #     self.resnet_prior_f.append(layer)
        # self.resnet_prior_f = nn.Sequential(*self.resnet_prior_f)
        # self.resnet_prior_f.apply(init_weights)

        self.resnet_prior = Unet(
            input_channels=4, 
            output_channels=4, 
            num_filters=[2**i for i in range(5, 10)],
            use_bn=2,
            use_deconv=0
        )

        self.K = K
        self.lambda_dll2 = nn.Parameter(torch.ones(4)*lambda_dll2, requires_grad=True).float()
        self.gd_stepsize = nn.Parameter(torch.ones(1)*gd_stepsize, requires_grad=False).float()

    def forward(self, mask, csm, kdata, mag, phase):
        device = kdata.get_device()
        # option one, two resnet
        # para_start1 = self.resnet_init_mag(mag)
        # para_start2 = self.resnet_init_phase(phase)
        # para_start = torch.cat((para_start1, para_start2), dim=1)

        # option two. one big unet
        para_start = self.resnet_init(torch.cat((mag, phase), dim=1))
        
        lambda_dll2 = self.lambda_dll2.to(device)
        # lambda_dll2 = lambda_dll2[None, :, None, None]
        # lambda_dll2 = lambda_dll2.repeat(csm.shape[0], 1, csm.shape[3], csm.shape[4])

        gd_stepsize = self.gd_stepsize.to(device)
        # gd_stepsize = gd_stepsize[None, :, None, None]
        # gd_stepsize = gd_stepsize.repeat(csm.shape[0], 1, csm.shape[3], csm.shape[4])

        paras = []
        paras_prior = []
        para = para_start
        for i in range(self.K):
            M_0 = para[:, 0:1, ...]
            R_2 = para[:, 1:2, ...]
            phi_0 = para[:, 2:3, ...]
            f = para[:, 3:4, ...]
            # # concatenate priors
            # M_0_prior = self.resnet_prior_M_0(M_0)
            # R_2_prior = self.resnet_prior_R_2(R_2)
            # phi_0_prior = self.resnet_prior_phi_0(phi_0)
            # f_prior = self.resnet_prior_f(f)
            # para_prior = torch.cat((M_0_prior, R_2_prior, phi_0_prior, f_prior), dim=1)
            para_prior = self.resnet_prior(para)
            M_0_prior = para_prior[:, 0:1, ...]
            R_2_prior = para_prior[:, 1:2, ...]
            phi_0_prior = para_prior[:, 2:3, ...]
            f_prior = para_prior[:, 3:4, ...]
            # my_isnan(para_prior, i)
            # gradient prior
            gradient_prior_M0 = lambda_dll2[0]  * (M_0 - M_0_prior)
            gradient_prior_R_2 = lambda_dll2[1]  * (R_2 - R_2_prior)
            gradient_prior_phi_0 = lambda_dll2[2]  * (phi_0 - phi_0_prior)
            gradient_prior_f = lambda_dll2[3]  * (f - f_prior)
            gradient_prior = torch.cat((gradient_prior_M0, gradient_prior_R_2, 
                                        gradient_prior_phi_0, gradient_prior_f), dim=1)
            # generate fidelity operator
            operators = OperatorsMultiEcho(mask, csm, M_0, R_2, phi_0, f)
            kdata_generated = operators.forward_operator()
            # my_isnan(kdata_generated, i)
            kdata_diff = kdata_generated - kdata
            # calculate fidelity gradient
            gradient_fidelity = operators.jacobian_conj(kdata_diff)
            # total gradient
            gradient_total = gradient_fidelity + gradient_prior
            # gradient descent step
            para = para - gd_stepsize * gradient_total
            paras.append(para)
            paras_prior.append(para_prior)
        return para_start, paras_prior, paras
