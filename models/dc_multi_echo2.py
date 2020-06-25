"""
    Unrolled network for multi-echo MR parameter estimation (archived one)
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


class MultiEchoDC2(nn.Module):

    def __init__(
        self,
        filter_channels,
        num_echos,
        lambda_dll2,
        norm_means,  # means for normalization
        norm_stds,  # stds for normalization
        K=1
    ):
        super(MultiEchoDC2, self).__init__()

        # resnet to do l2-regularized optimization
        self.resnet_prior_M_0 = multi_unet(
            input_channels=1, 
            output_channels=1, 
            num_filters=[2**i for i in range(4, 8)],
            use_bn=2,
            use_deconv=0,
            K=K
        )
        self.resnet_prior_R_2 = multi_unet(
            input_channels=1, 
            output_channels=1, 
            num_filters=[2**i for i in range(4, 8)],
            use_bn=2,
            use_deconv=0,
            K=K
        )
        self.resnet_prior_phi_0 = multi_unet(
            input_channels=1, 
            output_channels=1, 
            num_filters=[2**i for i in range(4, 8)],
            use_bn=2,
            use_deconv=0,
            K=K
        )
        self.resnet_prior_f = multi_unet(
            input_channels=1, 
            output_channels=1, 
            num_filters=[2**i for i in range(4, 8)],
            use_bn=2,
            use_deconv=0,
            K=K
        ) 
        
        self.num_echos = num_echos
        self.K = K
        self.lambda_dll2 = nn.Parameter(torch.Tensor(lambda_dll2), requires_grad=True).float()
        self.norm_means = torch.Tensor(norm_means).cuda()
        self.norm_stds = torch.Tensor(norm_stds).cuda()

    def forward(self, inputs, iField):
        device = inputs.get_device()
        paras, paras_prior = [], []
        para = inputs
        self.lambda_dll2 = self.lambda_dll2.to(device)
        for k in range(self.K):
            M_0 = para[:, 0:1, ...]
            R_2 = para[:, 1:2, ...]
            phi_0 = para[:, 2:3, ...]
            f = para[:, 3:4, ...]

            # norm
            M_0_norm = (M_0 - self.norm_means[0]) / self.norm_stds[0]
            R_2_norm = (R_2 - self.norm_means[1]) / self.norm_stds[1]
            phi_0_norm = (phi_0 - self.norm_means[2]) / self.norm_stds[2]
            f_norm = (f - self.norm_means[3]) / self.norm_stds[3]
            # denorm
            M_0_prior = self.resnet_prior_M_0[k](M_0_norm) * self.norm_stds[0] + self.norm_means[0]
            R_2_prior = self.resnet_prior_R_2[k](R_2_norm) * self.norm_stds[1] + self.norm_means[1]
            phi_0_prior = self.resnet_prior_phi_0[k](phi_0_norm) * self.norm_stds[2] + self.norm_means[2]
            f_prior = self.resnet_prior_f[k](f_norm) * self.norm_stds[3] + self.norm_means[3]
            para_prior = torch.cat((M_0_prior, R_2_prior, phi_0_prior, f_prior), dim=1)

            # Initialize operators   
            operators = OperatorsMultiEcho(M_0, R_2, phi_0, f, num_echos=self.num_echos)

            # DC layer for M_0 
            W = operators.forward_operator(flag=0)
            rhs = self.lambda_dll2[0]*M_0_prior + operators.jacobian_conj(iField, flag=1)
            # dc_layer = DC_layer_real(operators, rhs, flag=1, use_dll2=1, lambda_dll2=self.lambda_dll2[0])
            # M_0_new = dc_layer.CG_iter(max_iter=2)
            M_0_new = rhs / operators.AtA(flag=1, use_dll2=1, lambda_dll2=self.lambda_dll2[0])

            # DC layer for R_2
            B = operators.forward_operator(flag=0) - iField
            rhs = - self.lambda_dll2[1]*(R_2-R_2_prior) - operators.jacobian_conj(B, flag=2)
            # dc_layer = DC_layer_real(operators, rhs, flag=2, use_dll2=1, lambda_dll2=self.lambda_dll2[1])
            # R_2_new = R_2 + dc_layer.CG_iter(max_iter=2)
            R_2_new = R_2 + rhs / operators.AtA(flag=2, use_dll2=1, lambda_dll2=self.lambda_dll2[1])

            # DC layer for phi_0
            B = operators.forward_operator(flag=0) - iField
            rhs = - self.lambda_dll2[2]*(phi_0-phi_0_prior) - operators.jacobian_conj(B, flag=3)
            # dc_layer = DC_layer_real(operators, rhs, flag=3, use_dll2=1, lambda_dll2=self.lambda_dll2[2])
            # phi_0_new = phi_0 + dc_layer.CG_iter(max_iter=2)
            phi_0_new = phi_0 + rhs / operators.AtA(flag=3, use_dll2=1, lambda_dll2=self.lambda_dll2[2])

            # DC layer for f
            B = operators.forward_operator(flag=0) - iField
            rhs = - self.lambda_dll2[3]*(f-f_prior) - operators.jacobian_conj(B, flag=4)
            # dc_layer = DC_layer_real(operators, rhs, flag=4, use_dll2=1, lambda_dll2=self.lambda_dll2[3])
            # f_new = f + dc_layer.CG_iter(max_iter=2)
            f_new = f + rhs / operators.AtA(flag=4, use_dll2=1, lambda_dll2=self.lambda_dll2[3])

            # # concatenate
            para = torch.cat((M_0_new, R_2_new, phi_0_new, f_new), dim=1)
            paras.append(para)
            paras_prior.append(para_prior)
        return paras, paras_prior