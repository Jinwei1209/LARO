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
        filter_channels,
        num_echos,
        lambda_dll2,
        norm_means,  # means for normalization
        norm_stds,  # stds for normalization
        K=1,
        flag_model=1  # 0: Unet, 1: unrolled net
    ):
        super(MultiEchoDC, self).__init__()
        self.flag_model = flag_model
        if self.flag_model == 1:
            # resnet to do l2-regularized optimization
            # self.resnet_prior_M_0 = multi_unet(
            #     input_channels=1, 
            #     output_channels=1, 
            #     num_filters=[2**i for i in range(4, 8)],
            #     use_bn=2,
            #     use_deconv=1,
            #     K=K
            # )
            self.resnet_prior_R_2 = multi_unet(
                input_channels=1, 
                output_channels=1, 
                num_filters=[2**i for i in range(4, 8)],
                use_bn=2,
                use_deconv=1,
                K=K
            )
            # self.resnet_prior_phi_0 = multi_unet(
            #     input_channels=1, 
            #     output_channels=1, 
            #     num_filters=[2**i for i in range(4, 8)],
            #     use_bn=2,
            #     use_deconv=1,
            #     K=K
            # )
            self.resnet_prior_f = multi_unet(
                input_channels=1, 
                output_channels=1, 
                num_filters=[2**i for i in range(4, 8)],
                use_bn=2,
                use_deconv=1,
                K=K
            )
        elif self.flag_model == 0:
            self.unet = Unet(
                input_channels=2,
                output_channels=2,
                num_filters=[2**i for i in range(5, 10)],
                use_bn=2,
                use_deconv=1
            )

        self.num_echos = num_echos
        self.K = K
        self.lambda_dll2 = nn.Parameter(torch.Tensor(lambda_dll2), requires_grad=False).float()
        self.gd_stepsize = nn.Parameter(torch.ones(1)*5, requires_grad=False).float()
        self.norm_means = torch.Tensor(norm_means).cuda()
        self.norm_stds = torch.Tensor(norm_stds).cuda()

    # def forward(self, mask, csm, kdata, mag, phase):
    #     device = kdata.get_device()
    #     para_start = self.resnet_init(torch.cat((mag, phase), dim=1))
    #     M_0 = para_start[:, 0:1, ...]
    #     R_2 = para_start[:, 1:2, ...]
    #     phi_0 = para_start[:, 2:3, ...]
    #     f = para_start[:, 3:4, ...]
    #     self.lambda_dll2 = self.lambda_dll2.to(device)

    #     paras = []
    #     # DC layer for M_0    
    #     operators = OperatorsMultiEcho(mask, csm, M_0, R_2, phi_0, f, self.lambda_dll2[0])
    #     W = operators.forward_operator()
    #     rhs = self.lambda_dll2[0]*M_0 + operators.jacobian_conj(kdata, flag=1)
    #     dc_layer = DC_layer_real(operators, rhs, flag=1, use_dll2=1)
    #     M_0_new = dc_layer.CG_iter(max_iter=2)
    #     # DC layer for R_2
    #     operators = OperatorsMultiEcho(mask, csm, M_0_new, R_2, phi_0, f, self.lambda_dll2[1])
    #     B = operators.forward_operator() - kdata
    #     rhs = - operators.jacobian_conj(B, flag=2)
    #     dc_layer = DC_layer_real(operators, rhs, flag=2, use_dll2=1)
    #     R_2_new = R_2 + dc_layer.CG_iter(max_iter=2)
    #     # DC layer for phi_0
    #     operators = OperatorsMultiEcho(mask, csm, M_0_new, R_2_new, phi_0, f, self.lambda_dll2[2])
    #     B = operators.forward_operator() - kdata
    #     rhs = - operators.jacobian_conj(B, flag=3)
    #     dc_layer = DC_layer_real(operators, rhs, flag=3, use_dll2=1)
    #     phi_0_new = phi_0 + dc_layer.CG_iter(max_iter=2)
    #     # DC layer for f
    #     operators = OperatorsMultiEcho(mask, csm, M_0_new, R_2_new, phi_0_new, f, self.lambda_dll2[3])
    #     B = operators.forward_operator() - kdata
    #     rhs = - operators.jacobian_conj(B, flag=4)
    #     dc_layer = DC_layer_real(operators, rhs, flag=4, use_dll2=1)
    #     f_new = f + dc_layer.CG_iter(max_iter=2)
    #     # concatenate
    #     para = torch.cat((M_0_new, R_2_new, phi_0_new, f_new), dim=1)
    #     paras.append(para)

    #     return para_start, paras, paras

    def forward(self, inputs, iField):
        device = inputs.get_device()
        paras, paras_prior = [], []
        para = torch.cat((inputs[:, 1:2, ...], inputs[:, 3:4, ...]), dim=1)
        M_0 = inputs[:, 0:1, ...]
        phi_0 = inputs[:, 2:3, ...]
        lambda_dll2 = self.lambda_dll2.to(device)
        gd_stepsize = self.gd_stepsize.to(device)

        if self.flag_model == 1:
            for k in range(self.K):
                R_2 = para[:, 0:1, ...]
                f = para[:, 1:2, ...]

                # norm
                R_2_norm = (R_2 - self.norm_means[1]) / self.norm_stds[1]
                f_norm = (f - self.norm_means[3]) / self.norm_stds[3]

                # denorm
                R_2_prior = self.resnet_prior_R_2[k](R_2_norm) * self.norm_stds[1] + self.norm_means[1]
                f_prior = self.resnet_prior_f[k](f_norm) * self.norm_stds[3] + self.norm_means[3]
                para_prior = torch.cat((R_2_prior, f_prior), dim=1)

                # gradient prior
                gradient_prior_R_2 = lambda_dll2[1] * (R_2 - R_2_prior)
                gradient_prior_f = lambda_dll2[3] * (f - f_prior)

                # generate fidelity operator
                operators = OperatorsMultiEcho(M_0, R_2, phi_0, f, num_echos=self.num_echos)

                # calculate fidelity gradient
                gradient_fidelity_R_2 = operators.jacobian_conj(operators.forward_operator() - iField, flag=2)
                gradient_fidelity_f = operators.jacobian_conj(operators.forward_operator() - iField, flag=4)
                
                # total gradient
                gradient_total_R_2 = gradient_prior_R_2  # + gradient_fidelity_R_2
                gradient_total_f = gradient_prior_f  # + gradient_fidelity_f
                
                # gradient descent step
                para[:, 0:1, ...] = para[:, 0:1, ...] - gd_stepsize/(k+1) * gradient_total_R_2
                para[:, 1:2, ...] = para[:, 1:2, ...] - gd_stepsize/(k+1) * gradient_total_f

                paras.append(para)
                paras_prior.append(para_prior)
            return paras, paras_prior

        elif self.flag_model == 0:
            R_2 = para[:, 0:1, ...]
            f = para[:, 1:2, ...]
            # norm and concat
            R_2_norm = (R_2 - self.norm_means[1]) / self.norm_stds[1]
            f_norm = (f - self.norm_means[3]) / self.norm_stds[3]
            inputs_cat = torch.cat((R_2_norm, f_norm), dim=1)
            # forward
            outputs_cat = self.unet(inputs_cat)
            # denorm
            R_2_prior = outputs_cat[:, 0:1, ...] * self.norm_stds[1] + self.norm_means[1]
            f_prior = outputs_cat[:, 1:2, ...] * self.norm_stds[3] + self.norm_means[3]
            para = torch.cat((R_2_prior, f_prior), dim=1)
            return para, para

