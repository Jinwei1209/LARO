import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math


class passThroughSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        device = x.get_device()
        return (1 / (1 + torch.exp(-x))).to(device)
    @staticmethod
    def backward(ctx, g):
        return g


class binaryRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        device = x.get_device()
        return x.round().to(device)
    @staticmethod
    def backward(ctx, g):
        return g


class bernoulliSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        device = x.get_device()
        samples = torch.rand(x.shape).to(device)
        return (torch.ceil(x - samples)).to(device)
    @staticmethod
    def backward(ctx, g):
        return g


# class PmaskBlock():

#     def __init__(
#         self,
#         nrow,
#         ncol,
#         flag_ND=0,  # 0 for 1D Cartesian along row direciton, 1 for 1D Cartesian along column direction, 
#                     # 2 for 2D Cartesian, 3 for variable density random
#         slope=0.25,
#         passSigmoid=False,
#         stochasticSampling=True,
#         rescale=True,
#         samplingRatio=0.1, # sparsity level of the sampling mask
#         weight_parameters=0
#     ):
#         self.nrow = nrow
#         self.ncol = ncol
#         self.flag_ND = flag_ND
#         self.slope = slope
#         self.passSigmoid = passSigmoid
#         self.stochasticSampling = stochasticSampling
#         self.rescale = rescale
#         self.samplingRatio = samplingRatio
#         if flag_ND == 0:
#             temp = (torch.rand(nrow)-0.5)*30
#             temp[nrow//2-13 : nrow//2+12] = 15
#         elif flag_ND == 1:
#             temp = (torch.rand(ncol)-0.5)*30
#             temp[ncol//2-13 : ncol//2+12] = 15
#         elif flag_ND == 2:
#             temp1 = (torch.rand(nrow)-0.5)*30
#             temp2 = (torch.rand(ncol)-0.5)*30
#             temp1[nrow//2-13 : nrow//2+12] = 15
#             temp2[ncol//2-13 : ncol//2+12] = 15
#         elif flag_ND == 3:
#             temp = (torch.rand(nrow, ncol)-0.5)*30
#             temp[nrow//2-13 : nrow//2+12, ncol//2-13 : ncol//2+12] = 15
#         # if flag_ND != 2:
#         #     self.weight_parameters = nn.Parameter(temp, requires_grad=True).cuda()
#         # else:
#         #     self.weight_parameters1 = nn.Parameter(temp1, requires_grad=True).cuda()
#         #     self.weight_parameters2 = nn.Parameter(temp2, requires_grad=True).cuda()
#         self.weight_parameters = weight_parameters

#     def rescalePmask(self, Pmask, samplingRatio):
#         device = Pmask.get_device()
#         xbar = torch.mean(Pmask)
#         r = samplingRatio/xbar
#         beta = (1-samplingRatio) / (1-xbar)
#         le = (r<=1).to(device, dtype=torch.float32)
#         return le * Pmask * r + (1-le) * (1 - (1-Pmask) * beta)

#     def samplingPmask(self, Pmask_rescaled, flag_ND):
#         if flag_ND == 0:
#             Mask1D = bernoulliSample.apply(Pmask_rescaled)
#             Mask = Mask1D.repeat(self.ncol, 1).transpose(0,1)
#         elif flag_ND == 1:
#             Mask1D = bernoulliSample.apply(Pmask_rescaled)
#             Mask = Mask1D.repeat(self.nrow, 1)
#         elif flag_ND == 3:
#             Mask = bernoulliSample.apply(Pmask_rescaled)
#         return Mask

#     def generateMask(self):

#         if self.flag_ND != 2:
#             if self.passSigmoid:
#                 self.Pmask = passThroughSigmoid.apply(self.slope * self.weight_parameters)
#             else:
#                 self.Pmask = 1 / (1 + torch.exp(-self.slope * self.weight_parameters))
#             if self.rescale:
#                 self.Pmask_rescaled = self.rescalePmask(self.Pmask, self.samplingRatio)
#             else:
#                 self.Pmask_rescaled = self.Pmask
#             self.Mask = self.samplingPmask(self.Pmask_rescaled, self.flag_ND)
#             return self.Mask

#         elif self.flag_ND == 2:
#             if self.passSigmoid:
#                 self.Pmask1 = passThroughSigmoid.apply(self.slope * self.weight_parameters1)
#                 self.Pmask2 = passThroughSigmoid.apply(self.slope * self.weight_parameters2)
#             else:
#                 self.Pmask1 = 1 / (1 + torch.exp(-self.slope * self.weight_parameters1))
#                 self.Pmask2 = 1 / (1 + torch.exp(-self.slope * self.weight_parameters2))
#             self.Pmask = self.Pmask1.repeat(self.ncol, 1).transpose(0,1) + self.Pmask2.repeat(self.nrow, 1)
#             if self.rescale:
#                 self.Pmask_rescaled1 = self.rescalePmask(self.Pmask1, np.sqrt(self.samplingRatio))
#                 self.Pmask_rescaled2 = self.rescalePmask(self.Pmask2, np.sqrt(self.samplingRatio))
#             else:
#                 self.Pmask_rescaled1 = self.Pmask1
#                 self.Pmask_rescaled2 = self.Pmask2
#             Mask1 = self.samplingPmask(self.Pmask_rescaled1, flag_ND=0)
#             Mask2 = self.samplingPmask(self.Pmask_rescaled2, flag_ND=1)
#             self.Mask = Mask1 * Mask2
#             return self.Mask
            

