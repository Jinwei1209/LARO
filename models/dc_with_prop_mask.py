import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dc_blocks import *
from models.unet_blocks import *
from models.initialization import *
from models.resnet_with_dc import ResBlock


# class Prob_Mask(nn.Module):

#     def __init__(
#         self,
#         ncoil=1,
#         nrow=256,
#         ncol=184,
#         slope=0.25
#     ):
#         super(Prob_Mask, self).__init__()
#         self.slope = slope
#         self.weight_parameters = nn.Parameter(torch.zeros(ncoil, nrow, ncol, 1), requires_grad=True)

#     def forward(self, x):
#         device = x.get_device()
#         self.weight_parameters = self.weight_parameters.to(device)
#         self.prob_mask = 1/(1+torch.exp(-self.slope*self.weight_parameters))
        # return self.prob_mask


class DC_with_Prop_Mask(nn.Module):

    def __init__(
        self,
        input_channels,
        filter_channels,
        lambda_dll2, # initializing lambda_dll2
        ncoil=1,
        nrow=256,
        ncol=184,
        K=1,
        unc_map=False,
        slope=0.25,
        slope_threshold=12
    ):
        super(DC_with_Prop_Mask, self).__init__()
        self.resnet_block = []
        layers = ResBlock(input_channels, filter_channels, unc_map=unc_map)
        for layer in layers:
            self.resnet_block.append(layer)
        self.resnet_block = nn.Sequential(*self.resnet_block)
        self.resnet_block.apply(init_weights)
        self.K = K
        self.unc_map = unc_map
        self.lambda_dll2 = nn.Parameter(torch.ones(1)*lambda_dll2, requires_grad=True)
        # self.lambda_dll2 = torch.tensor(lambda_dll2)
        self.slope = slope
        self.slope_threshold = slope_threshold
        self.weight_parameters = nn.Parameter((torch.rand(ncoil, nrow, ncol, 1)-0.5)*30, requires_grad=True)
        # self.prob_mask = Prob_Mask()

    def At(self, kdata, mask, csm):
        self.ncoil = csm.shape[1]
        self.nrow = csm.shape[2] 
        self.ncol = csm.shape[3]
        self.npixels = torch.tensor(self.nrow*self.ncol)
        self.npixels = self.npixels.type(torch.DoubleTensor) 
        self.factor = torch.sqrt(self.npixels)
        temp = cplx_mlpy(kdata, mask)
        coilImgs = torch.ifft(temp, 2)*self.factor
        coilComb = torch.sum(
            cplx_mlpy(coilImgs, cplx_conj(csm)),
            dim=1,
            keepdim=False
        )
        coilComb = coilComb.permute(0, 3, 1, 2)
        return coilComb

    def ThresholdPmask(self):
        device = self.Pmask.get_device()
        thresh = torch.rand(self.Pmask.shape).to(device)
        return 1/(1+torch.exp(-self.slope_threshold*(self.Pmask-thresh)))

    def forward(self, kdata, csms):
        device = kdata.get_device()
        self.lambda_dll2 = self.lambda_dll2.to(device)
        self.weight_parameters = self.weight_parameters.to(device)
        self.Pmask = 1/(1+torch.exp(-self.slope*self.weight_parameters))
        masks = self.ThresholdPmask()
        masks = torch.cat((masks, torch.zeros(masks.shape).to(device)),-1)
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
                Unc_maps.append(x_block[:, 2, ...])
        if self.unc_map:
            return Xs, Unc_maps
        else:
            return Xs
