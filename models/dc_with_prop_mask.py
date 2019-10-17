import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dc_blocks import *
from models.unet_blocks import *
from models.initialization import *
from models.resnet_with_dc import ResBlock

class DC_with_Prop_Mask(nn.Module):

    def __init__(
        self,
        input_channels,
        filter_channels,
        lambda_dll2, # initializing lambda_dll2
        ncoil=32,
        nrow=256,
        ncol=192,
        K=1,
        unc_map=False,
        slope=0.25,
        slope_threshold=12,
        fixed_mask=False,
        testing=False,
        rescale=False,
        samplingRatio = 0.1 # sparsity level of the sampling mask
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
        self.fixed_mask = fixed_mask
        if self.fixed_mask:
            self.weight_parameters = nn.Parameter((torch.rand(1, nrow, ncol, 1)-0.5)*30, requires_grad=False)
            if testing:
                self.thresh_const = load_mat('/data/Jinwei/T2_slice_recon_GE/2_rolls/Thresh_0.015.mat', 'Thresh')
                self.thresh_const = torch.tensor(self.thresh_const)
            else:
                self.thresh_const = torch.rand(self.weight_parameters.shape)
        else:
            temp = (torch.rand(1, nrow, ncol, 1)-0.5)*30
            temp[:, nrow//2-10 : nrow//2+10, ncol//2-10 : ncol//2+10, :] = 15
            self.weight_parameters = nn.Parameter(temp, requires_grad=True)
            # self.weight_parameters = nn.Parameter(torch.ones(1, nrow, ncol, 1)*30, requires_grad=True)
        self.ncoil = ncoil
        self.nrow = nrow
        self.ncol = ncol
        self.rescale = rescale
        self.samplingRatio = samplingRatio

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

    def rescalePmask(self):
        device = self.Pmask.get_device()
        xbar = torch.mean(self.Pmask)
        r = self.samplingRatio/xbar
        beta = (1-self.samplingRatio) / (1-xbar)
        le = (r<=1).to(device, dtype=torch.float32)
        self.Pmask_recaled = le * self.Pmask * r + (1-le) * (1 - (1-self.Pmask) * beta)

    def ThresholdPmask(self):
        device = self.Pmask.get_device()
        if self.fixed_mask:
            thresh = self.thresh_const.to(device)
            masks = (self.Pmask_recaled > thresh).to(device, dtype=torch.float32)
            masks[:, self.nrow//2-10 : self.nrow//2+10, self.ncol//2-10 : self.ncol//2+10, :] = 1
        else:
            thresh = torch.rand(self.Pmask_recaled.shape).to(device)
            masks = 1/(1+torch.exp(-self.slope_threshold*(self.Pmask_recaled-thresh)))
        self.masks = masks
        return self.masks

    def forward(self, kdata, csms):
        device = kdata.get_device()
        self.lambda_dll2 = self.lambda_dll2.to(device)
        self.weight_parameters = self.weight_parameters.to(device)
        self.Pmask = 1/(1+torch.exp(-self.slope*self.weight_parameters))
        if self.rescale:
            self.rescalePmask()
        else:
            self.Pmask_recaled = self.Pmask
        masks = self.ThresholdPmask()

        # # keep the calibration region
        # masks[:, masks.size()[1]//2-15:masks.size()[1]//2+15,  masks.size()[2]//2-15:masks.size()[2]//2+15, :] = 1
        
        masks = torch.cat((masks, torch.zeros(masks.shape).to(device)),-1)
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
                Unc_maps.append(x_block[:, 2, ...])
        if self.unc_map:
            return Xs, Unc_maps
        else:
            return Xs
