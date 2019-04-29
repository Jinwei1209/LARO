import torch
import torch.nn as nn
import numpy as np
from utils.data import *


class Back_forward():

    def __init__(
        self,
        csm,
        mask,
        lambda_dll2,
    ):

        self.ncoil = csm.shape[1]
        self.nrow = csm.shape[2] 
        self.ncol = csm.shape[3]
        self.npixels = torch.tensor(self.nrow*self.ncol)
        self.npixels = self.npixels.type(torch.DoubleTensor) 
        self.csm = csm
        self.mask = mask
        self.factor = torch.sqrt(self.npixels)
        self.lambda_dll2 = lambda_dll2

    def AtA(self, img, use_dll2=False):

        img = img.permute(0, 2, 3, 1)
        img = img[:, None, ...]
        coilImages = cplx_mlpy(self.csm, img)
        kspace = torch.fft(coilImages, 2)/self.factor
        temp = cplx_mlpy(kspace, self.mask)
        coilImgs = torch.ifft(temp, 2)*self.factor
        coilComb = torch.sum(
            cplx_mlpy(coilImgs, cplx_conj(self.csm)),
            dim=1,
            keepdim=False
        )
        if use_dll2:
            coilComb = coilComb[:, None, ...]
            coilComb = coilComb + self.lambda_dll2*img
        else:
            coilComb = coilComb.permute(0, 3, 1, 2)

        return coilComb


def forward_operator(img, csm, mask, ncoil, nrow, ncol):
    
    coilImages = np.tile(img, [ncoil, 1, 1]) * csm
    kspace = np.fft.fft2(coilImages) / np.sqrt(nrow*ncol)
    res = kspace[mask!=0]

    return res


def backward_operator(kspaceUnder, csm, mask, ncoil, nrow, ncol):

    # axis 0 as the channel dim
    temp = np.zeros((ncoil, nrow, ncol), dtype=np.complex64)
    temp[mask!=0] = kspaceUnder
    img = np.fft.ifft2(temp) * np.sqrt(nrow*ncol)
    coilComb = np.sum(img*np.conj(csm), axis=0).astype(np.complex64)

    return coilComb


def lossL1():
    return nn.L1Loss()


def loss_classificaiton():
    return nn.BCELoss()


def lossL2():
    return nn.MSELoss()

    

