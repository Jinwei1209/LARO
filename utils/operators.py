import torch
import torch.nn as nn
import numpy as np
from utils.data import *


'''
    forward and backward imaging model operator
'''
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
        self.csm = csm
        self.mask = mask
        self.lambda_dll2 = lambda_dll2

        device = self.csm.get_device()
        self.flip = torch.ones([self.nrow, self.ncol, 1]) 
        self.flip = torch.cat((self.flip, torch.zeros(self.flip.shape)), -1).to(device)
        self.flip[::2, ...] = - self.flip[::2, ...] 
        self.flip[:, ::2, ...] = - self.flip[:, ::2, ...]

    def AtA(
        self, 
        img, 
        use_dll2=1  # 1 for l2-x0 reg, 2 for l2-TV reg, 3 for l1-TV reg
    ):
        # forward
        img_new = img.permute(0, 2, 3, 1)
        img_new = img_new[:, None, ...]  # multiply order matters (in torch implementation)
        coilImages = cplx_mlpy(self.csm, img_new)
        coilImages = cplx_mlpy(coilImages, self.flip) # for GE kdata
        coilImages = fft_shift_row(coilImages, self.nrow) # for GE kdata
        kspace = torch.fft(coilImages, 2)  
        temp = cplx_mlpy(kspace, self.mask)
        # inverse
        coilImgs = torch.ifft(temp, 2)
        coilImgs = fft_shift_row(coilImgs, self.nrow) # for GE kdata
        coilImgs = cplx_mlpy(coilImgs, self.flip) # for GE kdata
        coilComb = torch.sum(
            cplx_mlpy(coilImgs, cplx_conj(self.csm)),
            dim=1,
            keepdim=False
        )
        coilComb = coilComb.permute(0, 3, 1, 2)
        if use_dll2 == 1:
            coilComb = coilComb + self.lambda_dll2*img
        elif use_dll2 == 2:
            coilComb = coilComb + self.lambda_dll2*divergence(gradient(img))
        elif use_dll2 == 3:
            coilComb = coilComb + self.lambda_dll2*divergence(gradient(img)/torch.sqrt(gradient(img)**2+3e-5))  #1e-4 best, 5e-5 to have consistent result to ADMM
            # print(torch.mean(gradient(img)**2))
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


"""
    for 4d data: (batchsize, real/imag dim, row dim, col dim)
"""
def gradient(x):
    dx = torch.cat((x[:, :, :, 1:], x[:, :, :, -1:]), dim=3) - x
    dy = torch.cat((x[:, :, 1:, :], x[:, :, -1:, :]), dim=2) - x
    return torch.cat((dx[..., None], dy[..., None]), dim=-1)


"""
    for 5d data: (batchsize, real/imag dim, row dim, col dim, gradient dim)
"""
def divergence(d):
    device = d.get_device()
    dx = d[..., 0]
    dy = d[..., 1]
    zerox = torch.zeros(dx.size()[:3] + (1,)).to(device)
    zeroy = torch.zeros(dy.size()[:2] + (1,) + dy.size()[-1:]).to(device)
    dxx = torch.cat((dx[:, :, :, :-1], zerox), dim=3) - torch.cat((zerox, dx[:, :, :, :-1]), dim=3) 
    dyy = torch.cat((dy[:, :, :-1, :], zeroy), dim=2) - torch.cat((zeroy, dy[:, :, :-1, :]), dim=2)
    return  - dxx - dyy

"""
    backward operator for CardiacQSM rawdata recon
"""
def backward_CardiacQSM(kdata, csm, mask, flip):
    nrows = kdata.size()[2]
    ncols = kdata.size()[3]
    temp = cplx_mlpy(kdata, mask)
    temp = torch.ifft(temp, 2)
    temp = fft_shift_row(temp, nrows)
    temp = fft_shift_col(temp, ncols)
    coilComb = torch.sum(
        cplx_mlpy(temp, cplx_conj(csm)),
        dim=1,
        keepdim=False
    )
    coilComb = cplx_mlpy(coilComb, flip)
    coilComb = coilComb.permute(0, 3, 1, 2)
    return coilComb

"""
    forward operator for CardiacQSM rawdata recon
"""
def forward_CardiacQSM(image, csm, mask, flip):
    image = image.permute(0, 2, 3, 1)
    nrows = csm.size()[2]
    ncols = csm.size()[3]
    temp = cplx_mlpy(image, flip)
    temp = temp[:, None, ...]
    temp = cplx_mlpy(csm, temp)
    temp = fft_shift_row(temp, nrows)
    temp = fft_shift_col(temp, ncols)
    temp = torch.fft(temp, 2)
    return cplx_mlpy(temp, mask)

"""
    AtA operator for CardiacQSM data
"""
class backward_forward_CardiacQSM():
    
    def __init__(
        self,
        csm,
        mask,
        lambda_dll2,
    ):
        self.ncoil = csm.shape[1]
        self.nrow = csm.shape[2] 
        self.ncol = csm.shape[3] 
        self.csm = csm
        self.mask = mask
        self.lambda_dll2 = lambda_dll2

        device = self.csm.get_device()
        self.flip = torch.ones([self.nrow, self.ncol, 1]) 
        self.flip = torch.cat((self.flip, torch.zeros(self.flip.shape)), -1).to(device)
        self.flip[::2, ...] = - self.flip[::2, ...] 
        self.flip[:, ::2, ...] = - self.flip[:, ::2, ...]

    def AtA(
        self, 
        img, 
        use_dll2=1
    ):
        # forward
        image = img.permute(0, 2, 3, 1)
        temp = cplx_mlpy(image, self.flip)
        temp = temp[:, None, ...]
        temp = cplx_mlpy(self.csm, temp)
        temp = fft_shift_row(temp, self.nrow)
        temp = fft_shift_col(temp, self.ncol)
        temp = torch.fft(temp, 2)
        temp = cplx_mlpy(temp, self.mask)
        # inverse
        temp = torch.ifft(temp, 2)
        temp = fft_shift_row(temp, self.nrow)
        temp = fft_shift_col(temp, self.ncol)
        coilComb = torch.sum(
            cplx_mlpy(temp, cplx_conj(self.csm)),
            dim=1,
            keepdim=False
        )
        coilComb = cplx_mlpy(coilComb, self.flip)
        coilComb = coilComb.permute(0, 3, 1, 2)
        if use_dll2 == 1:
            coilComb = coilComb + self.lambda_dll2*img
        return coilComb

"""
    forward and Jacobian operators of multi-echo gradient echo data
"""
class OperatorsMultiEcho():
    
    def __init__(
        self,
        mask,
        csm,
        M_0,
        R_2,
        phi_0,
        f
    ):
        self.device = csm.get_device()
        self.num_samples = csm.shape[0]
        self.num_coils = csm.shape[1]
        self.num_echos = csm.shape[2]
        self.num_rows = csm.shape[3] 
        self.num_cols = csm.shape[4] 
        self.csm = csm
        self.mask = mask
        self.M_0 = M_0.repeat(1, self.num_echos, 1, 1)
        self.R_2 = R_2.repeat(1, self.num_echos, 1, 1)
        self.phi_0 = phi_0.repeat(1, self.num_echos, 1, 1)
        self.f = f.repeat(1, self.num_echos, 1, 1)
        
        # time slots for multi-echo data
        self.time_intervals = torch.arange(0, self.num_echos)[None, :, None, None].float()
        self.time_intervals = self.time_intervals.repeat(self.num_samples, 
                              1, self.num_rows, self.num_cols).to(self.device)

    def forward_operator(self, flag=0):
        # flag = 0: no additional operation
        # flag = 1: M_0, W/M0
        # flag = 2: R_2, W*(-T)
        # flag = 3: phi_0, W*(i)
        # flag = 4: f, W*(iT)
        self.tj_coils = self.time_intervals[:, None, ..., None]
        self.tj_coils = torch.cat((self.tj_coils, torch.zeros(self.tj_coils.shape).to(
                                   self.device)), dim=-1).repeat(1, self.num_coils, 1, 1, 1, 1)

        img0 = (torch.exp(-self.R_2 * self.time_intervals))[..., None]
        img0 = torch.cat((img0, torch.zeros(img0.shape).to(self.device)), dim=-1)
        self.img0 = img0[:, None, ...].repeat(1, self.num_coils, 1, 1, 1, 1)

        img1 = (self.M_0 * torch.exp(-self.R_2 * self.time_intervals))[..., None]
        img1 = torch.cat((img1, torch.zeros(img1.shape).to(self.device)), dim=-1)
        self.img1 = img1[:, None, ...].repeat(1, self.num_coils, 1, 1, 1, 1)

        img2_real = torch.cos(self.phi_0 + self.f * self.time_intervals)
        img2_imag = torch.sin(self.phi_0 + self.f * self.time_intervals)  
        img2 = torch.cat((img2_real[..., None], img2_imag[..., None]), dim=-1)
        self.img2 = img2[:, None, ...].repeat(1, self.num_coils, 1, 1, 1, 1)

        if flag == 0:
            img = cplx_mlpy(img1, img2)[:, None, ...]
            img = img.repeat(1, self.num_coils, 1, 1, 1, 1)
        elif flag == 1:
            img = cplx_mlpy(img0, img2)[:, None, ...]
            img = img.repeat(1, self.num_coils, 1, 1, 1, 1)
        elif flag == 2:
            img = cplx_mlpy(img1, img2)[:, None, ...]
            img = img.repeat(1, self.num_coils, 1, 1, 1, 1)
            img = cplx_mlpy(img, -self.tj_coils)
        elif flag == 3:
            img = cplx_mlpy(img1, img2)[:, None, ...]
            img = img.repeat(1, self.num_coils, 1, 1, 1, 1)
            img = torch.cat((-img[..., 1:2], img[..., 0:1]), dim=-1)
        elif flag == 4:
            img = cplx_mlpy(img1, img2)[:, None, ...]
            img = img.repeat(1, self.num_coils, 1, 1, 1, 1)
            img = torch.cat((-img[..., 1:2], img[..., 0:1]), dim=-1)
            img = cplx_mlpy(img, self.tj_coils)

        img_coils = cplx_mlpy(self.csm, img)
        kdata_coils = torch.fft(img_coils, signal_ndim=2)
        kdata_coils_under = cplx_mlpy(self.mask, kdata_coils)
        return kdata_coils_under

    def jacobian_conj(self, kdata):
        kdata_under = cplx_mlpy(self.mask, kdata)
        img_under = torch.ifft(kdata_under, signal_ndim=2)
        img_coils = cplx_mlpy(cplx_conj(self.csm), img_under)

        # for M_0, torch.Size([batchsize, ncoils, nechos, nrows, ncols, 2])
        J1 = cplx_mlpy(self.img0, img_coils)
        J1 = cplx_mlpy(cplx_conj(self.img2), J1)
        # for R_2, the same dim
        J2 = cplx_mlpy(self.img1, img_coils)
        J2 = cplx_mlpy(cplx_conj(self.img2), J2)
        J2 = cplx_mlpy(-self.tj_coils, J2)
        # for phi_0
        J3 = cplx_mlpy(self.img1, img_coils)
        J3 = cplx_mlpy(cplx_conj(self.img2), J3)
        J3 = torch.cat((J3[..., 1:2], -J3[..., 0:1]), dim=-1)
        # for f
        J4 = cplx_mlpy(self.img1, img_coils)
        J4 = cplx_mlpy(cplx_conj(self.img2), J4)
        J4 = cplx_mlpy(self.tj_coils, J4)
        J4 = torch.cat((J4[..., 1:2], -J4[..., 0:1]), dim=-1)

        J1 = torch.sum(J1, dim=(1,2), keepdim=False)[:, None, ...]
        J2 = torch.sum(J2, dim=(1,2), keepdim=False)[:, None, ...]
        J3 = torch.sum(J3, dim=(1,2), keepdim=False)[:, None, ...]
        J4 = torch.sum(J4, dim=(1,2), keepdim=False)[:, None, ...]
        J = torch.cat((J1, J2, J3, J4), dim=1)[..., 0]
        J[J>10] = 0
        print('Max value in the Jacobian matrix is = {0}'.format(torch.max(J)))
        return J








