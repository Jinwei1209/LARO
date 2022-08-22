from numpy.matrixlib.defmatrix import matrix
import torch
import torch.nn as nn
import numpy as np

from utils.data import *


class Back_forward():
    '''
        forward and backward imaging model operator
    '''
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


def gradient(x):
    """
        for 4d data: (batchsize, real/imag dim, row dim, col dim)
    """
    dx = torch.cat((x[:, :, :, 1:], x[:, :, :, -1:]), dim=3) - x
    dy = torch.cat((x[:, :, 1:, :], x[:, :, -1:, :]), dim=2) - x
    return torch.cat((dx[..., None], dy[..., None]), dim=-1)


def divergence(d):
    """
        for 5d data: (batchsize, real/imag dim, row dim, col dim, gradient dim)
    """
    # device = d.get_device()
    # dx = d[..., 0]
    # dy = d[..., 1]
    # zerox = torch.zeros(dx.size()[:3] + (1,)).to(device)
    # zeroy = torch.zeros(dy.size()[:2] + (1,) + dy.size()[-1:]).to(device)
    # dxx = torch.cat((dx[:, :, :, :-1], zerox), dim=3) - torch.cat((zerox, dx[:, :, :, :-1]), dim=3) 
    # dyy = torch.cat((dy[:, :, :-1, :], zeroy), dim=2) - torch.cat((zeroy, dy[:, :, :-1, :]), dim=2)

    dx = d[..., 0]
    dy = d[..., 1]
    dxx = dx - torch.cat((dx[:, :, :, :1], dx[:, :, :, :-1]), dim=3) 
    dyy = dy - torch.cat((dy[:, :, :1, :], dy[:, :, :-1, :]), dim=2)
    return  - dxx - dyy

def backward_CardiacQSM(kdata, csm, mask, flip):
    """
        backward operator for CardiacQSM rawdata recon
    """
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

def forward_CardiacQSM(image, csm, mask, flip):
    """
        forward operator for CardiacQSM rawdata recon
    """
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


class backward_forward_CardiacQSM():
    """
        AtA operator for CardiacQSM data
    """
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

class Back_forward_multiEcho():
    '''
        forward and backward imaging model operator for multi echo GRE data (scanner: 0 (GE) / 1 (Siemens))
        (echo dim as in the channel dim in CNN model)
    '''
    def __init__(
        self,
        csm,
        mask,
        flip,
        lambda_dll2,
        lambda_lowrank = 0,
        echo_cat = 1, # flag to concatenate echo dimension into channel
        necho = 10,
        kdata = None,
        csm_lowres = None,
        U = None, 
        rank = 0,
        scanner = 0,
    ):
        self.nrows = csm.size()[3]
        self.ncols = csm.size()[4]
        self.nechos = csm.size()[2]
        self.csm = csm
        self.mask = mask
        self.lambda_dll2 = lambda_dll2
        self.lambda_lowrank = lambda_lowrank
        self.flip = flip
        self.echo_cat = echo_cat
        self.necho = necho
        self.kdata = kdata
        self.csm_lowres = csm_lowres
        self.rank = rank
        self.scanner = scanner
        if self.rank > 0:
            self.Ur = U[:, :self.rank, :]  # (echo, rank, 2)

        # device = self.csm.get_device()   
        # self.flip = torch.ones([self.nechos, self.nrows, self.ncols, 1]) 
        # self.flip = torch.cat((self.flip, torch.zeros(self.flip.shape)), -1).to(device)
        # self.flip[:, ::2, ...] = - self.flip[:, ::2, ...] 
        # self.flip[:, :, ::2, ...] = - self.flip[:, :, ::2, ...]

    def AtA(
        self, 
        img, 
        use_dll2=1  # 1 for l2-x0 reg, 2 for l2-TV reg, 3 for l1-TV reg
    ):
        # forward
        if self.echo_cat:
            image = torch_channel_deconcate(img)  # (batch, 2, echo, row, col)
        else:
            image = img
        if self.scanner == 1:
            image = torch.flip(image, dims=[4])  # for Siemens data
            image = fft_shift_col(image, self.ncols, 1)  # for Siemens data
            image[:, 1, ...] = - image[:, 1, ...]  # for Siemens data
        image = image.permute(0, 2, 3, 4, 1) # (batch, echo, row, col, 2)
        temp = cplx_mlpy(image, self.flip) # for GE kdata
        temp = temp[:, None, ...] # multiply order matters (in torch implementation)
        temp = cplx_mlpy(self.csm, temp) # (batch, coil, echo, row, col, 2)
        temp = fft_shift_row(temp, self.nrows, 1) # for GE kdata
        temp = torch.fft(temp, 2) 
        temp = cplx_mlpy(temp, self.mask)
        # inverse
        coilImgs = torch.ifft(temp, 2)
        coilImgs = fft_shift_row(coilImgs, self.nrows, 1) # for GE kdata
        coilComb = torch.sum(
            cplx_mlpy(coilImgs, cplx_conj(self.csm)),
            dim=1,
            keepdim=False
        )
        coilComb = cplx_mlpy(coilComb, self.flip) # for GE kdata
        coilComb = coilComb.permute(0, 4, 1, 2, 3) # (batch, 2, echo, row, col)
        if self.scanner == 1:
            coilComb = fft_shift_col(coilComb, self.ncols, 1)  # for Siemens data
            coilComb = torch.flip(coilComb, dims=[4])  # for Siemens data
            coilComb[:, 1, ...] = - coilComb[:, 1, ...]  # for Siemens data
        if self.echo_cat:
            coilComb = torch_channel_concate(coilComb, self.necho) # (batch, 2*echo, row, col)
        if use_dll2 == 1:
            if self.rank == 0:
                if self.lambda_dll2.size()[0] == 1:
                    coilComb = coilComb + self.lambda_dll2 * img
                elif self.lambda_dll2.size()[0] == 2:
                    coilComb = coilComb + torch.cat((self.lambda_dll2[0] * img[:, :-2, ...], self.lambda_dll2[1] * img[:, -2:, ...]), 1)
                elif self.lambda_dll2.size()[0] == 3:
                    coilComb = coilComb + torch.cat((self.lambda_dll2[0] * img[:, :-4, ...], self.lambda_dll2[1] * img[:, -4:-2, ...], 
                                                     self.lambda_dll2[2] * img[:, -2:, ...]), 1)
            elif self.rank > 0:
                coilComb = coilComb + self.lambda_dll2 * img

                # for low rank regularization
                if self.echo_cat:
                    image = torch_channel_deconcate(img)  # (batch, 2, echo, row, col)
                image = image.permute(0, 3, 4, 2, 1)  # (batch, row, col, echo, 2)
                image = image.view(-1, self.nechos, 2).permute(1, 0, 2)  # (batch*row*col, echo, 2) => (echo, batch*row*col, 2)
                UrHx = cplx_matmlpy(cplx_matconj(self.Ur), image)  # (rank, echo, 2) * (echo, batch*row*col, 2) => (rank, batch*row*col, 2)
                image = cplx_matmlpy(self.Ur, UrHx)   # (echo, rank, 2) * (rank, batch*row*col, 2) => (echo, batch*row*col, 2)
                image = image.permute(1, 0, 2)  # (echo, batch*row*col, 2) => (batch*row*col, echo, 2)
                image = image.view(-1, self.nrows, self.ncols, self.nechos, 2)  # (batch, row, col, echo, 2)
                low_rank_approx = image.permute(0, 4, 3, 1, 2)  # (batch, 2, echo, row, col)
                if self.echo_cat:
                    low_rank_approx = torch_channel_concate(low_rank_approx, self.necho) # (batch, 2*echo, row, col)
                low_rank_reg = img - low_rank_approx
                # combine together
                coilComb += self.lambda_lowrank * low_rank_reg

        elif use_dll2 == 2:
            coilComb = coilComb + self.lambda_dll2 * divergence(gradient(img))
        elif use_dll2 == 3:
            coilComb = coilComb + self.lambda_dll2 * divergence(gradient(img) / torch.sqrt(gradient(img)**2+5e-4))  #1e-4 best, 5e-5 to have consistent result to ADMM
        return coilComb

    def low_rank_approx(self, img):
        img = img.permute(0, 2, 3, 1)  # (batch, row, col, 2*echo)
        s0 = img.size()
        M = s0[-1]  # M different contrast weighted images
        img = img.view(-1, M).permute(1, 0) # (M * N) with N voxels (samples) and M echos (features) 

        # row rank approximation of full resolution img
        tmp = torch.matmul(self.V.permute(1, 0), img)  # (k * M) * (M * N) => (k * N)
        tmp = torch.matmul(self.V, tmp)  # (M * k) * (k * N) => (M * N)
        tmp = tmp.permute(1, 0).view(s0[0], s0[1], s0[2], s0[3])  # (N * M) => (batch, row, col, 2*echo)
        img_low_rank = tmp.permute(0, 3, 1, 2)  # (batch, 2*echo, row, col)
        return img_low_rank

def compute_V(kdata, csm, k=10):
    '''
        Compute PCA from 25*25 central kspace "training data" and apply low rank approximation of the whole image
        kdata: auto-calibrated undersampled data (batch, coil, echo, row, col, 2)
        csm: low resolution sensitivity maps (batch, coil, echo, row, col, 2)
        k: rank (number of principle directions)
    '''
    nrow, ncol = kdata.size()[3], kdata.size()[4]
    M = 2 * kdata.size()[2]
    kdata = kdata[:, :, :, nrow//2-13:nrow//2+12, ncol//2-13:ncol//2+12, :]  # central fully sampled kspace
    ncoil, necho, nrow, ncol = kdata.size()[1], kdata.size()[2], kdata.size()[3], kdata.size()[4]
    # flip matrix
    flip = torch.ones([necho, nrow, ncol, 1]) 
    flip = torch.cat((flip, torch.zeros(flip.shape)), -1).to('cuda')
    flip[:, ::2, ...] = - flip[:, ::2, ...] 
    flip[:, :, ::2, ...] = - flip[:, :, ::2, ...]
    flip = flip[None, ...] # (1, necho, nrow, ncol, 2)
    # sampling mask (all ones)
    mask = torch.ones([ncoil, necho, nrow, ncol, 2]).to('cuda')
    mask[..., 1] = 0
    mask = mask[None, ...] # (1, ncoil, necho, nrow, ncol, 2)

    # generate low res fully sampled image
    low_res_img = backward_multiEcho(kdata, csm, mask, flip).permute(0, 2, 3, 1)  # (batch, row, col, 2*echo)
    low_res_img = low_res_img.view(-1, M)  # (N * M) with N voxels (samples) and M echos (features) 
    (_, _, V) = torch.pca_lowrank(low_res_img.cpu().detach(), q=k)  # V: (M * k) matrix
    V = V.to('cuda')
    return V

def low_rank_approx(img, kdata, csm, k=10):
    '''
        Compute PCA from 25*25 central kspace "training data" and apply low rank approximation of the whole image
        img: full resolution multi-echo image to do low rank approximation (batch, 2*echo, row, col)
        kdata: auto-calibrated undersampled data (batch, coil, echo, row, col, 2)
        csm: low resolution sensitivity maps (batch, coil, echo, row, col, 2)
        k: rank (number of principle directions)
    '''
    img = img.permute(0, 2, 3, 1)  # (batch, row, col, 2*echo)
    s0 = img.size()
    M = s0[-1]  # M different contrast weighted images
    img = img.view(-1, M).permute(1, 0) # (M * N) with N voxels (samples) and M echos (features) 

    V = compute_V(kdata, csm, k=k)

    # row rank approximation of full resolution img
    tmp = torch.matmul(V.permute(1, 0), img)  # (k * M) * (M * N) => (k * N)
    tmp = torch.matmul(V, tmp)  # (M * k) * (k * N) => (M * N)
    tmp = tmp.permute(1, 0).view(s0[0], s0[1], s0[2], s0[3])  # (N * M) => (batch, row, col, 2*echo)
    img_low_rank = tmp.permute(0, 3, 1, 2)  # (batch, 2*echo, row, col)
    return img_low_rank

class Back_forward_multiEcho_compressor():
    '''
        forward and backward imaging model operator for multi echo GRE data with a temporal compressor
        (echo dim as in the channel dim in CNN model)
    '''
    def __init__(
        self,
        csm,
        mask,
        flip,
        lambda_dll2,
        echo_cat = 1, # flag to concatenate echo dimension into channel
        necho = 10,
        kdata = None,
        U = None, # temporal compressor
        rank = 10,  # number of temporal basis to use
        flag_compressor = 0  # flag = 0: MFSUrUr*(x) (x is the original images); 
                             # flag = 1: MFSUr(x) (x is the compressed images);
                             # flag = 2: MUrFS(x) (x is the compressed images).
    ):
        self.nrows = csm.size()[3]
        self.ncols = csm.size()[4]
        self.nechos = csm.size()[2]
        self.csm = csm
        self.mask = mask
        self.lambda_dll2 = lambda_dll2
        self.flip = flip
        self.echo_cat = echo_cat
        self.necho = necho
        self.kdata = kdata
        self.rank = rank
        self.Ur = U[:, :self.rank, :]  # (echo, rank, 2)
        self.flag_compressor = flag_compressor

    def AtA(
        self, 
        img, 
        use_dll2=1  # 1 for l2-x0 reg, 2 for l2-TV reg, 3 for l1-TV reg
    ):
        # forward
        if self.echo_cat:
            image = torch_channel_deconcate(img)  # (batch, 2, echo, row, col)
        else:
            image = img
        if self.flag_compressor == 0:
            # for low rank projection
            image = image.permute(0, 3, 4, 2, 1)  # (batch, row, col, echo, 2)
            image = image.view(-1, self.nechos, 2).permute(1, 0, 2)  # (batch*row*col, echo, 2) => (echo, batch*row*col, 2)
            UrHx = cplx_matmlpy(cplx_matconj(self.Ur), image)  # (rank, echo, 2) * (echo, batch*row*col, 2) => (rank, batch*row*col, 2)
            image = cplx_matmlpy(self.Ur, UrHx)   # (echo, rank, 2) * (rank, batch*row*col, 2) => (echo, batch*row*col, 2)
            image = image.permute(1, 0, 2)  # (echo, batch*row*col, 2) => (batch*row*col, echo, 2)
            image = image.view(-1, self.nrows, self.ncols, self.nechos, 2)  # (batch, row, col, echo, 2)
            
            # forward model
            image = image.permute(0, 3, 1, 2, 4) # (batch, echo, row, col, 2)
            temp = cplx_mlpy(image, self.flip) # for GE kdata
            temp = temp[:, None, ...] # multiply order matters (in torch implementation)
            temp = cplx_mlpy(self.csm, temp) # (batch, coil, echo, row, col, 2)
            temp = fft_shift_row(temp, self.nrows, 1) # for GE kdata
            temp = torch.fft(temp, 2) 
            temp = cplx_mlpy(temp, self.mask)

            # inverse
            coilImgs = torch.ifft(temp, 2)
            coilImgs = fft_shift_row(coilImgs, self.nrows, 1) # for GE kdata
            coilComb = torch.sum(
                cplx_mlpy(coilImgs, cplx_conj(self.csm)),
                dim=1,
                keepdim=False
            )
            coilComb = cplx_mlpy(coilComb, self.flip) # for GE kdata
            
            # for low rank projection 
            image = coilComb.permute(0, 2, 3, 1, 4)  # (batch, row, col, echo, 2)
            image = image.view(-1, self.nechos, 2).permute(1, 0, 2)  # (batch*row*col, echo, 2) => (echo, batch*row*col, 2)
            UrHx = cplx_matmlpy(cplx_matconj(self.Ur), image)  # (rank, echo, 2) * (echo, batch*row*col, 2) => (rank, batch*row*col, 2)
            image = cplx_matmlpy(self.Ur, UrHx)   # (echo, rank, 2) * (rank, batch*row*col, 2) => (echo, batch*row*col, 2)
            image = image.permute(1, 0, 2)  # (echo, batch*row*col, 2) => (batch*row*col, echo, 2)
            image = image.view(-1, self.nrows, self.ncols, self.nechos, 2)  # (batch, row, col, echo, 2)
            
            coilComb = image.permute(0, 4, 3, 1, 2) # (batch, 2, echo, row, col)
            if self.echo_cat:
                coilComb = torch_channel_concate(coilComb, self.necho) # (batch, 2*echo, row, col)
            if use_dll2 == 1:
                coilComb = coilComb + self.lambda_dll2 * img
            return coilComb

        elif self.flag_compressor == 1:
            return 0
        
        elif self.flag_compressor == 2:
            return 0

def backward_multiEcho(kdata, csm, mask, flip, echo_cat=1, necho=10, scanner=0):
    """
    backward operator for multi-echo GRE data
    scanner: 0 (GE) / 1 (Siemens)
    """
    nrows = kdata.size()[3]
    ncols = kdata.size()[4]
    nechos = kdata.size()[2]
    temp = cplx_mlpy(kdata, mask)
    temp = torch.ifft(temp, 2)
    temp = fft_shift_row(temp, nrows, 1)
    coilComb = torch.sum(
        cplx_mlpy(temp, cplx_conj(csm)),
        dim=1,
        keepdim=False
    )
    coilComb = cplx_mlpy(coilComb, flip)
    coilComb = coilComb.permute(0, 4, 1, 2, 3) # (batch, 2, echo, row, col)
    if scanner == 1:
        coilComb = fft_shift_col(coilComb, ncols, 1)  # for Siemens data
        coilComb = torch.flip(coilComb, dims=[4])  # for Siemens data
        coilComb[:, 1, ...] = - coilComb[:, 1, ...]  # for Siemens data
    if echo_cat:
        coilComb = torch_channel_concate(coilComb, necho) # (batch, 2*echo, row, col)
    return coilComb

def backward_multiEcho_compressor(kdata, csm, mask, flip, U, rank,
                                  flag_compressor=0, echo_cat=1, necho=10, scanner=0):
    """
    backward operator for multi-echo GRE data with a temporal compressor
    scanner: 0 (GE) / 1 (Siemens)
    """
    nrows = kdata.size()[3]
    ncols = kdata.size()[4]
    nechos = kdata.size()[2]
    Ur = U[:, :rank, :]
    if flag_compressor == 0:
        temp = cplx_mlpy(kdata, mask)
        temp = torch.ifft(temp, 2)
        temp = fft_shift_row(temp, nrows, 1)
        coilComb = torch.sum(
            cplx_mlpy(temp, cplx_conj(csm)),
            dim=1,
            keepdim=False
        )
        coilComb = cplx_mlpy(coilComb, flip)

        # for low rank projection
        image = coilComb.permute(0, 2, 3, 1, 4)  # (batch, row, col, echo, 2)
        image = image.view(-1, nechos, 2).permute(1, 0, 2)  # (batch*row*col, echo, 2) => (echo, batch*row*col, 2)
        UrHx = cplx_matmlpy(cplx_matconj(Ur), image)  # (rank, echo, 2) * (echo, batch*row*col, 2) => (rank, batch*row*col, 2)
        image = cplx_matmlpy(Ur, UrHx)   # (echo, rank, 2) * (rank, batch*row*col, 2) => (echo, batch*row*col, 2)
        image = image.permute(1, 0, 2)  # (echo, batch*row*col, 2) => (batch*row*col, echo, 2)
        image = image.view(-1, nrows, ncols, nechos, 2)  # (batch, row, col, echo, 2)
        
        coilComb = image.permute(0, 4, 3, 1, 2) # (batch, 2, echo, row, col)
        if scanner == 1:
            coilComb = fft_shift_col(coilComb, ncols, 1)  # for Siemens data
            coilComb = torch.flip(coilComb, dims=[4])  # for Siemens data
            coilComb[:, 1, ...] = - coilComb[:, 1, ...]  # for Siemens data
        if echo_cat:
            coilComb = torch_channel_concate(coilComb, necho) # (batch, 2*echo, row, col)
        return coilComb

def forward_multiEcho(image, csm, mask, flip, echo_cat=1, scanner=0):
    """
        forward operator for multi-echo GRE data
        scanner: 0 (GE) / 1 (Siemens)
    """
    if echo_cat:
        image = torch_channel_deconcate(image)  # (batch, 2, echo, row, col)
    if scanner == 1:
        image = torch.flip(image, dims=[4])  # for Siemens data
        image = fft_shift_col(image, ncols, 1)  # for Siemens data
        image[:, 1, ...] = - image[:, 1, ...]  # for Siemens data
    image = image.permute(0, 2, 3, 4, 1) # (batch, echo, row, col, 2)
    nrows = csm.size()[3]
    ncols = csm.size()[4]
    nechos = csm.size()[2]
    temp = cplx_mlpy(image, flip)
    temp = temp[:, None, ...]
    temp = cplx_mlpy(csm, temp)
    temp = fft_shift_row(temp, nrows, 1)
    temp = torch.fft(temp, 2)
    return cplx_mlpy(temp, mask)


class Back_forward_MS():
    '''
        forward and backward imaging model operator for MS lesion multi echo GRE data (kdata computed from iField)
        (echo dim as in the channel dim in CNN model)
    '''
    def __init__(
        self,
        csm,
        mask,
        flip,
        lambda_dll2,
        lambda_lowrank = 0,
        echo_cat = 1, # flag to concatenate echo dimension into channel
        necho = 11,
        kdata = None,
        csm_lowres = None,
        rank = 0
    ):
        self.nrows = csm.size()[3]
        self.ncols = csm.size()[4]
        self.nechos = csm.size()[2]
        self.csm = csm
        self.mask = mask
        self.lambda_dll2 = lambda_dll2
        self.lambda_lowrank = lambda_lowrank
        self.flip = flip
        self.echo_cat = echo_cat
        self.necho = necho
        self.kdata = kdata
        self.csm_lowres = csm_lowres
        self.rank = rank

    def AtA(
        self, 
        img, 
        use_dll2=1  # 1 for l2-x0 reg, 2 for l2-TV reg, 3 for l1-TV reg
    ):
        # forward
        if self.echo_cat:
            image = torch_channel_deconcate(img)  # (batch, 2, echo, row, col)
        else:
            image = img
        image = image.permute(0, 2, 3, 4, 1) # (batch, echo, row, col, 2)
        temp = image[:, None, ...] # multiply order matters (in torch implementation)
        temp = cplx_mlpy(self.csm, temp) # (batch, coil, echo, row, col, 2)
        temp = torch.fft(temp, 2)
        temp = fft_shift_row(fft_shift_col(temp, self.ncols, 1), self.nrows, 1) 
        temp = cplx_mlpy(temp, self.mask)
        # inverse
        temp = fft_shift_row(fft_shift_col(temp, self.ncols, 1), self.nrows, 1)
        coilImgs = torch.ifft(temp, 2)
        coilComb = torch.sum(
            cplx_mlpy(coilImgs, cplx_conj(self.csm)),
            dim=1,
            keepdim=False
        )
        coilComb = coilComb.permute(0, 4, 1, 2, 3) # (batch, 2, echo, row, col)
        if self.echo_cat:
            coilComb = torch_channel_concate(coilComb, self.necho) # (batch, 2*echo, row, col)
        if use_dll2 == 1:
            coilComb = coilComb + self.lambda_dll2 * img
        elif use_dll2 == 2:
            coilComb = coilComb + self.lambda_dll2 * divergence(gradient(img))
        elif use_dll2 == 3:
            coilComb = coilComb + self.lambda_dll2 * divergence(gradient(img) / torch.sqrt(gradient(img)**2+5e-4))  #1e-4 best, 5e-5 to have consistent result to ADMM
        return coilComb

    def low_rank_approx(self, img):
        img = img.permute(0, 2, 3, 1)  # (batch, row, col, 2*echo)
        s0 = img.size()
        M = s0[-1]  # M different contrast weighted images
        img = img.view(-1, M).permute(1, 0) # (M * N) with N voxels (samples) and M echos (features) 

        # row rank approximation of full resolution img
        tmp = torch.matmul(self.V.permute(1, 0), img)  # (k * M) * (M * N) => (k * N)
        tmp = torch.matmul(self.V, tmp)  # (M * k) * (k * N) => (M * N)
        tmp = tmp.permute(1, 0).view(s0[0], s0[1], s0[2], s0[3])  # (N * M) => (batch, row, col, 2*echo)
        img_low_rank = tmp.permute(0, 3, 1, 2)  # (batch, 2*echo, row, col)
        return img_low_rank

def backward_MS(kdata, csm, mask, flip, echo_cat=1, necho=11):
    """
    backward operator for MS lesion multi-echo GE data
    """
    nrows = kdata.size()[3]
    ncols = kdata.size()[4]
    nechos = kdata.size()[2]
    temp = cplx_mlpy(kdata, mask)
    temp = fft_shift_row(fft_shift_col(temp, ncols, 1), nrows, 1)
    temp = torch.ifft(temp, 2)
    coilComb = torch.sum(
        cplx_mlpy(temp, cplx_conj(csm)),
        dim=1,
        keepdim=False
    )
    coilComb = coilComb.permute(0, 4, 1, 2, 3) # (batch, 2, echo, row, col)
    if echo_cat:
        coilComb = torch_channel_concate(coilComb, necho) # (batch, 2*echo, row, col)
    return coilComb

def forward_MS(image, csm, mask, flip, echo_cat=1):
    """
        forward operator for MS lesion multi-echo GE data
    """
    if echo_cat:
        image = torch_channel_deconcate(image)  # (batch, 2, echo, row, col)
    image = image.permute(0, 2, 3, 4, 1) # (batch, echo, row, col, 2)
    nrows = csm.size()[3]
    ncols = csm.size()[4]
    nechos = csm.size()[2]
    temp = image[:, None, ...]
    temp = cplx_mlpy(csm, temp)
    temp = torch.fft(temp, 2)
    temp = fft_shift_row(fft_shift_col(temp, ncols, 1), nrows, 1)
    return cplx_mlpy(temp, mask)


# class OperatorsMultiEcho():
    # """
    #     forward and Jacobian operators of multi-echo gradient echo data
    # """
#     def __init__(
#         self,
#         mask,
#         csm,
#         M_0,
#         R_2,
#         phi_0,
#         f,
#         lambda_dll2=0
#     ):
#         self.device = csm.get_device()
#         self.num_samples = csm.shape[0]
#         self.num_coils = csm.shape[1]
#         self.num_echos = csm.shape[2]
#         self.num_rows = csm.shape[3] 
#         self.num_cols = csm.shape[4] 
#         self.csm = csm
#         self.mask = mask
#         self.M_0 = M_0.repeat(1, self.num_echos, 1, 1)
#         self.R_2 = R_2.repeat(1, self.num_echos, 1, 1)
#         self.phi_0 = phi_0.repeat(1, self.num_echos, 1, 1)
#         self.f = f.repeat(1, self.num_echos, 1, 1)
#         self.lambda_dll2 = lambda_dll2
        
#         # time slots for multi-echo data
#         self.time_intervals = torch.arange(0, self.num_echos)[None, :, None, None].float()
#         self.time_intervals = self.time_intervals.repeat(self.num_samples, 
#                               1, self.num_rows, self.num_cols).to(self.device)

#         self.tj_coils = self.time_intervals[:, None, ..., None]
#         self.tj_coils = torch.cat((self.tj_coils, torch.zeros(self.tj_coils.shape).to(
#                                    self.device)), dim=-1).repeat(1, self.num_coils, 1, 1, 1, 1)

#     def forward_operator(self, dx=0, flag=0):
#         # dx: (or M) torch.Size([batchsize, 1, nrows, ncols])
#         # flag = 0: no additional operation
#         # flag = 1: M_0, W/M0
#         # flag = 2: R_2, W*(-T)
#         # flag = 3: phi_0, W*(i)
#         # flag = 4: f, W*(iT)
#         if flag > 0:
#             dx = dx[:, None, ..., None].repeat(1, self.num_coils, self.num_echos, 1, 1, 1)
#             dx = torch.cat((dx, torch.zeros(dx.shape).to(self.device)), dim=-1)

#         img0 = (torch.exp(-self.R_2 * self.time_intervals))[..., None]
#         img0 = torch.cat((img0, torch.zeros(img0.shape).to(self.device)), dim=-1)
#         self.img0 = img0[:, None, ...].repeat(1, self.num_coils, 1, 1, 1, 1)

#         img1 = (self.M_0 * torch.exp(-self.R_2 * self.time_intervals))[..., None]
#         img1 = torch.cat((img1, torch.zeros(img1.shape).to(self.device)), dim=-1)
#         self.img1 = img1[:, None, ...].repeat(1, self.num_coils, 1, 1, 1, 1)

#         img2_real = torch.cos(self.phi_0 + self.f * self.time_intervals)
#         img2_imag = torch.sin(self.phi_0 + self.f * self.time_intervals)  
#         img2 = torch.cat((img2_real[..., None], img2_imag[..., None]), dim=-1)
#         self.img2 = img2[:, None, ...].repeat(1, self.num_coils, 1, 1, 1, 1)

#         if flag == 0:
#             img = cplx_mlpy(img1, img2)[:, None, ...]
#             img = img.repeat(1, self.num_coils, 1, 1, 1, 1)
#         elif flag == 1:
#             img = cplx_mlpy(img0, img2)[:, None, ...]
#             img = img.repeat(1, self.num_coils, 1, 1, 1, 1)
#         elif flag == 2:
#             img = cplx_mlpy(img1, img2)[:, None, ...]
#             img = img.repeat(1, self.num_coils, 1, 1, 1, 1)
#             img = cplx_mlpy(img, -self.tj_coils)
#         elif flag == 3:
#             img = cplx_mlpy(img1, img2)[:, None, ...]
#             img = img.repeat(1, self.num_coils, 1, 1, 1, 1)
#             img = torch.cat((-img[..., 1:2], img[..., 0:1]), dim=-1)
#         elif flag == 4:
#             img = cplx_mlpy(img1, img2)[:, None, ...]
#             img = img.repeat(1, self.num_coils, 1, 1, 1, 1)
#             img = torch.cat((-img[..., 1:2], img[..., 0:1]), dim=-1)
#             img = cplx_mlpy(img, self.tj_coils)
#         if flag > 0:
#             img = cplx_mlpy(img, dx)
#         img_coils = cplx_mlpy(self.csm, img)
#         kdata_coils = torch.fft(img_coils, signal_ndim=2)
#         kdata_coils_under = cplx_mlpy(self.mask, kdata_coils)
#         return kdata_coils_under

#     def jacobian_conj(self, kdata, flag=0):
#         # flag = 0: compute the whole Jacobian
#         # flag = 1: M_0, (W/M0)^H
#         # flag = 2: R_2, (-T)^HW^H
#         # flag = 3: phi_0, i^HW^H
#         # flag = 4: f, (iT)^HW^H
#         kdata_under = cplx_mlpy(self.mask, kdata)
#         img_under = torch.ifft(kdata_under, signal_ndim=2)
#         img_coils = cplx_mlpy(cplx_conj(self.csm), img_under)
#         if flag == 0 or 1: 
#             # for M_0, torch.Size([batchsize, ncoils, nechos, nrows, ncols, 2])
#             J1 = cplx_mlpy(self.img0, img_coils)
#             J1 = cplx_mlpy(cplx_conj(self.img2), J1)
#             J1 = torch.sum(J1, dim=(1,2), keepdim=False)[:, None, ..., 0]
#             J = J1
#         if flag == 0 or 2: 
#             # for R_2, the same dim
#             J2 = cplx_mlpy(self.img1, img_coils)
#             J2 = cplx_mlpy(cplx_conj(self.img2), J2)
#             J2 = cplx_mlpy(-self.tj_coils, J2)
#             J2 = torch.sum(J2, dim=(1,2), keepdim=False)[:, None, ..., 0]
#             J = J2
#         if flag == 0 or 3: 
#             # for phi_0
#             J3 = cplx_mlpy(self.img1, img_coils)
#             J3 = cplx_mlpy(cplx_conj(self.img2), J3)
#             J3 = torch.cat((J3[..., 1:2], -J3[..., 0:1]), dim=-1)
#             J3 = torch.sum(J3, dim=(1,2), keepdim=False)[:, None, ..., 0]
#             J = J3
#         if flag == 0 or 4: 
#             # for f
#             J4 = cplx_mlpy(self.img1, img_coils)
#             J4 = cplx_mlpy(cplx_conj(self.img2), J4)
#             J4 = cplx_mlpy(self.tj_coils, J4)
#             J4 = torch.cat((J4[..., 1:2], -J4[..., 0:1]), dim=-1)
#             J4 = torch.sum(J4, dim=(1,2), keepdim=False)[:, None, ..., 0]
#             J = J4
#         if flag == 0: 
#             J = torch.cat((J1, J2, J3, J4), dim=1)
#         J[J>10] = 0
#         print('Max value in the Jacobian matrix is = {0}'.format(torch.max(J)))
#         return J

#     def AtA(self, dx, flag=1, use_dll2=1):
#         # dx: (or M) torch.Size([batchsize, 1, nrows, ncols])
#         kdata = self.forward_operator(dx=dx, flag=flag)
#         if use_dll2 == 1:
#             return self.jacobian_conj(kdata=kdata, flag=flag) + self.lambda_dll2 * dx
#         else:
#             return self.jacobian_conj(kdata=kdata, flag=flag)

class OperatorsMultiEcho():
    """
        forward and Jacobian operators of multi-echo gradient echo data
    """
    def __init__(
        self,
        M_0,
        R_2,
        phi_0,
        f,
        num_echos=3,
        te1=0.003224, # 0.0043 for Siemens, 0.003224 for GE
        delta_te=0.003884  # 0.0048 for Siemens, 0.003884 for GE
    ):
        self.device = M_0.get_device()
        self.num_samples = M_0.shape[0]
        self.num_coils = 1
        self.num_echos = num_echos
        self.num_rows = M_0.shape[2] 
        self.num_cols = M_0.shape[3] 
        self.M_0 = M_0.repeat(1, self.num_echos, 1, 1)
        self.R_2 = R_2.repeat(1, self.num_echos, 1, 1)
        self.phi_0 = phi_0.repeat(1, self.num_echos, 1, 1)
        self.f = f.repeat(1, self.num_echos, 1, 1)
        
        # time slots for multi-echo data
        self.time_intervals = torch.arange(0, self.num_echos) * delta_te
        self.time_intervals += te1
        self.time_intervals = self.time_intervals[None, :, None, None].float()
        self.time_intervals = self.time_intervals.repeat(self.num_samples, 
                              1, self.num_rows, self.num_cols).to(self.device)

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

        self.W = cplx_mlpy(self.img1, self.img2)
        self.WH = cplx_mlpy(cplx_conj(self.img2), self.img1)

    def forward_operator(self, flag=0):
        # flag = 0: no additional operation
        # flag = 1: M_0, W/M0
        # flag = 2: R_2, W*(-T)
        # flag = 3: phi_0, W*(i)
        # flag = 4: f, W*(iT)
        if flag == 0:
            img = self.W
        elif flag == 1:
            img = cplx_mlpy(self.img0, self.img2)
        elif flag == 2:
            img = self.W
            img = cplx_mlpy(img, -self.tj_coils)
        elif flag == 3:
            img = self.W
            img = torch.cat((-img[..., 1:2], img[..., 0:1]), dim=-1)
        elif flag == 4:
            img = self.W
            img = torch.cat((-img[..., 1:2], img[..., 0:1]), dim=-1)
            img = cplx_mlpy(img, self.tj_coils)
        return img

    def jacobian_conj(self, img, flag=0):
        # flag = 0: compute the whole Jacobian
        # flag = 1: M_0, (W/M0)^H
        # flag = 2: R_2, (-T)^HW^H
        # flag = 3: phi_0, i^HW^H
        # flag = 4: f, (iT)^HW^H
        if flag == 0 or 1: 
            # for M_0, torch.Size([batchsize, ncoils, nechos, nrows, ncols, 2])
            J1 = cplx_mlpy(self.img0, img)
            J1 = cplx_mlpy(cplx_conj(self.img2), J1)
            J1 = torch.sum(J1, dim=(1,2), keepdim=False)[:, None, ..., 0]
            J = J1
        if flag == 0 or 2: 
            # for R_2, the same dim
            J2 = cplx_mlpy(self.WH, img)
            J2 = cplx_mlpy(-self.tj_coils, J2)
            J2 = torch.sum(J2, dim=(1,2), keepdim=False)[:, None, ..., 0]
            J = J2
        if flag == 0 or 3: 
            # for phi_0
            J3 = cplx_mlpy(self.WH, img)
            J3 = torch.cat((J3[..., 1:2], -J3[..., 0:1]), dim=-1)
            J3 = torch.sum(J3, dim=(1,2), keepdim=False)[:, None, ..., 0]
            J = J3
        if flag == 0 or 4: 
            # for f
            J4 = cplx_mlpy(self.WH, img)
            J4 = cplx_mlpy(self.tj_coils, J4)
            J4 = torch.cat((J4[..., 1:2], -J4[..., 0:1]), dim=-1)
            J4 = torch.sum(J4, dim=(1,2), keepdim=False)[:, None, ..., 0]
            J = J4
        if flag == 0: 
            J = torch.cat((J1, J2, J3, J4), dim=1)
        # J[J>10] = 0
        # print('Max value in the Jacobian matrix is = {0}'.format(torch.max(J)))
        return J

    def AtA(self, flag=1, use_dll2=1, lambda_dll2=1):
        img = self.forward_operator(flag=flag)
        if use_dll2 == 1:
            return self.jacobian_conj(img=img, flag=flag) + lambda_dll2 * torch.ones(
                                 self.num_samples, 1, self.num_rows, self.num_cols).cuda()
        else:
            return self.jacobian_conj(img=img, flag=flag)


def hann_filter(matrix_size, voxel_size, fc=None):
    '''
        hann filter (numpy array)
    '''
    for idx, _ in enumerate(voxel_size):
        voxel_size[idx] = voxel_size[idx] / voxel_size[1]
    sy = 1
    # sx = matrix_size[1] / matrix_size[0] / voxel_size[0]
    sx = 1

    x = np.arange(-matrix_size[1]/2*sy, matrix_size[1]/2*sy, sy)
    y = np.arange(-matrix_size[0]/2*sx, matrix_size[0]/2*sx, sx)
    Y, X = np.meshgrid(x, y)

    n = np.pi * np.sqrt(Y**2 + X**2) / (fc/2)
    n[n>np.pi] = np.pi
    H = 0.5 * (1 + np.cos(n))
    return H


def hann_low(image, voxel_size, fc):
    kdata = np.fft.fftshift(np.fft.fft2(image))
    H = hann_filter(image.shape, voxel_size, fc)
    out = np.fft.ifft2(np.fft.fftshift(kdata * H))
    return out


def HPphase(image, voxel_size):
    matrix_size = image.shape
    cPhase = np.zeros(matrix_size)
    fc = matrix_size[0] / 8 * 3
    for slice in range(matrix_size[-1]):
        cPhase[:, :, slice] = np.angle(image[:, :, slice] / hann_low(image[:, :, slice], voxel_size, fc))
    return cPhase








