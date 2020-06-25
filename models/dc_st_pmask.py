'''
    straight strough estimation of pmask for T2w images (2D cartesian along two phase encoding direction)
    start on 02/03/2020
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

from models.dc_blocks import *
from models.unet_blocks import *
from models.initialization import *
from models.resBlocks import *
from models.unet import *
from models.straight_through_layers import *
        

class DC_ST_Pmask(nn.Module):

    def __init__(
        self,
        input_channels,
        filter_channels,
        lambda_dll2, # initializing lambda_dll2
        lambda_tv=0.01,
        rho_penalty=0.01,
        ncoil=32,
        nrow=256,
        ncol=192,
        flag_ND=2, # 0 for 1D Cartesian along row direciton, 1 for 1D Cartesian along column direction, 
                   # 2 for 2D Cartesian, 3 for variable density random
        flag_solver=0,  # -3 for Quasi-Newton without learnable parameters,
                        # -2 for Quasi-Newton with learnable parameters,
                        # -1 for deep Quasi-Newton with resnet denoiser (MoDL),
                        #  0 for deep ADMM with resnet denoiser, 
                        #  1 for ADMM solver with learnable parameters, 
                        #  2 for ADMM solver without learnable parameters.
                        #  3 for Unet recon
        flag_TV=0,  # 0 for l2 gradient reg, 1 for l1 gradient reg
        K=1,
        unc_map=False,
        slope=0.25,
        passSigmoid=False,
        stochasticSampling=True,
        rescale=False,
        samplingRatio=0.1, # sparsity level of the sampling mask
        flag_fix=0,  # 0 not fix, 1 LOUPE, 2 VD, 3 Uniform,
        flag_precond=0,  # o not using preconditional, 1 use
        flag_print_precond=0,
        contrast='T2',
        pmask_BO=None  # flag
    ):
        super(DC_ST_Pmask, self).__init__()
        self.K = K
        self.unc_map = unc_map
        self.slope = slope
        self.passSigmoid = passSigmoid
        self.stochasticSampling = stochasticSampling
        self.ncoil = ncoil
        self.nrow = nrow
        self.ncol = ncol
        self.flag_ND = flag_ND
        self.flag_solver = flag_solver
        self.flag_TV = flag_TV 
        self.rescale = rescale
        self.samplingRatio = samplingRatio
        self.flag_fix = flag_fix
        self.flag_precond = flag_precond
        self.flag_print_precond = flag_print_precond
        self.contrast = contrast
        self.pmask_BO = pmask_BO

        # flag for sampling pattern designs
        if flag_ND == 0:
            temp = (torch.rand(nrow)-0.5)*30
            temp[nrow//2-13 : nrow//2+12] = 15
        elif flag_ND == 1:
            temp = (torch.rand(ncol)-0.5)*30
            temp[ncol//2-13 : ncol//2+12] = 15
        elif flag_ND == 2:
            temp1 = (torch.rand(nrow)-0.5)*30
            temp2 = (torch.rand(ncol)-0.5)*30
            temp1[nrow//2-13 : nrow//2+12] = 15
            temp2[ncol//2-13 : ncol//2+12] = 15
        elif flag_ND == 3:
            temp = (torch.rand(nrow, ncol)-0.5)*30
            temp[nrow//2-13 : nrow//2+12, ncol//2-13 : ncol//2+12] = 15
        if flag_ND != 2:
            self.weight_parameters = nn.Parameter(temp, requires_grad=True)
        else:
            self.weight_parameters1 = nn.Parameter(temp1, requires_grad=True)
            self.weight_parameters2 = nn.Parameter(temp2, requires_grad=True)

        # flag for all the solvers
        if flag_solver == -3:
            self.lambda_dll2 = nn.Parameter(torch.ones(1)*lambda_dll2, requires_grad=False)
            # self.lambda_dll2 = nn.Parameter(torch.ones(1)*lambda_dll2, requires_grad=False)
        elif flag_solver == -2:
            self.lambda_dll2 = nn.Parameter(torch.ones(1)*lambda_dll2, requires_grad=True)
        elif -2 < flag_solver < 1:
            # self.resnet_block = []
            # layers = ResBlock(input_channels, filter_channels, use_norm=2, unc_map=unc_map)
            # for layer in layers:
            #     self.resnet_block.append(layer)
            # self.resnet_block = nn.Sequential(*self.resnet_block)
            # self.resnet_block.apply(init_weights)

            self.resnet_block = ResBlock2(input_channels, filter_channels, input_channels, use_norm=2)

            # self.resnet_block = Unet(input_channels, input_channels, num_filters=[2**i for i in range(4, 8)], use_bn=2)

            self.lambda_dll2 = nn.Parameter(torch.ones(1)*lambda_dll2, requires_grad=False)
        elif flag_solver == 1:
            self.lambda_tv = nn.Parameter(torch.ones(1)*lambda_tv, requires_grad=True)
            self.rho_penalty = nn.Parameter(torch.ones(1)*rho_penalty, requires_grad=True)
        elif flag_solver == 2:
            self.lambda_tv = nn.Parameter(torch.ones(1)*lambda_tv, requires_grad=False)
            self.rho_penalty = nn.Parameter(torch.ones(1)*rho_penalty, requires_grad=False)
        elif flag_solver == 3:
            self.unet = Unet(input_channels, input_channels, num_filters=[2**i for i in range(5, 10)], 
                             use_deconv=0, skip_connect=1)

        # flag for using preconditioning:
        if self.flag_precond == 1:
            self.preconditioner = Unet(input_channels, input_channels, num_filters=[2**i for i in range(4, 8)])

    def rescalePmask(self, Pmask, samplingRatio):
        device = Pmask.get_device()
        xbar = torch.mean(Pmask)
        r = samplingRatio/xbar
        beta = (1-samplingRatio) / (1-xbar)
        le = (r<=1).to(device, dtype=torch.float32)
        return le * Pmask * r + (1-le) * (1 - (1-Pmask) * beta)

    # def rescalePmask2(self, Pmask, samplingRatio):
    #     idx = 0
    #     while torch.mean(Pmask) > samplingRatio:
    #         Pmask = Pmask**1.1
    #         idx += 1
    #         # print(idx)
    #     return Pmask

    def samplingPmask(self, Pmask_rescaled, flag_ND):
        if flag_ND == 0:
            Mask1D = bernoulliSample.apply(Pmask_rescaled)
            Mask = Mask1D.repeat(self.ncol, 1).transpose(0,1)
        elif flag_ND == 1:
            Mask1D = bernoulliSample.apply(Pmask_rescaled)
            Mask = Mask1D.repeat(self.nrow, 1)
        elif flag_ND == 3:
            if self.stochasticSampling:
                Mask = bernoulliSample.apply(Pmask_rescaled)
            else:
                thresh = torch.rand(Pmask_rescaled.shape).to('cuda')
                Mask = 1/(1+torch.exp(-12*(Pmask_rescaled-thresh)))
        return Mask

    def generateMask(self):

        if self.flag_ND != 2:
            if self.passSigmoid:
                self.Pmask = passThroughSigmoid.apply(self.slope * self.weight_parameters)
            else:
                self.Pmask = 1 / (1 + torch.exp(-self.slope * self.weight_parameters))
            if self.rescale:
                self.Pmask_rescaled = self.rescalePmask(self.Pmask, self.samplingRatio)
            else:
                self.Pmask_rescaled = self.Pmask
            self.Mask = self.samplingPmask(self.Pmask_rescaled, self.flag_ND)
            return self.Mask

        elif self.flag_ND == 2:
            if self.passSigmoid:
                self.Pmask1 = passThroughSigmoid.apply(self.slope * self.weight_parameters1)
                self.Pmask2 = passThroughSigmoid.apply(self.slope * self.weight_parameters2)
            else:
                self.Pmask1 = 1 / (1 + torch.exp(-self.slope * self.weight_parameters1))
                self.Pmask2 = 1 / (1 + torch.exp(-self.slope * self.weight_parameters2))
            self.Pmask = self.Pmask1.repeat(self.ncol, 1).transpose(0,1) + self.Pmask2.repeat(self.nrow, 1)
            if self.rescale:
                self.Pmask_rescaled1 = self.rescalePmask(self.Pmask1, np.sqrt(self.samplingRatio))
                self.Pmask_rescaled2 = self.rescalePmask(self.Pmask2, np.sqrt(self.samplingRatio))
            else:
                self.Pmask_rescaled1 = self.Pmask1
                self.Pmask_rescaled2 = self.Pmask2
            Mask1 = self.samplingPmask(self.Pmask_rescaled1, flag_ND=0)
            Mask2 = self.samplingPmask(self.Pmask_rescaled2, flag_ND=1)
            self.Mask = Mask1 * Mask2
            return self.Mask
        
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

    def forward(self, kdata, csms):
        device = kdata.get_device()
        if self.flag_ND != 2:
            self.weight_parameters = self.weight_parameters.to(device)
        else:
            self.weight_parameters1 = self.weight_parameters1.to(device)
            self.weight_parameters2 = self.weight_parameters2.to(device)
        masks = self.generateMask()[None, :, :, None]
        self.masks = self.Mask
        # keep the calibration region
        masks[:, masks.size()[1]//2-13:masks.size()[1]//2+12,  masks.size()[2]//2-13:masks.size()[2]//2+12, :] = 1
        # # fftshift for simulated kspace data
        # masks = torch.cat((masks[:, self.nrow//2:self.nrow, ...], masks[:, 0:self.nrow//2, ...]), dim=1)
        # masks = torch.cat((masks[:, :, self.ncol//2:self.ncol, :], masks[:, :, 0:self.ncol//2, :]), dim=2)
        # load fixed mask
        if self.flag_fix == 1:
            masks = load_mat('/data/Jinwei/'+self.contrast+'_slice_recon_GE/Fixed_masks/LOUPE.mat', 'Mask')  # LOUPE/VD/Uniform
        elif self.flag_fix == 2:
            masks = load_mat('/data/Jinwei/'+self.contrast+'_slice_recon_GE/Fixed_masks/VD.mat', 'Mask')  # LOUPE/VD/Uniform
        elif self.flag_fix == 3:
            masks = load_mat('/data/Jinwei/'+self.contrast+'_slice_recon_GE/Fixed_masks/Uniform.mat', 'Mask')  # LOUPE/VD/Uniform
        if self.pmask_BO is not None:
            u = np.random.uniform(0, np.mean(self.pmask_BO)/self.samplingRatio, size=(256, 192))
            masks = self.pmask_BO > u
            masks[128-13:128+12, 96-13:96+12] = 1

            # plt.figure()
            # plt.imshow(masks)
            # plt.savefig('mask.png')
            # plt.close()

        if self.flag_fix:
            masks = masks[np.newaxis, ..., np.newaxis]
            masks = torch.tensor(masks, device=device).float()
        # to complex data
        masks = torch.cat((masks, torch.zeros(masks.shape).to(device)),-1)
        # add coil dimension
        masks = torch.cat(self.ncoil*[masks])
        x = self.At(kdata, masks, csms)    
        # input
        x_start = x
        # generate preconditioner
        if self.flag_precond == 1:
            precond = 9 / (1 + torch.exp(-0.1 * self.preconditioner(x_start))) + 1
            precond[:, 1, ...] = 0
        else:
            precond = 0

        # Quasi_newton
        if self.flag_solver < -1:
            epsilon = (torch.ones(1)*1e-7).to(device)
            self.lambda_dll2 = self.lambda_dll2.to(device)
            A = Back_forward(csms, masks, self.lambda_dll2)
            Xs = []
            for i in range(self.K):
                x_old = x
                if self.flag_TV == 0:
                    rhs = x_start - A.AtA(x, use_dll2=2)
                    dc_layer = DC_layer(A, rhs, flag_precond=self.flag_precond, precond=precond, use_dll2=2)
                elif self.flag_TV == 1:
                    rhs = x_start - A.AtA(x, use_dll2=3)
                    dc_layer = DC_layer(A, rhs, flag_precond=self.flag_precond, precond=precond, use_dll2=3)

                delta_x = dc_layer.CG_iter(max_iter=20)  # 20 for test
                x = x + delta_x
                Xs.append(x)
                # if i % 10 == 0:
                # print('Relative Change: {0}'.format(torch.mean(torch.abs((x-x_old)/(x_old+epsilon)))))
            if self.flag_print_precond == 0:
                return Xs
            else:
                return Xs, precond

        # Deep Quasi_newton (MoDL)
        elif self.flag_solver == -1:
            self.lambda_dll2 = self.lambda_dll2.to(device)
            A = Back_forward(csms, masks, self.lambda_dll2)
            Xs = []
            for i in range(self.K):
                x_block = self.resnet_block(x)
                x_block1 = x - x_block[:, 0:2, ...]
                rhs = x_start + self.lambda_dll2*x_block1
                dc_layer = DC_layer(A, rhs, flag_precond=self.flag_precond, precond=precond, use_dll2=1)
                x = dc_layer.CG_iter()
                Xs.append(x)
            if self.flag_print_precond == 0:
                return Xs
            else:
                return Xs, precond

        # Deep ADMM
        elif self.flag_solver == 0:
            self.lambda_dll2 = self.lambda_dll2.to(device)
            A = Back_forward(csms, masks, self.lambda_dll2)
            Xs = []
            uk = torch.zeros(x_start.size()).to(device)
            for i in range(self.K):
                # update auxiliary variable v
                v_block = self.resnet_block(x+uk/self.lambda_dll2)
                v_block1 = x + uk/self.lambda_dll2 - v_block[:, 0:2, ...]
                # update x using CG block
                x0 = v_block1 - uk/self.lambda_dll2
                rhs = x_start + self.lambda_dll2*x0
                dc_layer = DC_layer(A, rhs, flag_precond=self.flag_precond, precond=precond, use_dll2=1)
                x = dc_layer.CG_iter()
                Xs.append(x)
                # update dual variable uk
                uk = uk + self.lambda_dll2*(x - v_block1)
            if self.flag_print_precond == 0:
                return Xs
            else:
                return Xs, precond

        # ADMM
        elif 0 < self.flag_solver < 3:
            epsilon = (torch.ones(1)*1e-7).to(device)
            self.lambda_tv = self.lambda_tv.to(device)
            self.rho_penalty = self.rho_penalty.to(device)
            A = Back_forward(csms, masks, self.rho_penalty)
            Xs = []
            Unc_maps = []
            wk = torch.zeros(x_start.size()+(2,)).to(device)
            etak = torch.zeros(x_start.size()+(2,)).to(device)
            # etak = gradient(x_start).to(device)
            for i in range(self.K):
                # update auxiliary variable wk through threshold
                if self.flag_TV == 0:  # recon not change so much with large lambda in this case
                    wk = (self.rho_penalty*gradient(x) + etak) / (2*self.lambda_tv + self.rho_penalty)
                elif self.flag_TV == 1:
                    ek = gradient(x) + etak/self.rho_penalty
                    wk = ek.sign() * torch.max(torch.abs(ek) - self.lambda_tv/self.rho_penalty, torch.zeros(ek.size()).to(device))

                x_old = x
                # update x using CG block
                rhs = x_start + self.rho_penalty*divergence(wk) - divergence(etak)
                dc_layer = DC_layer(A, rhs, flag_precond=self.flag_precond, precond=precond, use_dll2=2)
                if self.flag_fix == 3:
                    x = dc_layer.CG_iter(max_iter=30)  # 20/30 for test (30 only for uniform mask)
                elif self.flag_fix > 0:
                    x = dc_layer.CG_iter(max_iter=20)  # 20/30 for test (30 only for uniform mask)
                else:
                    x = dc_layer.CG_iter(max_iter=10)  # for training
                Xs.append(x)
                
                # update dual variable etak
                etak = etak + self.rho_penalty * (gradient(x) - wk)
                # if i % 10 == 0:
                # print('Relative Change: {0}'.format(torch.mean(torch.abs((x-x_old)/(x_old+epsilon)))))
            if self.flag_print_precond == 0:
                return Xs
            else:
                return Xs, precond

        elif self.flag_solver == 3:
            Xs = []
            x = self.unet(x_start)
            Xs.append(x)
            return Xs

