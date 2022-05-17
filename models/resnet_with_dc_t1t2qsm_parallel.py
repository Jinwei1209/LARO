"""
    MoDL for Cardiac QSM data and multi_echo GRE brain data (kspace)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dc_blocks import *
from models.unet_blocks import *
from models.initialization import *
from models.resBlocks import *
from models.danet import daBlock, CAM_Module  
from models.fa import faBlockNew
from models.unet import *
from models.straight_through_layers import *
from models.BCRNN import BCRNNlayer, CRNNcell, Conv2dFT, MultiLevelBCRNNlayer
from models.BCLSTM import BCLSTMlayer
from fits.fits import fit_R2_LM, fit_complex, arlo
from utils.data import *
from utils.operators import *
from torch.utils.checkpoint import checkpoint


class Resnet_with_DC2(nn.Module):
    '''
        For multi_echo GRE brain data
    '''
    def __init__(
        self,
        input_channels,
        filter_channels,
        lambda_dll2, # initializing lambda_dll2
        lambda_tv=1e-3,
        rho_penalty=1e-2,
        necho=10, # number of echos of input
        necho_pred=0,  # number of echos to predict
        nrow=206,
        ncol=80,
        ncoil=12,
        delta_TE=0.003384,
        K=1,  # number of unrolls
        U=None,  # temporal principle vectors
        rank=0,  # rank > 0 if incorporate low rank compressor in forward model
        flag_compressor=0, # flag of different types of forward models with temporal compressor 
        echo_cat=1, # flag to concatenate echo dimension into channel
        flag_2D=1,  # flag to use 2D undersampling (variable density)
        flag_solver=0,  # 0 for deep Quasi-newton, 1 for deep ADMM,
                        # 2 for TV Quasi-newton, 3 for TV ADMM.
        flag_precond=0, # flag to use the preconditioner in the CG layer
        flag_loupe=0, # 1: same mask across echos, 2: mask for each echo
        flag_temporal_pred=0,  # predict the later echo images from the former echos
        norm_last=0, # put normalization after relu
        flag_temporal_conv=0,
        flag_convFT=0,  # flag to use conv2DFT layer
        flag_BCRNN=0,
        flag_hidden=1, # BCRNN hidden feature recurrency
        flag_unet=0,  # flag to use unet as denoiser
        flag_multi_level=0,  # flag to extract multi-level features and put into U-net
        flag_bn=2,  # flag to use group normalization: 0: no normalization, 2: use group normalization
        flag_t2w_redesign=0,
        flag_t1w_only=0,
        slope=0.25,
        passSigmoid=0,
        stochasticSampling=1,
        rescale=1,
        samplingRatio=0.2, # sparsity level of the sampling mask
        flag_att=0, # flag to use attention after temporal feature fusion
        # random=0, # flag to multiply the input data with a random complex number
        flag_cp=0,
        flag_dataset=1,  # 1: 'CBIC', 0: 'MS'
        flag_mc_fusion=1,  # flag to fuse
        split_K=[2, 4, 6],  # split K unrolls into several chunks 
    ):
        super(Resnet_with_DC2, self).__init__()
        self.resnet_block = []
        self.necho = necho
        self.necho_pred_value = necho_pred
        self.delta_TE = delta_TE
        self.nrow = nrow
        self.ncol = ncol
        self.ncoil = ncoil
        self.echo_cat = echo_cat
        self.flag_2D = flag_2D
        self.flag_solver = flag_solver
        self.flag_precond = flag_precond
        self.flag_loupe = flag_loupe
        self.flag_temporal_pred = flag_temporal_pred
        self.flag_BCRNN = flag_BCRNN
        self.flag_hidden = flag_hidden
        self.flag_unet = flag_unet
        self.flag_multi_level = flag_multi_level
        self.flag_bn = flag_bn
        self.flag_t2w_redesign = flag_t2w_redesign
        self.flag_t1w_only = flag_t1w_only
        self.slope = slope
        self.passSigmoid = passSigmoid
        self.stochasticSampling = stochasticSampling
        self.rescale = rescale
        self.samplingRatio = samplingRatio
        self.flag_att = flag_att
        # self.random = random
        self.flag_cp = flag_cp
        self.flag_dataset = flag_dataset
        self.flag_mc_fusion = flag_mc_fusion
        self.split_K = split_K

        if self.flag_solver <= 1:
            if self.flag_BCRNN == 0:
                if self.echo_cat == 1:
                    layers = ResBlock(input_channels, filter_channels, output_dim=input_channels, \
                                    use_norm=2, norm_last=norm_last, flag_temporal_conv=flag_temporal_conv)
                    # self.resnet_block = MultiBranch(input_channels, filter_channels, output_dim=input_channels)
                elif self.echo_cat == 0:
                    layers = ResBlock_3D(input_channels, filter_channels, \
                                    output_dim=input_channels, use_norm=2)
                for layer in layers:
                    self.resnet_block.append(layer)
                self.resnet_block = nn.Sequential(*self.resnet_block)
                self.resnet_block.apply(init_weights)

            elif self.flag_BCRNN > 0:
                n_ch = 2  # number of channels
                nd = 5  # number of CRNN/BCRNN/CNN layers in each iteration
                nf = 64  # number of filters
                ks = 3  # kernel size
                if self.flag_t1w_only == 0:
                    unet_first_level = 6
                    unet_last_level = 10
                    flag_slim = False
                else:
                    unet_first_level = 5
                    unet_last_level = 8
                    flag_slim = False
                    nf = 64
                self.n_ch = n_ch
                self.nd = nd
                self.nf = nf
                self.ks = ks
                if self.flag_BCRNN == 1:
                    print('Use BCRNN')
                    self.bcrnn = BCRNNlayer(n_ch, nf, ks, flag_convFT, flag_bn, flag_hidden)
                    if self.flag_t1w_only == 0:
                        self.featureExtractor_t1t2 = CRNNcell(n_ch*2, nf, ks, flag_hidden=0)
                    else:
                        self.featureExtractor_t1t2 = CRNNcell(n_ch*1, nf, ks, flag_hidden=0)

                if self.flag_unet == 0:
                    self.conv1_x = nn.Conv2d(nf, nf, ks, padding = ks//2)
                    # self.conv1_x = Conv2dFT(nf, nf, ks)
                    self.bn1_x = nn.GroupNorm(nf, nf)
                    # self.conv1_h = nn.Conv2d(nf, nf, ks, padding = ks//2)
                    self.conv2_x = nn.Conv2d(nf, nf, ks, padding = ks//2)
                    # self.conv2_x = Conv2dFT(nf, nf, ks)
                    self.bn2_x = nn.GroupNorm(nf, nf)
                    # self.conv2_h = nn.Conv2d(nf, nf, ks, padding = ks//2)
                    self.conv3_x = nn.Conv2d(nf, nf, ks, padding = ks//2)
                    # self.conv3_x = Conv2dFT(nf, nf, ks)
                    self.bn3_x = nn.GroupNorm(nf, nf)
                    # self.conv3_h = nn.Conv2d(nf, nf, ks, padding = ks//2)
                    self.conv4_x = nn.Conv2d(nf, n_ch, ks, padding = ks//2)
                    # self.conv4_x = Conv2dFT(nf, n_ch, ks)
                    self.relu = nn.ReLU(inplace=True)
                if self.flag_unet == 1:
                    self.denoiser = Unet(
                        input_channels=nf,
                        output_channels=n_ch,
                        num_filters=[2**i for i in range(unet_first_level, unet_last_level)],
                        use_bn=flag_bn,
                        use_deconv=1,
                        skip_connect=False,
                        slim=flag_slim,
                        convFT=flag_convFT
                    )
                    if self.flag_t1w_only == 0:
                        self.denoiser_t1t2 = Unet(
                            input_channels=nf,
                            output_channels=n_ch*2,
                            num_filters=[2**i for i in range(unet_first_level, unet_last_level)],
                            use_bn=flag_bn,
                            use_deconv=1,
                            skip_connect=False,
                            slim=flag_slim,
                            convFT=flag_convFT
                        )
                    else:
                        self.denoiser_t1t2 = Unet(
                            input_channels=nf,
                            output_channels=n_ch*1,
                            num_filters=[2**i for i in range(unet_first_level, unet_last_level)],
                            use_bn=flag_bn,
                            use_deconv=1,
                            skip_connect=False,
                            slim=flag_slim,
                            convFT=flag_convFT
                        )
            
            self.lambda_dll2 = nn.Parameter(torch.ones(1)*lambda_dll2, requires_grad=True)
        
        elif self.flag_solver == 2:
            self.lambda_dll2 = nn.Parameter(torch.ones(1)*lambda_dll2, requires_grad=False)
        
        elif self.flag_solver == 3:
            self.rho_penalty = nn.Parameter(torch.ones(1)*rho_penalty, requires_grad=False)
            self.lambda_tv = nn.Parameter(torch.ones(1)*lambda_tv, requires_grad=False)

        if self.flag_temporal_pred:
            self.densenet = DenseBlock(2*necho, 2*(10-necho))
        
        if self.flag_att == 1:
            print('Use Temporal Attention')
            self.attBlock = CAM_Module(self.nf)

        self.K = K
        self.U = U
        self.rank = rank
        self.flag_compressor = flag_compressor
        self.lambda_lowrank = nn.Parameter(torch.ones(1)*lambda_dll2, requires_grad=True)

        # flag for using preconditioning:
        if self.flag_precond == 1:
            print('Apply preconditioning in the CG block')
            self.preconditioner = Unet(2*self.necho, 2*self.necho, num_filters=[2**i for i in range(4, 8)])

        # flag for mask learning strategy
        if self.flag_2D == 1:
            print('Use 2D variable density sampling strategy')

            if self.flag_loupe == 2 or self.flag_loupe == -2 or self.flag_loupe == -3:
                # temp = (torch.rand(self.necho, self.nrow, self.ncol)-0.5)*30  # (necho, nrow, ncol)
                # temp[:, self.nrow//2-13 : self.nrow//2+12, self.ncol//2-13 : self.ncol//2+12] = 15
                temp = torch.zeros(self.necho, self.nrow, self.ncol)  # (necho, nrow, ncol)
                self.weight_parameters = nn.Parameter(temp, requires_grad=True)
            elif self.flag_loupe < 2:
                # temp = (torch.rand(self.nrow, self.ncol)-0.5)*30  # (nrow, ncol)
                # temp[self.nrow//2-13 : self.nrow//2+12, self.ncol//2-13 : self.ncol//2+12] = 15
                temp = torch.zeros(self.nrow, self.ncol)  # (nrow, ncol)
                self.weight_parameters = nn.Parameter(temp, requires_grad=True)
            
        else:
            print('Use 1D variable density sampling strategy')
            if self.flag_loupe == 2 or self.flag_loupe == -2:
                temp = (torch.rand(self.necho, self.nrow)-0.5)*30  # (necho, nrow)
                temp[:, self.nrow//2-13 : self.nrow//2+12] = 15
                self.weight_parameters = nn.Parameter(temp, requires_grad=True)
            elif self.flag_loupe < 2:
                temp = (torch.rand(self.nrow)-0.5)*30  # (nrow)
                temp[self.nrow//2-13 : self.nrow//2+12] = 15
                self.weight_parameters = nn.Parameter(temp, requires_grad=True)

    def generateMask(self, weight_parameters):
        if self.passSigmoid:
            Pmask = passThroughSigmoid.apply(self.slope * weight_parameters)
        else:
            Pmask = 1 / (1 + torch.exp(-self.slope * weight_parameters))
        if self.rescale:
            Pmask_rescaled = self.rescalePmask(Pmask, self.samplingRatio)
            self.Pmask_rescaled = Pmask_rescaled
        else:
            Pmask_rescaled = Pmask
            self.Pmask_rescaled = Pmask_rescaled
        
        if self.flag_loupe == 1:
            masks = self.samplingPmask(Pmask_rescaled)[:, :, None] # (nrow, ncol, 1)
            # keep central calibration region to 1
            masks[self.nrow//2-9:self.nrow//2+9, self.ncol//2-9:self.ncol//2+9, :] = 1
            # to complex data
            masks = torch.cat((masks, torch.zeros(masks.shape).to('cuda')),-1) # (nrow, ncol, 2)
            # add echo dimension
            masks = masks[None, ...] # (1, nrow, ncol, 2)
            masks = torch.cat(self.necho*[masks]) # (necho, nrow, ncol, 2)
            # add coil dimension
            masks = masks[None, ...] # (1, necho, nrow, ncol, 2)
            masks = torch.cat(self.ncoil*[masks]) # (ncoil, necho, nrow, ncol, 2)
            # add batch dimension
            masks = masks[None, ...] # (1, ncoil, necho, nrow, ncol, 2)
            return masks
        elif self.flag_loupe == 2:
            masks = self.samplingPmask(Pmask_rescaled)[..., None] # (necho, nrow, ncol, 1)
            # keep central calibration region to 1
            if self.flag_t2w_redesign == 0:
                masks[:, self.nrow//2-9:self.nrow//2+9, self.ncol//2-9:self.ncol//2+9, :] = 1
            else:
                masks[:, self.nrow//2-9:self.nrow//2+9, self.ncol//2-9:self.ncol//2+9, :-2] = 1
                masks[:, self.nrow//2-6:self.nrow//2+6, self.ncol//2-6:self.ncol//2+6, -2:] = 1
            # to complex data
            masks = torch.cat((masks, torch.zeros(masks.shape).to('cuda')),-1) # (necho, nrow, ncol, 2)
            # add coil dimension
            masks = masks[None, ...] # (1, necho, nrow, ncol, 2)
            masks = torch.cat(self.ncoil*[masks]) # (ncoil, necho, nrow, ncol, 2)
            # add batch dimension
            masks = masks[None, ...] # (1, ncoil, necho, nrow, ncol, 2)
            return masks

    def rescalePmask(self, Pmask, samplingRatio):
        if self.flag_loupe == 1:
            xbar = torch.mean(Pmask)
            r = samplingRatio / xbar
            beta = (1-samplingRatio) / (1-xbar)
            # le = (r<=1).to('cuda', dtype=torch.float32)
            le = (r<=1).float()
            return le * Pmask * r + (1-le) * (1 - (1-Pmask) * beta)
        elif self.flag_loupe == 2 and self.flag_2D == 1:
            xbar = torch.mean(Pmask, dim=[1,2])
            r = samplingRatio / xbar
            beta = (1-samplingRatio) / (1-xbar)
            le = (r<=1).float()
            return le[:, None, None] * Pmask * r[:, None, None] + (1-le[:, None, None]) * (1 - (1-Pmask) * beta[:, None, None])
        elif self.flag_loupe == 2 and self.flag_2D == 0:
            xbar = torch.mean(Pmask, dim=1)
            r = samplingRatio / xbar
            beta = (1-samplingRatio) / (1-xbar)
            le = (r<=1).float()
            return le[:, None] * Pmask * r[:, None] + (1-le[:, None]) * (1 - (1-Pmask) * beta[:, None])

    def samplingPmask(self, Pmask_rescaled):
        if self.flag_2D == 1:
            if self.stochasticSampling:
                Mask = bernoulliSample.apply(Pmask_rescaled)
            else:
                thresh = torch.rand(Pmask_rescaled.shape).to('cuda')
                Mask = 1/(1+torch.exp(-12*(Pmask_rescaled-thresh)))
        else:
            if self.stochasticSampling:
                Mask1D = bernoulliSample.apply(Pmask_rescaled)
                if self.flag_loupe == 1:
                    Mask = Mask1D[..., None].repeat(1, self.ncol)
                else:
                    Mask = Mask1D[..., None].repeat(1, 1, self.ncol)
            else:
                thresh = torch.rand(Pmask_rescaled.shape).to('cuda')
                Mask1D = 1/(1+torch.exp(-12*(Pmask_rescaled-thresh)))
                if self.flag_loupe == 1:
                    Mask = Mask1D[..., None].repeat(1, self.ncol)
                else:
                    Mask = Mask1D[..., None].repeat(1, 1, self.ncol)
        return Mask

    def forward(self, kdatas, csms, csm_lowres, masks, flip, test=False, x_input=None):
        # generate sampling mask
        if self.flag_loupe == 1:
            masks = self.generateMask(self.weight_parameters)
            self.Pmask = 1 / (1 + torch.exp(-self.slope * self.weight_parameters))
            self.Mask = masks[0, 0, 0, :, :, 0]
        elif self.flag_loupe == 2:
            masks = self.generateMask(self.weight_parameters)
            self.Pmask = 1 / (1 + torch.exp(-self.slope * self.weight_parameters))
            self.Mask = masks[0, 0, :, :, :, 0].permute(1, 2, 0)
        else:
            self.Mask = masks[0, 0, 0, :, :, 0]
        # input
        if self.flag_dataset:
            x = backward_multiEcho(kdatas, csms, masks, flip, self.echo_cat, self.necho)
            # if self.rank:
            #     x = backward_multiEcho_compressor(kdatas, csms, masks, flip, self.U, self.rank,
            #                                       self.flag_compressor, self.echo_cat, self.necho)
        else:
            x = backward_MS(kdatas, csms, masks, flip, self.echo_cat, self.necho)
        x_start = x
        
        if self.echo_cat == 0:
            x_start_ = torch_channel_concate(x_start, self.necho)
        else:
            x_start_ = x_start

        if x_input is not None:
            x = x_input
            self.necho_pred = self.necho_pred_value
            # print('Echo prediction')
        else:
            self.necho_pred = 0
            # print('No prediction')

        # generate preconditioner
        if self.flag_precond == 1:
            precond = 3 / (1 + torch.exp(-0.1 * self.preconditioner(x_start_))) + 1
            precond = torch_channel_deconcate(precond)
            # precond[:, 1, ...] = 0
            self.precond = precond
        else:
            self.precond = 0

        # Deep Quasi-newton
        if self.flag_solver == 0:
            if self.flag_BCRNN == 0:
                if self.flag_dataset:
                    A = Back_forward_multiEcho(csms, masks, flip, self.lambda_dll2, echo_cat=self.echo_cat, necho=self.necho)
                else:
                    A = Back_forward_MS(csms, masks, flip, self.lambda_dll2, echo_cat=self.echo_cat, necho=self.necho)
                Xs = []
                for i in range(self.K):
                    # if self.random:
                    #     mag = (1 + torch.randn(1)/3).to(device)
                    #     phase = (torch.rand(1) * 3.14/2 - 3.14/4).to(device)
                    #     factor = torch.cat((mag*torch.cos(phase), mag*torch.sin(phase)), 0)[None, :, None, None, None]

                    #     if self.echo_cat == 0:
                    #         x = mlpy_in_cg(x, factor)  # for echo_cat=0
                    #     elif self.echo_cat == 1:
                    #         x = torch_channel_concate(mlpy_in_cg(torch_channel_deconcate(x), factor))  # for echo_cat=1

                    # if i != self.K // 2:
                    #     x_block = self.resnet_block(x)
                    # else:
                    #     if self.att == 1:
                    #         x_block = self.attBlock(x)
                    #     else:
                    #         x_block = self.resnet_block(x)

                    x_block = self.resnet_block(x)
                    x_block1 = x - x_block

                    # if self.random:
                    #     factor = torch.cat((1/mag*torch.cos(phase), -1/mag*torch.sin(phase)), 0)[None, :, None, None, None]
                    #     if self.echo_cat == 0:
                    #         x = mlpy_in_cg(x, factor)  # for echo_cat=0
                    #     elif self.echo_cat == 1:
                    #         x = torch_channel_concate(mlpy_in_cg(torch_channel_deconcate(x), factor))  # for echo_cat=1

                    rhs = x_start + self.lambda_dll2*x_block1
                    dc_layer = DC_layer_multiEcho(A, rhs, echo_cat=self.echo_cat, necho=self.necho,
                                                  flag_precond=self.flag_precond, precond=self.precond)
                    x = dc_layer.CG_iter()

                    if self.echo_cat:
                        x = torch_channel_concate(x, self.necho)
                    Xs.append(x)
                return Xs
            elif self.flag_BCRNN > 0:
                x = torch_channel_deconcate(x).permute(0, 1, 3, 4, 2)  # (n, 2, nx, ny, n_seq)
                x = x.contiguous()
                net = {}
                n_batch, n_ch, width, height, n_seq = x.size()
                size_h = [n_seq*n_batch, self.nf, width, height]
                if test:
                    with torch.no_grad():
                        hid_init = Variable(torch.zeros(size_h)).cuda()
                else:
                    hid_init = Variable(torch.zeros(size_h)).cuda()
                for j in range(self.nd-1):
                    net['t0_x%d'%j]=hid_init
                if self.flag_dataset:
                    A = Back_forward_multiEcho(csms, masks, flip, self.lambda_dll2, echo_cat=self.echo_cat, necho=self.necho)
                else:
                    A = Back_forward_MS(csms, masks, flip, self.lambda_dll2, echo_cat=self.echo_cat, necho=self.necho)
                Xs = []
                for i in range(1, self.K+1):
                    # update auxiliary variable x0
                    x_ = x.permute(4, 0, 1, 2, 3) # (n_seq, n, 2, nx, ny)
                    x_ = x_.contiguous()
                    # net['t%d_x0'%(i-1)] = net['t%d_x0'%(i-1)].view(n_seq, n_batch, self.nf, width, height)
                    # net['t%d_x0'%i] = self.bcrnn(x_, net['t%d_x0'%(i-1)], test)
                    if x_.requires_grad and self.flag_cp:
                        net['t%d_x0'%i] = checkpoint(self.bcrnn, x_)
                    else:
                        net['t%d_x0'%i] = self.bcrnn(x_, test)
                    if self.flag_att == 1:
                        # net['t%d_x0'%i] = net['t%d_x0'%i].permute(1, 2, 0, 3, 4)  # (nt, 1, nf, nx, ny) to (1, nf, nt, nx, ny)
                        # net['t%d_x0'%i] = self.attBlock(net['t%d_x0'%i])
                        # net['t%d_x0'%i] = net['t%d_x0'%i].permute(2, 0, 1, 3, 4)  # (1, nf, nt, nx, ny) to (nt, 1, nf, nx, ny)
                        
                        net['t%d_x0'%i] = net['t%d_x0'%i].permute(1, 0, 2, 3, 4).view(n_batch, n_seq, self.nf, width*height)  # (nt, 1, nf, nx, ny) to (1, nt, nf, nx*ny)
                        net['t%d_x0'%i] = self.attBlock(net['t%d_x0'%i])
                        net['t%d_x0'%i].view(n_batch, n_seq, self.nf, width, height).permute(1, 0, 2, 3, 4)

                    net['t%d_x0'%i] = net['t%d_x0'%i].view(-1, self.nf, width, height)

                    net['t%d_x1'%i] = self.conv1_x(net['t%d_x0'%i])
                    net['t%d_x1'%i] = self.bn1_x(net['t%d_x1'%i])
                    # net['t%d_h1'%i] = self.conv1_h(net['t%d_x1'%(i-1)])
                    # net['t%d_x1'%i] = self.relu(net['t%d_h1'%i]+net['t%d_x1'%i])
                    net['t%d_x1'%i] = self.relu(net['t%d_x1'%i])

                    net['t%d_x2'%i] = self.conv2_x(net['t%d_x1'%i])
                    net['t%d_x2'%i] = self.bn2_x(net['t%d_x2'%i])
                    # net['t%d_h2'%i] = self.conv2_h(net['t%d_x2'%(i-1)])
                    # net['t%d_x2'%i] = self.relu(net['t%d_h2'%i]+net['t%d_x2'%i])
                    net['t%d_x2'%i] = self.relu(net['t%d_x2'%i])

                    net['t%d_x3'%i] = self.conv3_x(net['t%d_x2'%i])
                    net['t%d_x3'%i] = self.bn3_x(net['t%d_x3'%i])
                    # net['t%d_h3'%i] = self.conv3_h(net['t%d_x3'%(i-1)])
                    # net['t%d_x3'%i] = self.relu(net['t%d_h3'%i]+net['t%d_x3'%i])
                    net['t%d_x3'%i] = self.relu(net['t%d_x3'%i])

                    net['t%d_x4'%i] = self.conv4_x(net['t%d_x3'%i])

                    x_ = x_.view(-1, n_ch, width, height)
                    net['t%d_out'%i] = x_ - net['t%d_x4'%i]

                    # update x using CG block
                    x0 = net['t%d_out'%i] # (n_seq, 2, nx, ny)
                    x0_ = torch_channel_concate(x0[None, ...].permute(0, 2, 1, 3, 4), self.necho).contiguous()
                    rhs = x_start + self.lambda_dll2*x0_
                    dc_layer = DC_layer_multiEcho(A, rhs, echo_cat=self.echo_cat, necho=self.necho,
                                                  flag_precond=self.flag_precond, precond=self.precond)
                    x = dc_layer.CG_iter()
                    if self.echo_cat:
                        x = torch_channel_concate(x, self.necho)
                    if self.flag_temporal_pred:
                        x_last_echos = self.densenet(x)
                        Xs.append(torch.cat((x, x_last_echos), 1))
                    else:
                        Xs.append(x)
                    x = torch_channel_deconcate(x).permute(0, 2, 1, 3, 4).view(-1, n_ch, width, height) # (n_seq, 2, nx, ny)
                    x = x[None, ...].permute(0, 2, 3, 4, 1).contiguous()
                return Xs
        # Deep ADMM
        elif self.flag_solver == 1:
            if self.flag_BCRNN == 0:
                if self.flag_dataset:
                    A = Back_forward_multiEcho(csms, masks, flip, self.lambda_dll2, 
                                            self.lambda_lowrank, self.echo_cat, self.necho,
                                            kdata=kdatas, csm_lowres=csm_lowres, U=self.U, rank=self.rank)
                    # if self.rank:
                    #     A = Back_forward_multiEcho_compressor(csms, masks, flip, self.lambda_dll2, 
                    #                         self.echo_cat, self.necho, kdata=kdatas, 
                    #                         U=self.U, rank=self.rank, flag_compressor=self.flag_compressor)
                else:
                    A = Back_forward_MS(csms, masks, flip, self.lambda_dll2, 
                                        self.lambda_lowrank, self.echo_cat, self.necho,
                                        kdata=kdatas, csm_lowres=csm_lowres, rank=self.rank)
                Xs = []
                uk = torch.zeros(x_start.size()).to('cuda')
                for i in range(self.K):
                    # update auxiliary variable v
                    v_block = self.resnet_block(x+uk/self.lambda_dll2)
                    v_block1 = x + uk/self.lambda_dll2 - v_block
                    # update x using CG block
                    x0 = v_block1 - uk/self.lambda_dll2
                    rhs = x_start + self.lambda_dll2*x0
                    dc_layer = DC_layer_multiEcho(A, rhs, echo_cat=self.echo_cat, necho=self.necho,
                                                  flag_precond=self.flag_precond, precond=self.precond)
                    x = dc_layer.CG_iter()
                    if self.echo_cat:
                        x = torch_channel_concate(x, self.necho)
                    Xs.append(x)
                    # update dual variable uk
                    uk = uk + self.lambda_dll2*(x - v_block1)
                return Xs
            elif self.flag_BCRNN > 0:
                x = torch_channel_deconcate(x).permute(0, 1, 3, 4, 2)  # (n, 2, nx, ny, n_seq)
                x = x.contiguous()
                net = {}
                n_batch, n_ch, width, height, n_seq = x.size()
                n_seq = self.necho + self.necho_pred - 2  # exclude T1w and T2w contrasts
                size_h = [n_seq*n_batch, self.nf, width, height]
                if test:
                    with torch.no_grad():
                        hid_init = Variable(torch.zeros(size_h)).cuda()
                else:
                    hid_init = Variable(torch.zeros(size_h)).cuda()
                for j in range(self.nd-1):
                    net['t0_x%d'%j]=hid_init
                if self.flag_dataset:
                    A = Back_forward_multiEcho(csms, masks, flip, self.lambda_dll2, 
                                            self.lambda_lowrank, self.echo_cat, self.necho,
                                            kdata=kdatas, csm_lowres=csm_lowres, U=self.U, rank=self.rank)
                else:
                    A = Back_forward_MS(csms, masks, flip, self.lambda_dll2, 
                                        self.lambda_lowrank, self.echo_cat, self.necho,
                                        kdata=kdatas, csm_lowres=csm_lowres, rank=self.rank)
                Xs = []
                uk = torch.zeros(n_batch, n_ch, width, height, n_seq+2).to('cuda')
                for i in range(1, self.K+1):
                    if i <= self.split_K[0]:
                        x_start = x_start.to('cuda:0')
                        x = x.to('cuda:0')
                        uk = uk.to('cuda:0')
                        # self.lambda_dll2 = self.lambda_dll2.to('cuda:0')
                        self.bcrnn.to('cuda:0')
                        self.featureExtractor_t1t2.to('cuda:0')
                        self.denoiser.to('cuda:0')
                        self.denoiser_t1t2.to('cuda:0')
                    elif i <= self.split_K[1]:
                        x_start = x_start.to('cuda:1')
                        x = x.to('cuda:1')
                        uk = uk.to('cuda:1')
                        # self.lambda_dll2 = self.lambda_dll2.to('cuda:1')
                        self.bcrnn.to('cuda:1')
                        self.featureExtractor_t1t2.to('cuda:1')
                        self.denoiser.to('cuda:1')
                        self.denoiser_t1t2.to('cuda:1')
                    elif i <= self.split_K[2]:
                        x_start = x_start.to('cuda:2')
                        x = x.to('cuda:2')
                        uk = uk.to('cuda:2')
                        # self.lambda_dll2 = self.lambda_dll2.to('cuda:2')
                        self.bcrnn.to('cuda:2')
                        self.featureExtractor_t1t2.to('cuda:2')
                        self.denoiser.to('cuda:2')
                        self.denoiser_t1t2.to('cuda:2')
                    # update auxiliary variable v
                    x_ = (x+uk/self.lambda_dll2).permute(4, 0, 1, 2, 3) # (n_seq+2, n, 2, nx, ny)
                    x_ = x_.contiguous()
                    if x_.requires_grad and self.flag_cp:
                        # net['t%d_x0'%i] = checkpoint(self.bcrnn, x_[2:, ...])
                        if self.flag_t1w_only == 0:
                            net['t%d_x0'%i] = checkpoint(self.bcrnn, x_[:-2, ...])
                        else:
                            net['t%d_x0'%i] = checkpoint(self.bcrnn, x_[:-1, ...])
                    else:
                        # net['t%d_x0'%i] = self.bcrnn(x_[2:, ...], test)
                        if self.flag_t1w_only == 0:
                            net['t%d_x0'%i] = self.bcrnn(x_[:-2, ...], test)
                        else:
                            net['t%d_x0'%i] = self.bcrnn(x_[:-1, ...], test)
                    # net['t%d_t1t2_0'%i] = self.featureExtractor_t1t2(torch.cat((x_[0, ...], x_[1, ...]), 1), None)
                    if self.flag_t1w_only == 0:
                        net['t%d_t1t2_0'%i] = self.featureExtractor_t1t2(torch.cat((x_[-2, ...], x_[-1, ...]), 1), None)
                    else:
                        net['t%d_t1t2_0'%i] = self.featureExtractor_t1t2(x_[-1, ...], None)

                    net['t%d_x0'%i] = net['t%d_x0'%i].view(-1, self.nf, width, height)
                    net['t%d_t1t2_0'%i] = net['t%d_t1t2_0'%i].view(-1, self.nf, width, height)

                    if self.flag_unet == 1:
                        if x_.requires_grad and self.flag_cp:
                            if self.flag_mc_fusion == 1:
                                # concatenate t1t2 features to all multi-echo recurrent features
                                net['t%d_x4'%i] = checkpoint(self.denoiser, net['t%d_x0'%i] + net['t%d_t1t2_0'%i])
                                # concatenate 1st echo features to t1t2 features
                                net['t%d_t1t2_4'%i] = checkpoint(self.denoiser_t1t2, net['t%d_x0'%i][0:1, ...] + net['t%d_t1t2_0'%i])
                            else:
                                net['t%d_x4'%i] = checkpoint(self.denoiser, net['t%d_x0'%i])
                                net['t%d_t1t2_4'%i] = checkpoint(self.denoiser_t1t2, net['t%d_t1t2_0'%i])
                        else:
                            if self.flag_mc_fusion == 1:
                                net['t%d_x4'%i] = self.denoiser(net['t%d_x0'%i] + net['t%d_t1t2_0'%i])
                                net['t%d_t1t2_4'%i] = self.denoiser_t1t2(net['t%d_x0'%i][0:1, ...] + net['t%d_t1t2_0'%i])
                            else:
                                net['t%d_x4'%i] = self.denoiser(net['t%d_x0'%i])
                                net['t%d_t1t2_4'%i] = self.denoiser_t1t2(net['t%d_t1t2_0'%i])
                        if self.flag_t1w_only == 0:
                            # concatenate denoised t1w and t2w
                            net['t%d_t1t2_4'%i] = torch.cat((net['t%d_t1t2_4'%i][:, 0:2], net['t%d_t1t2_4'%i][:, 2:4]), 0)
                    
                    x_ = x_.view(-1, n_ch, width, height)
                    # net['t%d_out'%i] = x_ - torch.cat((net['t%d_t1t2_4'%i], net['t%d_x4'%i]), 0)
                    net['t%d_out'%i] = x_ - torch.cat((net['t%d_x4'%i], net['t%d_t1t2_4'%i]), 0)
 
                    # update x using CG block
                    uk_ = uk.permute(4, 0, 1, 2, 3).view(-1, n_ch, width, height)
                    x0 = net['t%d_out'%i] - uk_/self.lambda_dll2  # (n_seq, 2, nx, ny)
                    x0_ = torch_channel_concate(x0[None, ...].permute(0, 2, 1, 3, 4), self.necho+self.necho_pred).contiguous()
                    rhs = x_start + self.lambda_dll2*x0_[:, :self.necho*2, ...]
                    dc_layer = DC_layer_multiEcho(A, rhs, echo_cat=self.echo_cat, necho=self.necho,
                                                  flag_precond=self.flag_precond, precond=self.precond)
                    x = dc_layer.CG_iter()
                    if self.necho_pred > 0:
                        x = torch.cat((x, net['t%d_out'%i][None, self.necho:, ...].permute(0, 2, 1, 3, 4)), dim=2)
                    if self.echo_cat:
                        x = torch_channel_concate(x, self.necho+self.necho_pred)
                    if self.flag_temporal_pred:
                        x_last_echos = self.densenet(x)
                        Xs.append(torch.cat((x, x_last_echos), 1))
                    else:
                        Xs.append(x)
                        # Xs.append(x0_)
                        # Xs.append(torch_channel_concate(net['t%d_out'%i][None, ...].permute(0, 2, 1, 3, 4), self.necho+self.necho_pred))
                    x = torch_channel_deconcate(x).permute(0, 2, 1, 3, 4).view(-1, n_ch, width, height) # (n_seq, 2, nx, ny)
                    # update dual variable uk
                    uk = uk_ + self.lambda_dll2*(x - net['t%d_out'%i])

                    x = x[None, ...].permute(0, 2, 3, 4, 1).contiguous()
                    uk = uk[None, ...].permute(0, 2, 3, 4, 1).contiguous()
                return Xs

        # TV Quasi-newton
        elif self.flag_solver == 2:
            A = Back_forward_multiEcho(csms, masks, flip, 
                                    self.lambda_dll2, self.echo_cat)
            Xs = []
            for i in range(self.K):
                rhs = x_start - A.AtA(x, use_dll2=3)
                dc_layer = DC_layer_multiEcho(A, rhs, echo_cat=self.echo_cat,
                    flag_precond=self.flag_precond, precond=self.precond, use_dll2=3)
                delta_x = dc_layer.CG_iter(max_iter=10)
                if self.echo_cat:
                    delta_x = torch_channel_concate(delta_x)
                x = x + delta_x
                Xs.append(x)
            return Xs

        # TV ADMM
        elif self.flag_solver == 3:
            A = Back_forward_multiEcho(csms, masks, flip, 
                                    self.rho_penalty, self.echo_cat)
            Xs = []
            wk = torch.zeros(x_start.size()+(2,)).to('cuda')
            etak = torch.zeros(x_start.size()+(2,)).to('cuda')
            zeros_ = torch.zeros(x_start.size()+(2,)).to('cuda')
            for i in range(self.K):
                # update auxiliary variable wk through threshold
                ek = gradient(x) + etak/self.rho_penalty
                wk = ek.sign() * torch.max(torch.abs(ek) - self.lambda_tv/self.rho_penalty, zeros_)

                x_old = x
                # update x using CG block
                rhs = x_start + self.rho_penalty*divergence(wk) - divergence(etak)
                dc_layer = DC_layer_multiEcho(A, rhs, echo_cat=self.echo_cat,
                            flag_precond=self.flag_precond, precond=self.precond, use_dll2=2)
                x = dc_layer.CG_iter(max_iter=10)
                if self.echo_cat:
                    x = torch_channel_concate(x)
                Xs.append(x)
                # update dual variable etak
                etak = etak + self.rho_penalty * (gradient(x) - wk)
            return Xs


class DenseBlock(nn.Module):
    def __init__(
        self,
        input_channels,  # input_echos*2
        output_channels,  # output_echos*2
        filter_channels=32  # channel after each conv
    ):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, filter_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(input_channels+filter_channels, filter_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(input_channels+2*filter_channels, filter_channels, 3, padding=1)
        self.conv4 = nn.Conv2d(input_channels+3*filter_channels, filter_channels, 3, padding=1)
        self.conv5 = nn.Conv2d(input_channels+4*filter_channels, filter_channels, 3, padding=1)
        self.conv_final = nn.Conv2d(filter_channels, output_channels, 1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x0):
        x1 = self.relu(self.conv1(x0))
        x2 = self.relu(self.conv2(torch.cat((x0, x1), 1)))
        x3 = self.relu(self.conv3(torch.cat((x0, x1, x2), 1)))
        x4 = self.relu(self.conv4(torch.cat((x0, x1, x2, x3), 1)))
        x5 = self.relu(self.conv5(torch.cat((x0, x1, x2, x3, x4), 1)))
        return self.conv_final(x5)
