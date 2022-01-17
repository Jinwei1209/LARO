import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.autograd import Variable
from math import exp
from utils.data import *
from fits.fits import *


def lossL1():
    return nn.L1Loss()


def loss_classificaiton():
    return nn.BCELoss()


def lossL2():
    return nn.MSELoss()


class CrossEntropyMask(torch.nn.Module):
    def __init__(self, necho=10, nrow=206, ncol=80, radius=30):
        super(CrossEntropyMask, self).__init__()
        self.necho = necho
        self.nrow = nrow
        self.ncol = ncol
        self.non_calib = torch.tensor(self.gen_pattern(radius=radius)).to('cuda')


    def gen_pattern(self, num_row=206, num_col=80, radius=30):
        M_c = int(num_row/2)  # center of kspace
        N_c = int(num_col/2)  # center of kspace
        p_pattern = np.ones((num_row, num_col)).flatten()
        indices = np.array(np.meshgrid(range(num_row), range(num_col))).T.reshape(-1,2)  # row first
        distances = np.sqrt((indices[:, 0] - M_c)**2 * 0.5 + (indices[:, 1] - N_c)**2)
        distances_orders = distances // radius  # get the order of the distances
        p_pattern[distances_orders == 0] = 0
        p_pattern = p_pattern.reshape(num_row, num_col)
        return p_pattern


    def forward(self, pmask):
        loss = 0
        for i in range(self.necho):
            for j in range(self.necho):
                if j == i:
                    continue
                else:
                    a = torch.clamp(pmask[i][self.non_calib==1], min=1e-12, max=1-1e-12)
                    b = pmask[j][self.non_calib==1]
                    loss -= torch.sum(b*torch.log(a+1e-9)) / b.size()[0]
        return loss


class FittingError(torch.nn.Module):
    # not working
    def __init__(self, necho=10, nrow=206, ncol=80):
        super(FittingError, self).__init__()
        self.necho = necho
        self.nrow = nrow
        self.ncol = ncol
        self.loss = nn.MSELoss()

    def forward(self, x):
        x = torch_channel_deconcate(x)
        # compute parameters
        x_ = x.clone().detach().permute(0, 3, 4, 1, 2)
        x_pred = torch.zeros((1, 2, self.necho, self.nrow, self.ncol), requires_grad=True).to('cuda')
        with torch.no_grad():
            [_, water] = fit_R2_LM(x_)
            r2s = arlo(range(self.necho), torch.sqrt(x_[:, :, :, 0, :]**2 + x_[:, :, :, 1, :]**2))
            [p1, p0] = fit_complex(x_)
            water = water[0, ...].clone().detach()
            r2s = r2s[0, ...].clone().detach()
            p1 = p1[0, ...].clone().detach()
            p0 = p0[0, ...].clone().detach()
        for echo in range(self.necho):
            x_pred[0, 0, echo, :, :] = water * torch.exp(-r2s * echo) * torch.cos(p0 + p1*echo)
            x_pred[0, 1, echo, :, :] = water * torch.exp(-r2s * echo) * torch.sin(p0 + p1*echo)
        x_pred[torch.isnan(x_pred)] = 0
        x_pred[x_pred > 1] = 0
        loss = torch.mean((x - x_pred)**2)
        return loss

                    
            
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)


def snr_gain(r2s, te, weighting=0):
    '''
        SNR gain from multi-echo acquisition with or without weighted combination (not helping)
    '''
    tmp = torch.zeros(r2s.size()).to('cuda')

    if weighting == 0:
        N = len(te)
        for i in range(N):
            tmp += 1 / (te[i] * torch.exp(-r2s*te[i]))**2
        snr =  N * r2s / 0.37 / torch.sqrt(tmp)

    elif weighting == 1:
        N = len(te)
        for i in range(N):
            tmp += te[i] * torch.exp(-r2s*te[i])
        snr = tmp * r2s / 0.37 / np.sqrt(N)
    
    snr[torch.isnan(snr)] = 0
    snr[torch.isinf(snr)] = 0
    return snr.mean()

    

