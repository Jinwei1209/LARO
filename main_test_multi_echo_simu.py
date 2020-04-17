# PYTHON_ARGCOMPLETE_OK
import os
import time
import torch
import math
import argparse
import argcomplete
import numpy as np

from torch.utils import data
from loader.multi_echo_simu_loader import MultiEchoSimu
from utils.data import *
from models.unet import Unet
from models.initialization import *
from models.discriminator import Basic_D
from utils.train import *
from IPython.display import clear_output
from utils.loss import *    
from models.dc_blocks import *
from models.unet_with_dc import *
from models.dc_with_prop_mask import *
from models.dc_with_straight_through_pmask import *
from models.dc_st_pmask import *
from models.dc_multi_echo import *
from utils.test import *
from utils.operators import OperatorsMultiEcho


if __name__ == '__main__':

    rootName = '/data/Jinwei/Multi_echo_kspace'
    subject_IDs = ['MS1']
    num_echos = 5
    lambda_dll2 = 0.01
    gd_stepsize = 0.1
    batch_size = 1
    K = 1
    niter = 500
    epoch = 0
    lrG_dc = 1e-3

    dataLoader = MultiEchoSimu(rootDir=rootName+'/dataset', subject_IDs=subject_IDs, num_echos=num_echos)
    trainLoader = data.DataLoader(dataLoader, batch_size=batch_size, shuffle=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # network
    netG_dc = MultiEchoDC(
        input1_channels=num_echos, 
        filter1_channels=32,
        filter2_channels=32,
        lambda_dll2=lambda_dll2,
        gd_stepsize=gd_stepsize,
        K=K
    )
    # print(netG_dc)
    netG_dc.to(device)
    weights_dict = torch.load(rootName+'/weights/weight.pt')
    netG_dc.load_state_dict(weights_dict)
    netG_dc.eval()

    with torch.no_grad():
        for idx, (target, brain_mask, mask, csm, kdata, mag, phase) in enumerate(trainLoader):
            print(idx)
            print(netG_dc.lambda_dll2)
            print(netG_dc.gd_stepsize)
            target = target.to(device)
            mask = mask.to(device)
            csm = csm.to(device)
            kdata = kdata.to(device)
            mag = mag.to(device)
            phase = phase.to(device)
            para_start, paras_prior, paras = netG_dc(mask, csm, kdata, mag, phase)

            if idx == 30:
                adict = {}
                adict['para_start'] = np.squeeze(np.asarray(para_start.cpu().detach()))
                sio.savemat('para_start.mat', adict)

                adict = {}
                adict['para_prior'] = np.squeeze(np.asarray(paras_prior[-1].cpu().detach()))
                sio.savemat('para_prior.mat', adict)

                adict = {}
                adict['para'] = np.squeeze(np.asarray(paras[-1].cpu().detach()))
                sio.savemat('para.mat', adict)

                break