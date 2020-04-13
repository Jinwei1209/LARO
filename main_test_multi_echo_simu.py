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
    num_echos = 6
    lambda_dll2 = np.array([0.01, 0.01, 0.01, 0.01])
    gd_stepsize = np.array([0.001])
    batch_size = 1
    niter = 500
    epoch = 0
    lrG_dc = 1e-3

    dataLoader = MultiEchoSimu(rootDir=rootName+'/dataset', subject_IDs=subject_IDs)
    trainLoader = data.DataLoader(dataLoader, batch_size=batch_size, shuffle=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # network
    netG_dc = MultiEchoDC(
        input1_channels=num_echos*2, 
        filter1_channels=32,
        output1_channels=4,
        filter2_channels=32,
        lambda_dll2=lambda_dll2,
        gd_stepsize=gd_stepsize,
        K=1
    )
    # print(netG_dc)
    netG_dc.to(device)

    # optimizer
    optimizerG_dc = optim.Adam(netG_dc.parameters(), lr=lrG_dc, betas=(0.9, 0.999))
    # loss
    lossl1 = lossL1()

    while epoch < niter:
        epoch += 1 

        # training phase
        netG_dc.train()
        for idx, (target, brain_mask, mask, csm, kdata) in enumerate(trainLoader):
            target = target.to(device)
            mask = mask.to(device)
            csm = csm.to(device)
            kdata = kdata.to(device)
            # forward
            init_params, final_params = netG_dc(mask, csm, kdata)
            # stochastic gradient descnet
            optimizerG_dc.zero_grad()
            loss_total = lossl1(init_params, target) + lossl1(final_params, target)
            loss_total.backward()
            optimizerG_dc.step()
            print('Loss = {0}'.format(loss_total.item()))

        torch.save(netG_dc.state_dict(), rootName+'/weights/weight.pt')