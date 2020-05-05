# PYTHON_ARGCOMPLETE_OK
import os
import time
import torch
import math
import argparse
import argcomplete
import numpy as np

from torch.utils import data
from torch import autograd
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
    subject_IDs_train = ['MS1', 'MS2']
    num_echos = 3
    lambda_dll2 = 1
    gd_stepsize = 0.1
    batch_size = 1
    K = 1
    niter = 500
    epoch = 0
    lrG_dc = 1e-3

    dataLoader = MultiEchoSimu(rootDir=rootName+'/dataset', subject_IDs=subject_IDs_train, num_echos=num_echos)
    trainLoader = data.DataLoader(dataLoader, batch_size=batch_size, shuffle=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
    netG_dc.to(device)
    # weights_dict = torch.load(rootName+'/weights/weight.pt')
    # netG_dc.load_state_dict(weights_dict)

    # optimizer
    optimizerG_dc = optim.Adam(netG_dc.parameters(), lr=lrG_dc, betas=(0.9, 0.999))
    # loss
    lossl1 = lossL1()

    while epoch < niter:
        epoch += 1 

        # training phase
        netG_dc.train()
        for idx, (target, brain_mask, mask, csm, kdata, mag, phase) in enumerate(trainLoader):
            with autograd.detect_anomaly():
                print(idx)
                target = target.to(device)
                mask = mask.to(device)
                csm = csm.to(device)
                kdata = kdata.to(device)
                mag = mag.to(device)
                phase = phase.to(device)
                brain_mask = brain_mask.cuda()
                # forward
                para_start, paras_prior, paras = netG_dc(mask, csm, kdata, mag, phase)
                # stochastic gradient descnet
                optimizerG_dc.zero_grad()

                # for CG style pre-trian
                loss_total = lossl1(para_start, target*brain_mask)
                # # for Jacobian style pre-train
                # loss_total = lossl1(para_start, target) + lossl1(paras_prior[0], target)

                # further-train
                # loss_total = lossl1(para_start, target)
                # for i in range(K):
                #     loss_total = loss_total + lossl1(paras[i], target)

                loss_total.backward()
                optimizerG_dc.step()
                print('Loss = {0}'.format(loss_total.item()))
                print('Lambda = {0}'.format(netG_dc.lambda_dll2))
                # print('Step size = {0}'.format(netG_dc.gd_stepsize))

        torch.save(netG_dc.state_dict(), rootName+'/weights/weight.pt')
