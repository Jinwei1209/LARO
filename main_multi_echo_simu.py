"""
    Experiment on parameters denoising using the first several echos instead of all echos
"""

# PYTHON_ARGCOMPLETE_OK
import os
import time
import torch
import math
import argparse
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
from models.dc_multi_echo2 import *
from utils.test import *
from utils.operators import OperatorsMultiEcho

if __name__ == '__main__':
    # typein parameters
    parser = argparse.ArgumentParser(description='LOUPE-ST')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--num_echos', type=int, default=5)
    # 0: Unet, 1: unrolled unet, 2: unrolled resnet, -1: progressive resnet 
    parser.add_argument('--model', type=int, default=1)
    opt = {**vars(parser.parse_args())}

    num_echos = opt['num_echos']
    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_id']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    t0 = time.time()
    epoch = 0
    niter = 100
    batch_size = 1
    K = 9
    lambda_dll2 = np.array([1e-6, 5e-2, 1e-6, 5e-2])
    gen_iterations = 0
    display_iters = 10
    lrG_dc = 1e-3

    rootName = '/data/Jinwei/Multi_echo_kspace'
    subject_IDs_train = ['MS2', 'MS3', 'MS4', 'MS5', 'MS6']
    # subject_IDs_train = ['MS2']
    subject_IDs_val = ['MS7']

    dataLoader = MultiEchoSimu(
        rootDir=rootName+'/dataset', 
        subject_IDs=subject_IDs_train, 
        num_echos=num_echos,
        flag_train=1
    )
    num_samples = dataLoader.num_samples
    trainLoader = data.DataLoader(dataLoader, batch_size=batch_size, shuffle=True)
    para_means, para_stds = dataLoader.parameters_means, dataLoader.parameters_stds
    np.save(rootName+'/weights/parameters_means_{0}.npy'.format(num_echos), para_means)
    np.save(rootName+'/weights/parameters_stds_{0}.npy'.format(num_echos), para_stds)

    dataLoader = MultiEchoSimu(
        rootDir=rootName+'/dataset', 
        subject_IDs=subject_IDs_val, 
        num_echos=num_echos,
        flag_train=0
    )
    valLoader = data.DataLoader(dataLoader, batch_size=batch_size, shuffle=False)

    # network
    if opt['model'] >= 0:
        netG_dc = MultiEchoDC(
            filter_channels=32,
            num_echos=num_echos,
            lambda_dll2=lambda_dll2,
            K=K,
            flag_model=opt['model']
        )
    else:
        netG_dc = MultiEchoPrg(
            filter_channels=32,
            num_echos=num_echos,
            lambda_dll2=lambda_dll2,
            K=K,
            flag_model=opt['model']
        )
    print(netG_dc)
    netG_dc.to(device)

    # optimizer
    optimizerG_dc = optim.Adam(netG_dc.parameters(), lr=lrG_dc, betas=(0.9, 0.999))
    # loss
    lossl1 = lossL1()
    Validation_loss = []

    while epoch < niter:
        epoch += 1 

        # training phase
        netG_dc.train()
        for idx, (targets, brain_mask, iField, inputs) in enumerate(trainLoader):
            with autograd.detect_anomaly():
                brain_mask = brain_mask.to(device)
                inputs = inputs.to(device) * brain_mask
                targets = targets.to(device) * brain_mask
                targets = torch.cat((targets[:, 1:2, ...], targets[:, 3:4, ...]), dim=1)
                brain_mask_iField = brain_mask[:, 0, None, None, :, :, None].repeat(1, 1, num_echos, 1, 1, 2)
                iField = iField.to(device).permute(0, 3, 4, 1, 2, 5) * brain_mask_iField
                # forward
                paras, paras_prior = netG_dc(inputs, iField, para_means, para_stds)
                # stochastic gradient descent
                optimizerG_dc.zero_grad()
                loss_total = 0
                if opt['model'] != 0:
                    for i in range(K):
                        loss_total += lossl1(paras[i], targets) + lossl1(paras_prior[i], targets)
                elif opt['model'] == 0:
                    loss_total = lossl1(paras, targets)
                loss_total.backward()
                optimizerG_dc.step()

                if gen_iterations%display_iters == 0:
                    print('epochs: [%d/%d], batchs: [%d/%d], time: %ds, Loss = %f'
                    % (epoch, niter, idx, num_samples//batch_size+1, time.time()-t0, loss_total.item()))
                    if epoch > 1:
                        print('Loss in Validation dataset is %f' % (Validation_loss[-1]))
                gen_iterations += 1

        # validation phase
        netG_dc.eval()
        loss_total_list = []
        with torch.no_grad():  # to solve memory exploration issue
            for idx, (targets, brain_mask, iField, inputs) in enumerate(valLoader):
                brain_mask = brain_mask.to(device)
                inputs = inputs.to(device) * brain_mask
                targets = targets.to(device) * brain_mask
                targets = torch.cat((targets[:, 1:2, ...], targets[:, 3:4, ...]), dim=1)
                brain_mask_iField = brain_mask[:, 0, None, None, :, :, None].repeat(1, 1, num_echos, 1, 1, 2)
                iField = iField.to(device).permute(0, 3, 4, 1, 2, 5) * brain_mask_iField
                # forward
                paras, paras_prior = netG_dc(inputs, iField, para_means, para_stds)
                loss_total = 0
                if opt['model'] != 0:
                    for i in range(K):
                        loss_total += lossl1(paras[i], targets) + lossl1(paras_prior[i], targets)
                elif opt['model'] == 0:
                    loss_total = lossl1(paras, targets)
                loss_total_list.append(np.asarray(loss_total.cpu().detach()))
            Validation_loss.append(sum(loss_total_list) / float(len(loss_total_list)))
        
        if Validation_loss[-1] == min(Validation_loss):
            torch.save(netG_dc.state_dict(), rootName+'/weights/weight_{0}_model={1}.pt'.format(num_echos, opt['model']))
