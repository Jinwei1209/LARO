# PYTHON_ARGCOMPLETE_OK
import os
import time
import torch
import math
import argparse
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
from models.dc_multi_echo2 import *
from utils.test import *
from utils.operators import OperatorsMultiEcho


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rootName = '/data/Jinwei/Multi_echo_kspace'
    subject_IDs_test = ['MS1']
    num_echos = 3
    lambda_dll2 = np.array([1e-6, 5e-2, 1e-6, 5e-2])
    batch_size = 1
    K = 9

    dataLoader = MultiEchoSimu(rootDir=rootName+'/dataset', 
        subject_IDs=subject_IDs_test, 
        num_echos=num_echos,
        flag_train=0)
    testLoader = data.DataLoader(dataLoader, batch_size=batch_size, shuffle=False)

    para_means, para_stds = np.load(rootName+'/parameters_means.npy'), np.load(rootName+'/parameters_stds.npy')

    # network
    # netG_dc = MultiEchoDC2(
    #     filter_channels=32,
    #     num_echos=num_echos,
    #     lambda_dll2=lambda_dll2,
    #     norm_means=para_means,
    #     norm_stds=para_stds,
    #     K=K
    # )
    netG_dc = MultiEchoDC(
        filter_channels=32,
        num_echos=num_echos,
        lambda_dll2=lambda_dll2,
        norm_means=para_means,
        norm_stds=para_stds,
        K=K
    )
    # print(netG_dc)
    netG_dc.to(device)
    weights_dict = torch.load(rootName+'/weights/weight.pt')
    netG_dc.load_state_dict(weights_dict)
    netG_dc.eval()

    Paras = []
    with torch.no_grad():
        for idx, (targets, brain_mask, iField, inputs) in enumerate(testLoader):
            print(idx)
            brain_mask = brain_mask.to(device)
            inputs = inputs.to(device) * brain_mask
            targets = targets.to(device) * brain_mask
            brain_mask_iField = brain_mask[:, 0, None, None, :, :, None].repeat(1, 1, num_echos, 1, 1, 2)
            iField = iField.to(device).permute(0, 3, 4, 1, 2, 5) * brain_mask_iField
            paras, paras_prior = netG_dc(inputs, iField)

            Paras.append(paras[-1].cpu().detach())

    Paras = np.squeeze(np.concatenate(Paras, axis=0))
    Paras = np.transpose(Paras, [2,3,0,1])
    adict = {}
    adict['Paras'] = Paras
    sio.savemat(rootName+'/Paras.mat', adict)