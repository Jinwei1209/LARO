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
from utils.test import *
from utils.operators import OperatorsMultiEcho


if __name__ == '__main__':

    rootName = '/data/Jinwei/Multi_echo_kspace/dataset'
    subject_IDs = ['MS1']
    batch_size = 10
    niter = 1  # 500
    epoch = 0

    dataLoader = MultiEchoSimu(rootDir=rootName, subject_IDs=subject_IDs)
    trainLoader = data.DataLoader(dataLoader, batch_size=batch_size, shuffle=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    while epoch < niter:
        epoch += 1 

        # training phase
        for idx, (target, brain_mask, mask, csm, kdata) in enumerate(trainLoader):
            # print(target.shape)
            # print(brain_mask.shape)
            # print(mask.shape)
            # print(csm.shape)
            print(kdata.shape)

            mask = mask.to(device)
            csm = csm.to(device)
            M_0 = target[:, 0:1, ...].to(device)
            R_2 = target[:, 1:2, ...].to(device)
            phi_0 = target[:, 2:3, ...].to(device)
            f = target[:, 3:4, ...].to(device)
            lambda_dll2 = torch.tensor([0.001]).to(device)
            kdata = kdata.to(device)

            operators = OperatorsMultiEcho(mask, csm, M_0, R_2, phi_0, f, lambda_dll2)
            kdata_forward = operators.forward_operator()

            gradient_forward = operators.jacobian_conj(kdata_forward)
            gradient = operators.jacobian_conj(kdata)
            diff = gradient_forward - gradient

            img_forward = torch.ifft(kdata_forward, signal_ndim=2)
            img = torch.ifft(kdata, signal_ndim=2)

            if idx == 3:
                adict = {}
                adict['diff'] = np.squeeze(np.asarray(diff.cpu().detach()))
                sio.savemat('diff.mat', adict)

                break
