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

    rootName = '/data/Jinwei/Multi_echo_kspace'
    subject_IDs_test = ['MS1']
    lambda_dll2 = np.array([1e-6, 5e-2, 1e-6, 5e-2])
    batch_size = 1
    K = 9

    dataLoader = MultiEchoSimu(rootDir=rootName+'/dataset', 
        subject_IDs=subject_IDs_test, 
        num_echos=num_echos,
        flag_train=0)
    testLoader = data.DataLoader(dataLoader, batch_size=batch_size, shuffle=False)

    para_means = np.load(rootName+'/weights/parameters_means_{0}.npy'.format(num_echos))
    para_stds = np.load(rootName+'/weights/parameters_stds_{0}.npy'.format(num_echos))

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
    # print(netG_dc)
    netG_dc.to(device)
    weights_dict = torch.load(rootName+'/weights/weight_{0}_model={1}.pt'.format(num_echos, opt['model']))
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
            paras, paras_prior = netG_dc(inputs, iField, para_means, para_stds)

            if opt['model'] != 0:
                Paras.append(paras[-1].cpu().detach())
            elif opt['model'] == 0:
                Paras.append(paras.cpu().detach())

    Paras = np.squeeze(np.concatenate(Paras, axis=0))
    Paras = np.transpose(Paras, [2,3,0,1])
    adict = {}
    adict['paras'] = Paras
    sio.savemat(rootName+'/Paras_{0}_model={1}.mat'.format(num_echos, opt['model']), adict)