import os
import time
import torch
import numpy as np
import math
import argparse

from torch.utils import data
from loader.kdata_loader_GE import kdata_loader_GE
from models.unet import Unet
from models.initialization import *
from models.dc_blocks import *
from models.unet_with_dc import *
from models.resnet_with_dc import *
from models.dc_with_prop_mask import *
from models.dc_with_straight_through_pmask import *     
from models.dc_st_pmask import *
from utils.test import *
from utils.data import *
from utils.test import *

if __name__ == '__main__':

    lrG_dc = 1e-3
    niter = 1000
    batch_size = 1
    display_iters = 10
    lambda_Pmask = 0
    lambda_dll2 = 0.01
    use_uncertainty = False
    passSigmoid = False
    fixed_mask = False  # +/-
    optimal_mask = False  # +/-
    rescale = True

    # typein parameters
    parser = argparse.ArgumentParser(description='LOUPE-ST')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--flag_ND', type=int, default=3)
    parser.add_argument('--contrast', type=str, default='T2')
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--samplingRatio', type=float, default=0.1) # 0.1/0.2
    opt = {**vars(parser.parse_args())}

    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_id']
    rootName = '/data/Jinwei/{}_slice_recon_GE'.format(opt['contrast'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataLoader_test = kdata_loader_GE(
        rootDir=rootName,
        contrast=opt['contrast'], 
        split='test'
        )
    testLoader = data.DataLoader(dataLoader_test, batch_size=batch_size, shuffle=False)

    netG_dc = DC_ST_Pmask(
        input_channels=2, 
        filter_channels=32, 
        lambda_dll2=lambda_dll2,
        flag_ND=opt['flag_ND'],
        K=opt['K'], 
        unc_map=use_uncertainty,
        passSigmoid=passSigmoid,
        rescale=rescale,
        samplingRatio=opt['samplingRatio']
    )
    netG_dc.to(device)
    netG_dc.load_state_dict(torch.load(rootName+'/weights/K={0}_flag_ND={1}_ratio={2}.pt'.format(
                                                opt['K'], opt['flag_ND'], opt['samplingRatio'])))
    netG_dc.eval()
    
    # print(netG_dc)
    print(netG_dc.lambda_dll2)
    metrices_test = Metrices()

    Recons = []
    for idx, (inputs, targets, csms, brain_masks) in enumerate(testLoader):
        print(idx)
        inputs = inputs.to(device)
        targets = targets.to(device)
        csms = csms.to(device)
        # calculating metrices
        Xs = netG_dc(inputs, csms)
        Recons.append(Xs[-1].cpu().detach())
        metrices_test.get_metrices(Xs[-1], targets)
        if idx == 0:
            print('Sampling Raito : {}, \n'.format(torch.mean(netG_dc.masks)))
            adict = {}
            adict['Mask'] = np.squeeze(np.asarray(netG_dc.masks.cpu().detach()))
            sio.savemat(rootName+'/results/Mask_K={0}_flag_ND={1}_ratio={2}.mat'.format(
                        opt['K'], opt['flag_ND'], opt['samplingRatio']), adict)

            adict = {}
            adict['Pmask'] = np.squeeze(np.asarray(netG_dc.Pmask.cpu().detach()))
            sio.savemat(rootName+'/results/Pmask_K={0}_flag_ND={1}_ratio={2}.mat'.format(
                        opt['K'], opt['flag_ND'], opt['samplingRatio']), adict)

    print(np.mean(np.asarray(metrices_test.PSNRs)))
    Recons = np.concatenate(Recons, axis=0)
    Recons = np.transpose(Recons, [2,3,0,1])
    Recons = np.squeeze(np.sqrt(Recons[..., 0]**2 + Recons[..., 1]**2))

    adict = {}
    adict['Recons'] = Recons
    sio.savemat(rootName+'/results/Recons_K={0}_flag_ND={1}_ratio={2}.mat'.format(
                opt['K'], opt['flag_ND'], opt['samplingRatio']), adict)



