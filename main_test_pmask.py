import os
import time
import torch
import numpy as np
import math

from torch.utils import data
from loader.kdata_loader_GE import kdata_loader_GE
from models.unet import Unet
from models.initialization import *
from models.dc_blocks import *
from models.unet_with_dc import *
from models.resnet_with_dc import *
from models.dc_with_prop_mask import *
from models.dc_with_straight_through_pmask import *     
from utils.test import *
from utils.data import *
from utils.test import *

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    K = 2
    K_model = 2
    straight_through = True
    samplingRatio = 0.1
    use_uncertainty = False
    passSigmoid = False  # +/-
    fixed_mask = False  # +/-
    optimal_mask = True  # +/-
    rescale = True
    lambda_Pmask = 0  
    lambda_dll2 = 0.01
    batch_size = 1
    folderName = '{0}_rolls'.format(K)
    contrast = 'T1'
    rootName = '/data/Jinwei/{}_slice_recon_GE'.format(contrast)

    dataLoader_test = kdata_loader_GE(
        rootDir=rootName,
        contrast=contrast, 
        split='test'
        )
    testLoader = data.DataLoader(dataLoader_test, batch_size=batch_size, shuffle=False)

    if not straight_through:
        netG_dc = DC_with_Prop_Mask(
            input_channels=2, 
            filter_channels=32, 
            lambda_dll2=lambda_dll2, 
            K=K,
            unc_map=use_uncertainty,
            fixed_mask=fixed_mask,
            rescale=rescale
        )
        netG_dc.to(device)
        netG_dc.load_state_dict(torch.load(rootName+'/'+folderName+'/weights/L1_mask'+
            '/weights_lambda_pmask=0.015_optimal.pt'))
        netG_dc.eval()
        print('Loading LOUPE weights')
    else:
        netG_dc = DC_with_Straight_Through_Pmask(
            input_channels=2, 
            filter_channels=32, 
            lambda_dll2=lambda_dll2,
            K=K_model, 
            unc_map=use_uncertainty,
            fixed_mask=fixed_mask,
            optimal_mask=optimal_mask,
            passSigmoid=passSigmoid,
            rescale=rescale,
            samplingRatio=samplingRatio,
            contrast=contrast
        )
        netG_dc.to(device)
        if optimal_mask:
            print('K=10')
            netG_dc.load_state_dict(torch.load(rootName+'/'+folderName+'/weights/{}'.format(math.floor(samplingRatio*100))+
                '/weights_ratio_pmask={}%_optimal_ST.pt'.format(math.floor(samplingRatio*100))))
        else:
            print('heihei \n')
            netG_dc.load_state_dict(torch.load(rootName+'/'+folderName+'/weights/{}'.format(math.floor(samplingRatio*100))+
                '/weights_ratio_pmask={}%_optimal_ST_vd_K=8.pt'.format(math.floor(samplingRatio*100))))
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
            sio.savemat(rootName+'/'+folderName+
                        '/Optimal_mask_{}.mat'.format(math.floor(samplingRatio*100)), adict)

    print(np.mean(np.asarray(metrices_test.PSNRs)))
    Recons = np.concatenate(Recons, axis=0)
    Recons = np.transpose(Recons, [2,3,0,1])
    Recons = np.squeeze(np.sqrt(Recons[..., 0]**2 + Recons[..., 1]**2))

    adict = {}
    adict['Recons'] = Recons
    sio.savemat(rootName+'/'+folderName+
                '/Recons_{}.mat'.format(math.floor(samplingRatio*100)), adict)



