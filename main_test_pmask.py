import os
import time
import torch
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt

from torch.utils import data
from loader.kdata_loader_GE import kdata_loader_GE
from loader.real_and_kdata_loader import real_and_kdata_loader
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
from utils.operators import *

if __name__ == '__main__':

    lrG_dc = 1e-3
    niter = 1000
    batch_size = 1
    display_iters = 10
    lambda_dll2 = 1e-4  # 1e-4 as default, 1e-5 best
    lambda_tv = 1e-4  # 1e-4 as default
    rho_penalty = lambda_tv*100  # 2 as previous default, 100 current best
    use_uncertainty = False
    passSigmoid = False
    fixed_mask = False  # +/-
    optimal_mask = False  # +/-
    rescale = True
    
    # typein parameters
    parser = argparse.ArgumentParser(description='LOUPE-ST')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--weight_dir', type=str, default='weights_new')
    parser.add_argument('--flag_ND', type=int, default=3)
    parser.add_argument('--flag_solver', type=int, default=0)
    parser.add_argument('--flag_TV', type=int, default=1)
    parser.add_argument('--contrast', type=str, default='T2')
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--samplingRatio', type=float, default=0.1) # 0.1/0.2
    parser.add_argument('--flag_fix', type=int, default=0)  # 0 not fix, 1 LOUPE, 2 VD, 3 Uniform
    opt = {**vars(parser.parse_args())}

    if opt['flag_fix'] == 1:
        result_dir = '/results_test/results_loupe'
        opt['weight_dir'] = 'weights_loupe'
    elif opt['flag_fix'] == 2:
        result_dir = '/results_test/results_vd'
        opt['weight_dir'] = 'weights_vd'
    elif opt['flag_fix'] == 3:
        result_dir = '/results_test/results_uniform'
        opt['weight_dir'] = 'weights_uniform'
    else:
        result_dir = '/results'

    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_id']
    rootName = '/data/Jinwei/{}_slice_recon_GE'.format(opt['contrast'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataLoader_test = kdata_loader_GE(
        rootDir=rootName,
        contrast=opt['contrast'], 
        split='test'
        )
    # dataLoader_test = real_and_kdata_loader(
    #     rootDir='/data/Jinwei/T2_slice_recon_GE/',
    #     contrast=opt['contrast'], 
    #     split='test'
    #     )
    testLoader = data.DataLoader(dataLoader_test, batch_size=batch_size, shuffle=False)

    if opt['flag_solver'] == 0 or -1:
        K = opt['K']
    else:
        K = opt['K'] + 10

    netG_dc = DC_ST_Pmask(
        input_channels=2, 
        filter_channels=32, 
        lambda_dll2=lambda_dll2,
        lambda_tv=lambda_tv,
        rho_penalty=rho_penalty,
        flag_ND=opt['flag_ND'],
        flag_solver=opt['flag_solver'],
        flag_TV=opt['flag_TV'],
        K=K,  # +10 for TV
        unc_map=use_uncertainty,
        passSigmoid=passSigmoid,
        rescale=rescale,
        samplingRatio=opt['samplingRatio'],
        nrow=256,
        ncol=192,
        ncoil=32,
        flag_fix=opt['flag_fix']
    )
    netG_dc.to(device)
    # print(netG_dc)
    weights_dict = torch.load(rootName+'/{0}/Solver={1}_K={2}_flag_ND={3}_ratio={4}.pt'.format(
                              opt['weight_dir'], opt['flag_solver'], opt['K'], opt['flag_ND'], opt['samplingRatio']))
    if opt['flag_solver'] == -3:
        weights_dict['lambda_dll2'] = (torch.ones(1)*lambda_dll2).to(device)
        print('Lambda_dll2={0}'.format(netG_dc.lambda_dll2))
    elif opt['flag_solver'] == 2:
        weights_dict['lambda_tv'] = (torch.ones(1)*lambda_tv).to(device)
        weights_dict['rho_penalty'] = (torch.ones(1)*rho_penalty).to(device)
        print('Lambda_tv={0}'.format(netG_dc.lambda_tv))
        print('Rho_penalty={0}'.format(netG_dc.rho_penalty)) 
    netG_dc.load_state_dict(weights_dict)
    netG_dc.eval()

    metrices_test = Metrices()

    Recons = []
    for idx, (inputs, targets, csms, brain_masks) in enumerate(testLoader):
        # if idx % 10 == 0:
        print(idx)
        inputs = inputs.to(device)
        targets = targets.to(device)
        csms = csms.to(device)
        # calculating metrices
        Xs = netG_dc(inputs, csms)
        Recons.append(Xs[-1].cpu().detach())
        metrices_test.get_metrices(Xs[-1], targets)
        # if idx == 0:
        #     print('Pmask: {}, \n'.format(torch.mean(netG_dc.Pmask)))
        #     print('Sampling Raito : {}, \n'.format(torch.mean(netG_dc.masks)))
        #     adict = {}
        #     Mask = np.squeeze(np.asarray(netG_dc.masks.cpu().detach()))
        #     Mask[netG_dc.nrow//2-14:netG_dc.nrow//2+13, netG_dc.ncol//2-14:netG_dc.ncol//2+13] = 1
        #     adict['Mask'] = Mask
        #     sio.savemat(rootName+'/results/Mask_Solver={0}_K={1}_flag_ND={2}_ratio={3}.mat'.format(
        #                 opt['flag_solver'], opt['K'], opt['flag_ND'], opt['samplingRatio']), adict)

        #     adict = {}
        #     Pmask = np.squeeze(np.asarray(netG_dc.Pmask.cpu().detach()))
        #     Pmask = Pmask * opt['samplingRatio'] / np.mean(Pmask)
        #     Pmask[netG_dc.nrow//2-14:netG_dc.nrow//2+13, netG_dc.ncol//2-14:netG_dc.ncol//2+13] = np.amax(Pmask)
        #     adict['Pmask'] = Pmask
        #     sio.savemat(rootName+'/results/Pmask_Solver={0}_K={1}_flag_ND={2}_ratio={3}.mat'.format(
        #                 opt['flag_solver'], opt['K'], opt['flag_ND'], opt['samplingRatio']), adict)

    print(np.mean(np.asarray(metrices_test.PSNRs)))
    Recons = np.concatenate(Recons, axis=0)
    Recons = np.transpose(Recons, [2,3,0,1])
    Recons = np.squeeze(np.sqrt(Recons[..., 0]**2 + Recons[..., 1]**2))

    adict = {}
    adict['Recons'] = Recons
    sio.savemat(rootName+result_dir+'/Recons_Solver={0}_K={1}_flag_ND={2}_ratio={3}.mat'.format(
                opt['flag_solver'], opt['K'], opt['flag_ND'], opt['samplingRatio']), adict)

    # # Display the output images
    # # plot= lambda x: plt.imshow(x,cmap=plt.cm.gray, clim=(0.0, np.amax(Pmask)))
    # plot= lambda x: plt.imshow(x,cmap=plt.cm.gray, clim=(0.0, 0.371))
    # plt.clf()
    # plot(Pmask)
    # plt.axis('off')
    # plt.title('Solver = {0}, TV = {1}'.format(opt['flag_solver'], opt['flag_TV']))
    # plt.show()



