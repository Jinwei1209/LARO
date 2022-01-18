"""
    Experiment on multi-echo kspace data reconstruction from GE scanner (0.75*0.75*1.5 mm^3)
"""
import os
import time
import torch
import math
import argparse
import scipy.io as sio
import numpy as np

from torch.optim.lr_scheduler import MultiStepLR
from IPython.display import clear_output
from torch.utils import data
from loader.kdata_multi_echo_CBIC import kdata_multi_echo_CBIC
from loader.kdata_multi_echo_CBIC_075 import kdata_multi_echo_CBIC_075
from utils.data import r2c, save_mat, readcfl, memory_pre_alloc, torch_channel_deconcate, torch_channel_concate, Logger, c2r_kdata
from utils.loss import lossL1, lossL2, SSIM, snr_gain, CrossEntropyMask, FittingError
from utils.test import Metrices
from utils.operators import backward_multiEcho
from models.resnet_with_dc import Resnet_with_DC2
from fits.fits import fit_R2_LM, arlo, fit_complex, fit_complex_all
from utils.operators import low_rank_approx

if __name__ == '__main__':

    lrG_dc = 1e-3
    batch_size = 1
    display_iters = 10
    gen_iterations = 1
    t0 = time.time()
    epoch = 0
    errL2_dc_sum = 0
    Validation_loss = []
    Validation_psnr = []
    ncoil = 8
    lambda_dll2 = 1e-3

    # typein parameters
    parser = argparse.ArgumentParser(description='Multi_echo_GE')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--mask', type=int, default=0)
    parser.add_argument('--res', type=int, default=0)  # 0: 0.75*0.75*1.5, 1: 1*1*2
    opt = {**vars(parser.parse_args())}

    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_id']
    # rootName = '/data2/Jinwei/QSM_raw_CBIC_075'  # GPU1
    rootName = '/data/Jinwei/QSM_raw_CBIC'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if opt['res'] == 0:
        nrow = 258
        ncol = 112
        nslice = 320
        necho = 7
    elif opt['res'] == 1:
        nrow = 206
        ncol = 80
        nslice = 200
        necho = 10

    if opt['mask'] == 0:
        masks = np.real(readcfl(rootName+'/masks2/mask_ellipt'))
    elif opt['mask'] == 1:
        masks = np.real(readcfl(rootName+'/masks2/mask_partialFourier_0.87'))
    elif opt['mask'] == 2:
        masks = np.real(readcfl(rootName+'/masks2/mask_partialFourier_0.75'))
    elif opt['mask'] == -1:
        masks = np.ones((nrow, ncol))

    # for 2D random sampling 
    masks = masks[..., np.newaxis] # (nrow, ncol, 1)

    masks = torch.tensor(masks, device=device).float()
    # to complex data
    masks = torch.cat((masks, torch.zeros(masks.shape).to(device)),-1) # (nrow, ncol, 2)
    # add echo dimension
    masks = masks[None, ...] # (1, nrow, ncol, 2)
    masks = torch.cat(necho*[masks]) # (necho, nrow, ncol, 2)
    # add coil dimension
    masks = masks[None, ...] # (1, necho, nrow, ncol, 2)
    masks = torch.cat(ncoil*[masks]) # (ncoil, necho, nrow, ncol, 2)
    # add batch dimension
    masks = masks[None, ...] # (1, ncoil, necho, nrow, ncol, 2)
 
    # flip matrix
    flip = torch.ones([necho, nrow, ncol, 1]) 
    flip = torch.cat((flip, torch.zeros(flip.shape)), -1).to(device)
    flip[:, ::2, ...] = - flip[:, ::2, ...] 
    flip[:, :, ::2, ...] = - flip[:, :, ::2, ...]
    # add batch dimension
    flip = flip[None, ...] # (1, necho, nrow, ncol, 2)

    Recons = []
    
    if opt['res'] == 0:
        dataLoader_test = kdata_multi_echo_CBIC_075(
            rootDir=rootName,
            contrast='MultiEcho', 
            split='test',
            subject=0,
            normalization=0,
            echo_cat=1
        )
    elif opt['res'] == 1:
        dataLoader_test = kdata_multi_echo_CBIC(
            rootDir=rootName,
            contrast='MultiEcho', 
            split='test',
            subject=0,
            normalization=0,
            echo_cat=1
        )
    testLoader = data.DataLoader(dataLoader_test, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        if opt['res'] == 0:
            for idx, (kdatas, targets, csms, brain_masks) in enumerate(testLoader):
                kdatas = kdatas.to(device)
                targets = targets.to(device)
                csms = csms.to(device)
                
                inputs = backward_multiEcho(kdatas, csms, masks, flip, echo_cat=1, necho=necho)
                inputs = torch_channel_deconcate(inputs)
                Recons.append(inputs.cpu().detach())

                if idx == 1:
                    print('Sampling ratio: {}%'.format(torch.mean(masks[0, 0, 0, :, :, 0])*100))
                if idx % 10 == 0:
                    print('Finish slice #', idx)
        elif opt['res'] == 1:
            for idx, (kdatas, targets, recon_input, csms, csm_lowres, brain_masks, brain_masks_erode) in enumerate(testLoader):
                kdatas = kdatas.to(device)
                targets = targets.to(device)
                csms = csms.to(device)
                
                inputs = backward_multiEcho(kdatas, csms, masks, flip, echo_cat=1, necho=necho)
                inputs = torch_channel_deconcate(inputs)
                Recons.append(inputs.cpu().detach())

                if idx == 1:
                    print('Sampling ratio: {}%'.format(torch.mean(masks[0, 0, 0, :, :, 0])*100))
                if idx % 10 == 0:
                    print('Finish slice #', idx)
        
        # write into .mat file
        Recons_ = np.squeeze(r2c(np.concatenate(Recons, axis=0), 1))
        Recons_ = np.transpose(Recons_, [0, 2, 3, 1])
        if opt['res'] == 1:
            kdata = np.fft.fftshift(np.fft.fftn(Recons_, axes=(0, 1, 2)), axes=(0, 1, 2))
            kdata = np.pad(kdata, ((nslice//2, nslice//2), (nrow//2, nrow//2), (ncol//2, ncol//2), (0, 0)))
            Recons_ = np.fft.ifftn(np.fft.ifftshift(kdata, axes=(0, 1, 2)), axes=(0, 1, 2))  # (nslice, nrow, ncol, necho)
            nslice *= 2
            nrow *= 2
            ncol *= 2
        save_mat(rootName+'/results_ablation2/iField_mask={}.mat'.format(opt['mask']), 'Recons', Recons_)
        
        # write into .bin file
        # (nslice, 2, 7, 258, 112) to (112, 258, nslice, 7, 2)
        iField = np.transpose(np.concatenate(Recons, axis=0), [4, 3, 0, 2, 1])
        if opt['res'] == 1:
            Recons_ = np.transpose(Recons_, [0, 3, 1, 2])
            iField = np.concatenate((Recons_.real[:, np.newaxis, ...], Recons_.imag[:, np.newaxis, ...]), axis=1)
            iField = np.transpose(iField, [4, 3, 0, 2, 1])
            iField = iField.astype('float32')
        iField[:, :, 1::2, :, :] = - iField[:, :, 1::2, :, :]
        iField[..., 1] = - iField[..., 1]
        print('iField size is: ', iField.shape)
        if os.path.exists(rootName+'/results_QSM/iField.bin'):
            os.remove(rootName+'/results_QSM/iField.bin')
        iField.tofile(rootName+'/results_QSM/iField.bin')
        print('Successfully save iField.bin')

        # run MEDIN
        if opt['res'] == 0:
            os.system('medi ' + rootName + '/results_QSM/iField.bin' 
                    + ' --parameter ' + rootName + '/results_QSM/parameter.txt'
                    + ' --temp ' + rootName +  '/results_QSM/'
                    + ' --GPU ' + ' --device ' + opt['gpu_id'] 
                    + ' --CSF ' + ' -of QR')
        elif opt['res'] == 1:
            os.system('medi ' + rootName + '/results_QSM/iField.bin' 
                    + ' --parameter ' + rootName + '/results_QSM/parameter_interpolate.txt'
                    + ' --temp ' + rootName +  '/results_QSM/'
                    + ' --GPU ' + ' --device ' + opt['gpu_id'] 
                    + ' --CSF ' + ' -of QR')
        
        # read .bin files and save into .mat files
        if opt['res'] == 0:
            QSM = np.fromfile(rootName+'/results_QSM/recon_QSM_07.bin', 'f4')
        elif opt['res'] == 1:
            QSM = np.fromfile(rootName+'/results_QSM/recon_QSM_10.bin', 'f4')
        QSM = np.transpose(QSM.reshape([ncol, nrow, nslice]), [2, 1, 0])

        iMag = np.fromfile(rootName+'/results_QSM/iMag.bin', 'f4')
        iMag = np.transpose(iMag.reshape([ncol, nrow, nslice]), [2, 1, 0])

        RDF = np.fromfile(rootName+'/results_QSM/RDF.bin', 'f4')
        RDF = np.transpose(RDF.reshape([ncol, nrow, nslice]), [2, 1, 0])

        R2star = np.fromfile(rootName+'/results_QSM/R2star.bin', 'f4')
        R2star = np.transpose(R2star.reshape([ncol, nrow, nslice]), [2, 1, 0])

        Mask = np.fromfile(rootName+'/results_QSM/Mask.bin', 'f4')
        Mask = np.transpose(Mask.reshape([ncol, nrow, nslice]), [2, 1, 0]) > 0

        adict = {}
        adict['QSM'], adict['iMag'], adict['RDF'] = QSM, iMag, RDF
        adict['R2star'], adict['Mask'] = R2star, Mask
        sio.savemat(rootName+'/results_ablation2/QSM_mask={}.mat'.format(opt['mask']), adict)


