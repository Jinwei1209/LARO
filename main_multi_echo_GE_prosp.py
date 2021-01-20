"""
    Prospectove recon on multi-echo kspace data from the GE scanner
"""
import os
import time
import torch
import math
import argparse
import cfl
import sys
import scipy.io as sio
import numpy as np
os.putenv("DEBUG_LEVEL", "5") 
os.system('ln -fs GERecon.so.python35 GERecon.so')

from IPython.display import clear_output
from torch.utils import data
from loader.kdata_multi_echo_GE import kdata_multi_echo_GE
from utils.data import *
from utils.loss import lossL1
from utils.test import Metrices
from utils.operators import backward_multiEcho
from models.resnet_with_dc import Resnet_with_DC2
from fits.fits import fit_R2_LM
from bart import bart
from GERecon import Pfile

if __name__ == '__main__':

    lrG_dc = 1e-3
    niter = 500
    batch_size = 1
    display_iters = 10
    gen_iterations = 1
    t0 = time.time()
    epoch = 0
    errL2_dc_sum = 0
    PSNRs_val = []
    Validation_loss = []
    ncoil = 8
    nrow = 206
    ncol = 80
    necho = 10
    nslice = 256
    lambda_dll2 = 1e-3
    
    # typein parameters
    parser = argparse.ArgumentParser(description='Multi_echo_GE')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--flag_train', type=int, default=0)  # 1 for training, 0 for testing
    parser.add_argument('--echo_cat', type=int, default=1)  # flag to concatenate echo dimension into channel
    parser.add_argument('--solver', type=int, default=1)  # 0 for deep Quasi-newton, 1 for deep ADMM,
                                                          # 2 for TV Quasi-newton, 3 for TV ADMM.
    parser.add_argument('--K', type=int, default=10)  # number of unrolls
    parser.add_argument('--loupe', type=int, default=0)  #-1: manually designed mask, 0 fixed learned mask
                                                         # 1: mask learning, same mask across echos, 2: mask learning, mask for each echo
    parser.add_argument('--norm_last', type=int, default=0)  # 0: norm+relu, 1: relu+norm
    parser.add_argument('--temporal_conv', type=int, default=0) # 0: no temporal, 1: center, 2: begining
    parser.add_argument('--1d_type', type=str, default='shear')  # 'shear' or 'random' sampling type of 1D mask
    parser.add_argument('--samplingRatio', type=float, default=0.2)

    parser.add_argument('--precond', type=int, default=0)  # flag to use preconsitioning
    parser.add_argument('--att', type=int, default=0)  # flag to use attention-based denoiser
    parser.add_argument('--random', type=int, default=0)  # flag to multiply the input data with a random complex number
    parser.add_argument('--normalization', type=int, default=0)  # 0 for no normalization
    opt = {**vars(parser.parse_args())}

    K = opt['K']
    norm_last = opt['norm_last']
    flag_temporal_conv = opt['temporal_conv']

    # pfile = Pfile('/data/Jinwei/QSM_raw_CBIC/pfiles/alexey/P27136.7')
    # kspace = np.zeros([nslice, nrow, ncol, 32, necho], dtype=np.complex_)
    # for echo in range(1):
    #     print('Loading echo #', echo)
    #     for slice_num in range(ncol):
    #         kspace[:, :, slice_num, :, echo] = pfile.KSpace(slice_num, echo)
    # print('Compressing echo: ', 0)
    # kspace_cc = bart(1, 'cc -p 8 -S', kspace[..., 0])
    # print('Estimating coil sensitivity maps from the first echo')
    # sens_1echo_3d = bart(1, 'ecalib -m1 -I', kspace_cc)
    # del kspace
    
    # loading kspace data from pfile directly
    pfile = Pfile('/data/Jinwei/QSM_raw_CBIC/pfiles/alexey/P29184.7')
    kspace = np.zeros([nslice, nrow, ncol, 32, necho], dtype=np.complex_)
    for echo in range(necho):
        print('Loading echo #', echo)
        for slice_num in range(ncol):
            kspace[:, :, slice_num, :, echo] = pfile.KSpace(slice_num, echo)
    
    # 3D coil compression and ESPIRiT
    necho = kspace.shape[-1]
    ncoil = 8
    nslice = kspace.shape[0]
    nrow = kspace.shape[1]
    ncol = kspace.shape[2]
    kspace_cc = np.zeros((nslice, nrow, ncol, ncoil, necho), dtype=np.complex_)

    for i in range(necho):
        print('Compressing echo: ', i)
        kspace_cc[..., i] = bart(1, 'cc -p 8 -S', kspace[..., i])
        if i == 0:
            print('Estimating coil sensitivity maps from the first echo')
            sens_1echo_3d = bart(1, 'ecalib -m1 -I', kspace_cc[..., i])
    print('Finish Compression and ESPIRiT')
    kspace = np.transpose(kspace_cc, (1, 2, 0, 3, 4))
    kspace_fft_z = np.fft.fftshift(np.fft.ifft(kspace, axis=2), axes=2)
    print('Finish IFFT along readout direction')

    # concatenate echo dimension to the channel dimension for TV regularization
    if opt['solver'] > 1:
        opt['echo_cat'] = 1

    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_id']
    # rootName = '/data/Jinwei/Multi_echo_slice_recon_GE'
    rootName = '/data/Jinwei/QSM_raw_CBIC'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if opt['loupe'] == -1:
        # load manually designed mask
        masks = np.real(readcfl(rootName+'/masks/mask_{}m'.format(opt['samplingRatio'])))
        # masks = np.real(readcfl(rootName+'/masks/mask_{}_1d_{}'.format(opt['samplingRatio'], opt['1d_type'])))
    elif opt['loupe'] == 0:
        # load fixed loupe optimized mask
        masks = np.real(readcfl(rootName+'/masks/mask_{}'.format(opt['samplingRatio'])))
        
    if opt['loupe'] < 1:
        # for 2D random sampling 
        masks = masks[..., np.newaxis] # (nrow, ncol, 1)
        
        # # for 1D echo-identical sampling
        # masks = masks[..., 0, np.newaxis] # (nrow, ncol, 1)
        # masks[nrow//2-13:nrow//2+12, ncol//2-13:ncol//2+12, ...] = 1 # add calibration region

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

        # # for 1D sampling
        # masks = masks[..., np.newaxis] # (nrow, ncol, necho, 1)
        # masks[nrow//2-13:nrow//2+12, ncol//2-13:ncol//2+12, ...] = 1 # add calibration region
        # masks = torch.tensor(masks, device=device).float()
        # # to complex data
        # masks = torch.cat((masks, torch.zeros(masks.shape).to(device)),-1) # (nrow, ncol, necho, 2)
        # # permute echo dimension
        # masks = masks.permute(2, 0, 1, 3) # (necho, nrow, ncol, 2)
        # # add coil dimension
        # masks = masks[None, ...] # (1, necho, nrow, ncol, 2)
        # masks = torch.cat(ncoil*[masks]) # (ncoil, necho, nrow, ncol, 2)
        # # add batch dimension
        # masks = masks[None, ...] # (1, ncoil, necho, nrow, ncol, 2)
    else:
        masks = []

    if np.mean(abs(kspace[:, :, 100, 0, 0]) > 0) > 0.3:
        flag_fully_sampled = 1
        masks[..., 0] = 1
        print('Fully sampled scan with sampling ratio:', np.mean(abs(kspace[:, :, 100, 0, 0]) > 0))
    else:
        flag_fully_sampled = 0
        print('Under sampled scan with sampling ratio:', np.mean(abs(kspace[:, :, 100, 0, 0]) > 0))
 
    # flip matrix
    flip = torch.ones([necho, nrow, ncol, 1]) 
    flip = torch.cat((flip, torch.zeros(flip.shape)), -1).to(device)
    flip[:, ::2, ...] = - flip[:, ::2, ...] 
    flip[:, :, ::2, ...] = - flip[:, :, ::2, ...]
    # add batch dimension
    flip = flip[None, ...] # (1, necho, nrow, ncol, 2)

    # for test
    if opt['flag_train'] == 0:
        if opt['echo_cat'] == 1:
            netG_dc = Resnet_with_DC2(
                input_channels=2*necho,
                filter_channels=32*necho,
                lambda_dll2=lambda_dll2,
                ncoil=ncoil,
                K=K,
                echo_cat=1,
                flag_solver=opt['solver'],
                flag_precond=opt['precond'],
                flag_loupe=opt['loupe'],
                samplingRatio=opt['samplingRatio'],
                norm_last=norm_last,
                flag_temporal_conv=flag_temporal_conv,
                flag_BCRNN=1
            )
        else:
            netG_dc = Resnet_with_DC2(
                input_channels=2,
                filter_channels=32,
                lambda_dll2=lambda_dll2,
                ncoil=ncoil,
                K=K,
                echo_cat=0,
                flag_solver=opt['solver'],
                flag_precond=opt['precond'],
                flag_loupe=opt['loupe'],
                samplingRatio=opt['samplingRatio']
            )
        if opt['solver'] < 2:
            weights_dict = torch.load(rootName+'/weights/echo_cat={}_solver={}_K={}_loupe={}_ratio={}_{}{}_bcrnn_.pt' \
            .format(opt['echo_cat'], opt['solver'], opt['K'], opt['loupe'], opt['samplingRatio'], norm_last, flag_temporal_conv))
            netG_dc.load_state_dict(weights_dict)
        netG_dc.to(device)
        netG_dc.eval()

        Recons = []
        with torch.no_grad():
            for idx in range(nslice):
                kdata = kspace_fft_z[:, :, idx, :, :]
                kdata = np.transpose(kdata, (2, 3, 0, 1))  # (coil, echo, row, col)
                kdata = c2r_kdata(kdata) # (coil, echo, row, col, 2) with last dimension real&imag
                kdatas = torch.from_numpy(kdata[np.newaxis, ...])

                csm = np.concatenate((sens_1echo_3d[idx, :, ncol//2:, :], sens_1echo_3d[idx, :, :ncol//2, :]), axis=1)
                csm = np.transpose(csm, (2, 0, 1))[:, np.newaxis, ...]  # (coil, 1, row, col)
                csm = np.repeat(csm, necho, axis=1)  # (coil, echo, row, col)
                csm = c2r_kdata(csm) # (coil, echo, row, col, 2) with last dimension real&imag
                csms = torch.from_numpy(csm[np.newaxis, ...])

                if idx == 1 and opt['loupe'] > 0:
                    Mask = netG_dc.Mask.cpu().detach().numpy()
                    print('Saving sampling mask: %', np.mean(Mask)*100)
                    save_mat(rootName+'/results/Mask_echo_cat={}_solver={}_K={}_loupe={}_ratio={}.mat' \
                            .format(opt['echo_cat'], opt['solver'], opt['K'], opt['loupe'], opt['samplingRatio']), 'Mask', Mask)
                if (idx == 1) and (flag_fully_sampled == 0):
                    print('Sampling ratio: {}%'.format(torch.mean(netG_dc.Mask)*100))
                if idx % 10 == 0:
                    print('Finish slice #', idx)
                kdatas = kdatas.to(device)
                csms = csms.to(device)

                if flag_fully_sampled == 1:
                    Xs_1 = backward_multiEcho(kdatas, csms, masks, flip, opt['echo_cat'])
                else:
                    Xs_1 = netG_dc(kdatas, csms, masks, flip)[-1]
                if opt['echo_cat']:
                    Xs_1 = torch_channel_deconcate(Xs_1)
                Recons.append(Xs_1.cpu().detach())

            # write into .mat file
            Recons_ = np.squeeze(r2c(np.concatenate(Recons, axis=0), opt['echo_cat']))
            Recons_ = np.transpose(Recons_, [0, 2, 3, 1])
            if opt['loupe'] == -1:
                save_mat(rootName+'/results/iField_{}_opt=0_cc.mat'.format(opt['samplingRatio']), 'Recons', Recons_)
            elif opt['loupe'] == 0:
                save_mat(rootName+'/results/iField_{}_opt=1_cc.mat'.format(opt['samplingRatio']), 'Recons', Recons_)
        

            # write into .bin file
            # (256, 2, 10, 206, 80) to (80, 206, 256, 10, 2)
            print('iField size is: ', np.concatenate(Recons, axis=0).shape)
            iField = np.transpose(np.concatenate(Recons, axis=0), [4, 3, 0, 2, 1])
            iField[:, :, 1::2, :, :] = - iField[:, :, 1::2, :, :]
            iField[..., 1] = - iField[..., 1]
            if os.path.exists(rootName+'/results_QSM_prosp/iField.bin'):
                os.remove(rootName+'/results_QSM_prosp/iField.bin')
            iField.tofile(rootName+'/results_QSM_prosp/iField.bin')
            print('Successfully save iField.bin')

            # run MEDIN
            os.system('medi ' + rootName + '/results_QSM_prosp/iField.bin' 
                    + ' --parameter ' + rootName + '/results_QSM_prosp/parameter.txt'
                    + ' --temp ' + rootName +  '/results_QSM_prosp/'
                    + ' --GPU ' + ' --device ' + opt['gpu_id'] 
                    + ' --CSF ' + ' -of QR')
            
            # read .bin files and save into .mat files
            QSM = np.fromfile(rootName+'/results_QSM_prosp/recon_QSM_10.bin', 'f4')
            QSM = np.transpose(QSM.reshape([ncol, nrow, nslice]), [2, 1, 0])

            iMag = np.fromfile(rootName+'/results_QSM_prosp/iMag.bin', 'f4')
            iMag = np.transpose(iMag.reshape([ncol, nrow, nslice]), [2, 1, 0])

            RDF = np.fromfile(rootName+'/results_QSM_prosp/RDF.bin', 'f4')
            RDF = np.transpose(RDF.reshape([ncol, nrow, nslice]), [2, 1, 0])

            R2star = np.fromfile(rootName+'/results_QSM_prosp/R2star.bin', 'f4')
            R2star = np.transpose(R2star.reshape([ncol, nrow, nslice]), [2, 1, 0])

            Mask = np.fromfile(rootName+'/results_QSM_prosp/Mask.bin', 'f4')
            Mask = np.transpose(Mask.reshape([ncol, nrow, nslice]), [2, 1, 0]) > 0

            adict = {}
            adict['QSM'], adict['iMag'], adict['RDF'] = QSM, iMag, RDF
            adict['R2star'], adict['Mask'] = R2star, Mask
            if opt['loupe'] == -1:
                sio.savemat(rootName+'/results/QSM_{}_opt=0_cc.mat'.format(opt['samplingRatio']), adict)
            else:
                sio.savemat(rootName+'/results/QSM_{}_opt=1_cc.mat'.format(opt['samplingRatio']), adict)


