"""
    Experiment on multi-echo kspace data reconstruction from GE scanner
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
from loader.kdata_multi_echo_GE import kdata_multi_echo_GE
from loader.kdata_multi_echo_CBIC import kdata_multi_echo_CBIC
from loader.kdata_multi_echo_CBIC_prosp import kdata_multi_echo_CBIC_prosp
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
    nrow = 206
    ncol = 80
    nslice = 200
    lambda_dll2 = 1e-3
    TEs = [0.001972, 0.005356, 0.008740, 0.012124, 0.015508, 0.018892, 0.022276, 0.025660, 0.029044, 0.032428]

    # typein parameters
    parser = argparse.ArgumentParser(description='Multi_echo_GE')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--flag_train', type=int, default=1)  # 1 for training, 0 for testing
    parser.add_argument('--test_sub', type=int, default=0)  # 0: junghun, 1: chao, 2: alexey, 3: hang
    parser.add_argument('--K', type=int, default=10)  # number of unrolls
    parser.add_argument('--loupe', type=int, default=0)  # -3: fixed manually designed mask across echos,
                                                         # -2: fixed learned mask across echos,
                                                         # -1: fixed manually designed mask, 
                                                         #  0: fixed learned mask, 
                                                         #  1: mask learning, same mask across echos, 
                                                         #  2: mask learning, mask for each echo
    parser.add_argument('--bcrnn', type=int, default=1)  #  0: without bcrnn blcok, 1: with bcrnn block, 2: with bclstm block
    parser.add_argument('--solver', type=int, default=1)  # 0 for deep Quasi-newton, 1 for deep ADMM,
                                                          # 2 for TV Quasi-newton, 3 for TV ADMM.
    parser.add_argument('--samplingRatio', type=float, default=0.2)  # Under-sampling ratio
    parser.add_argument('--prosp', type=int, default=0)  # flag to test on prospective data
    parser.add_argument('--flag_unet', type=int, default=0)  # flag to use unet as denoiser

    parser.add_argument('--flag_2D', type=int, default=1)  # flag to use 2D undersampling (variable density)
    parser.add_argument('--necho', type=int, default=10)  # number of echos with kspace data
    parser.add_argument('--temporal_pred', type=int, default=0)  # flag to use a 2nd recon network with temporal under-sampling
    parser.add_argument('--convft', type=int, default=0)  # 0: conventional conv layer, 1: conv2DFT layer
    parser.add_argument('--bn', type=int, default=2)  # flag to use group normalization: 0: no normalization, 2: use group normalization
    parser.add_argument('--multilevel', type=int, default=0)  # 0: original image space feature extraction and denoising, 1: multi-level
    parser.add_argument('--rank', type=int, default=0)  #  rank of compressor in forward model
    parser.add_argument('--lambda0', type=float, default=0.0)  # weighting of low rank approximation loss
    parser.add_argument('--lambda1', type=float, default=0.0)  # weighting of r2s reconstruction loss
    parser.add_argument('--lambda2', type=float, default=0.0)  # weighting of p1 reconstruction loss
    parser.add_argument('--lambda_maskbce', type=float, default=0.0)  # weighting of Maximal cross entropy in masks
    parser.add_argument('--loss', type=int, default=0)  # 0: SSIM loss, 1: L1 loss, 2: L2 loss
    parser.add_argument('--weights_dir', type=str, default='weights_ablation2')
    parser.add_argument('--echo_cat', type=int, default=1)  # flag to concatenate echo dimension into channel
    parser.add_argument('--norm_last', type=int, default=0)  # 0: norm+relu, 1: relu+norm
    parser.add_argument('--temporal_conv', type=int, default=0) # 0: no temporal, 1: center, 2: begining
    parser.add_argument('--1d_type', type=str, default='shear')  # 'shear' or 'random' sampling type of 1D mask
    parser.add_argument('--precond', type=int, default=0)  # flag to use preconsitioning
    parser.add_argument('--att', type=int, default=0)  # flag to use attention-based denoiser
    parser.add_argument('--random', type=int, default=0)  # flag to multiply the input data with a random complex number
    parser.add_argument('--normalization', type=int, default=0)  # 0 for no normalization
    opt = {**vars(parser.parse_args())}
    K = opt['K']
    norm_last = opt['norm_last']
    flag_temporal_conv = opt['temporal_conv']
    lambda0 = opt['lambda0']
    lambda1 = opt['lambda1']
    lambda2 = opt['lambda2']
    lambda_maskbce = opt['lambda_maskbce']  # 0.01 too large
    rank = opt['rank']
    necho = opt['necho']
    necho_pred = 10 - necho
    # concatenate echo dimension to the channel dimension for TV regularization
    if opt['solver'] > 1:
        opt['echo_cat'] = 1
    if opt['loupe'] > 0:
        niter = 500
    else:
        niter = 100

    # flag to use hidden state recurrent pass in BCRNN layer
    if opt['solver'] == 1 and opt['bcrnn'] == 0:
        flag_bcrnn = 1
        flag_hidden = 0
    elif opt['solver'] == 1 and opt['bcrnn'] == 1:
        flag_bcrnn = 1
        flag_hidden = 1
    elif opt['solver'] == 0 and opt['bcrnn'] == 0:
        flag_bcrnn = 0
        flag_hidden = 0

    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_id']
    # rootName = '/data/Jinwei/Multi_echo_slice_recon_GE'
    rootName = '/data/Jinwei/QSM_raw_CBIC'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # torch.manual_seed(0)

    if opt['loupe'] == 0:
        # load fixed loupe optimized mask
        # masks = np.real(readcfl(rootName+'/masks/mask_{}_ssim_echo_new'.format(opt['samplingRatio'])))[0, :, :]
        masks = np.real(readcfl(rootName+'/masks2/mask_{}_ssim'.format(opt['samplingRatio'])))
    elif opt['loupe'] == -1:
        # load manually designed mask
        # masks = np.real(readcfl(rootName+'/masks/mask_{}m_echo'.format(opt['samplingRatio'])))[0, :, :]
        masks = np.real(readcfl(rootName+'/masks2/mask_{}m'.format(opt['samplingRatio'])))
    elif opt['loupe'] == -2:
        # load fixed loupe optimized mask across echos
        # masks = np.real(readcfl(rootName+'/masks/mask_{}_ssim_echo_new'.format(opt['samplingRatio'])))  # equal ratios from same PDF
        # masks = np.real(readcfl(rootName+'/masks/mask_{}_ssim_echo_test'.format(opt['samplingRatio'])))  # equal ratios from same PDF, prospective qihao
        masks = np.real(readcfl(rootName+'/masks2/mask_{}_echo'.format(opt['samplingRatio'])))
    elif opt['loupe'] == -3:
        masks = np.real(readcfl(rootName+'/masks/mask_{}m_echo'.format(opt['samplingRatio'])))

    if opt['loupe'] < 1 and opt['loupe'] > -2:
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
    elif opt['loupe'] == -2 or opt['loupe'] == -3:
        masks = masks[..., np.newaxis] # (necho, nrow, ncol, 1)
        masks = torch.tensor(masks, device=device).float()
        # to complex data
        masks = torch.cat((masks, torch.zeros(masks.shape).to(device)),-1) # (necho, nrow, ncol, 2)
        # add coil dimension
        masks = masks[None, ...] # (1, necho, nrow, ncol, 2)
        masks = torch.cat(ncoil*[masks]) # (ncoil, necho, nrow, ncol, 2)
        # add batch dimension
        masks = masks[None, ...] # (1, ncoil, necho, nrow, ncol, 2)
    else:
        masks = []
 
    # flip matrix
    flip = torch.ones([necho, nrow, ncol, 1]) 
    flip = torch.cat((flip, torch.zeros(flip.shape)), -1).to(device)
    flip[:, ::2, ...] = - flip[:, ::2, ...] 
    flip[:, :, ::2, ...] = - flip[:, :, ::2, ...]
    # add batch dimension
    flip = flip[None, ...] # (1, necho, nrow, ncol, 2)

    # principle vectors
    U = readcfl(rootName+'/data_cfl/dictionary/U')
    U = torch.tensor(c2r_kdata(U)).to(device)

    # training
    if opt['flag_train'] == 1:
        # memory_pre_alloc(opt['gpu_id'])
        if opt['loss'] == 0:
            loss = SSIM()
        elif opt['loss'] == 1:
            loss = lossL1()
        elif opt['loss'] == 2:
            loss = lossL2()

        loss_cem = CrossEntropyMask(radius=30)  # cross entropy loss for mask

        # dataLoader = kdata_multi_echo_GE(
        dataLoader = kdata_multi_echo_CBIC(
            rootDir=rootName,
            contrast='MultiEcho', 
            split='train',
            normalization=opt['normalization'],
            echo_cat=opt['echo_cat']
        )
        trainLoader = data.DataLoader(dataLoader, batch_size=batch_size, shuffle=True, num_workers=1)

        # dataLoader_val = kdata_multi_echo_GE(
        dataLoader_val = kdata_multi_echo_CBIC(  
            rootDir=rootName,
            contrast='MultiEcho', 
            split='val',
            normalization=opt['normalization'],
            echo_cat=opt['echo_cat']
        )
        valLoader = data.DataLoader(dataLoader_val, batch_size=batch_size, shuffle=True, num_workers=1)

        if opt['echo_cat'] == 1:
            netG_dc = Resnet_with_DC2(
                input_channels=2*necho,
                filter_channels=32*necho,
                necho=necho,
                necho_pred=necho_pred,
                lambda_dll2=lambda_dll2,
                ncoil=ncoil,
                K=K,
                U=U,
                rank=opt['rank'],
                echo_cat=1,
                flag_2D=opt['flag_2D'],
                flag_solver=opt['solver'],
                flag_precond=opt['precond'],
                flag_loupe=opt['loupe'],
                flag_temporal_pred=0,
                flag_convFT=opt['convft'],
                flag_multi_level=opt['multilevel'],
                flag_bn=opt['bn'],
                samplingRatio=opt['samplingRatio'],
                norm_last=norm_last,
                flag_temporal_conv=flag_temporal_conv,
                flag_BCRNN=flag_bcrnn,
                flag_hidden=flag_hidden,
                flag_unet=opt['flag_unet'],
                flag_att=opt['att'],
                flag_cp=1
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
        netG_dc.to(device)
        if opt['loupe'] < 1 and opt['loupe'] > -2:
            weights_dict = torch.load(rootName+'/'+opt['weights_dir']+'/bcrnn={}_loss={}_K=2_loupe=1_ratio={}_solver={}_unet={}_last.pt'
                        .format(opt['bcrnn'], 0, opt['samplingRatio'], opt['solver'], opt['flag_unet']))
            weights_dict['lambda_lowrank'] = torch.tensor([lambda_dll2])
            netG_dc.load_state_dict(weights_dict)
        elif opt['loupe'] == -2 or opt['loupe'] == -3:
            weights_dict = torch.load(rootName+'/'+opt['weights_dir']+'/bcrnn={}_loss={}_K=2_loupe=2_ratio={}_solver={}_unet={}_last.pt'
                        .format(opt['bcrnn'], 0, opt['samplingRatio'], opt['solver'], opt['flag_unet']))
            weights_dict['lambda_lowrank'] = torch.tensor([lambda_dll2])
            netG_dc.load_state_dict(weights_dict)

        # if opt['temporal_pred'] == 1:
        #     weights_dict = torch.load(rootName+'/'+opt['weights_dir']+'/bcrnn={}_loss={}_K=10_loupe=0_ratio={}_solver={}.pt'
        #                 .format(opt['bcrnn'], 0, opt['samplingRatio'], opt['solver']))
        #     netG_dc.load_state_dict(weights_dict)

        # optimizer
        optimizerG_dc = torch.optim.Adam(netG_dc.parameters(), lr=lrG_dc, betas=(0.9, 0.999))
        ms = [0.2, 0.4, 0.6, 0.8]
        ms = [np.floor(m * niter).astype(int) for m in ms]
        scheduler = MultiStepLR(optimizerG_dc, milestones = ms, gamma = 0.2)

        # logger
        logger = Logger(rootName+'/'+opt['weights_dir'], opt)

        while epoch < niter:

            epoch += 1

            # training phase
            netG_dc.train()
            metrices_train = Metrices()
            for idx, (kdatas, targets, recon_input, csms, csm_lowres, brain_masks, brain_masks_erode) in enumerate(trainLoader):
                kdatas = kdatas[:, :, :necho, ...]  # temporal undersampling
                csms = csms[:, :, :necho, ...]  # temporal undersampling
                recon_input = recon_input[:, :2*necho, ...]  # temporal undersampling
                if opt['temporal_pred'] == 0:
                    targets = targets[:, :2*necho, ...]  # temporal undersampling
                    brain_masks = brain_masks[:, :2*necho, ...]  # temporal undersampling

                if torch.sum(brain_masks) == 0:
                    continue

                if gen_iterations%display_iters == 0:

                    print('epochs: [%d/%d], batchs: [%d/%d], time: %ds'
                    % (epoch, niter, idx, 1600//batch_size, time.time()-t0))

                    print('bcrnn: {}, loss: {}, K: {}, loupe: {}, solver: {}, rank: {}'.format( \
                            opt['bcrnn'], opt['loss'], opt['K'], opt['loupe'], opt['solver'], opt['rank']))
                    
                    if opt['loupe'] > 0:
                        print('Sampling ratio cal: %f, Sampling ratio setup: %f, Pmask: %f' 
                        % (torch.mean(netG_dc.Mask), netG_dc.samplingRatio, torch.mean(netG_dc.Pmask)))
                    else:
                        print('Sampling ratio cal: %f' % (torch.mean(netG_dc.Mask)))

                    if opt['solver'] < 3:
                        print('netG_dc --- loss_L2_dc: %f, lambda_dll2: %f, lambda_lowrank: %f'
                            % (errL2_dc_sum/display_iters, netG_dc.lambda_dll2, netG_dc.lambda_lowrank))
                    else:
                        print('netG_dc --- loss_L2_dc: %f, lambda_tv: %f, rho_penalty: %f'
                            % (errL2_dc_sum/display_iters, netG_dc.lambda_tv, netG_dc.rho_penalty))

                    print('Average PSNR in Training dataset is %.2f' 
                    % (np.mean(np.asarray(metrices_train.PSNRs[-1-display_iters*batch_size:]))))
                    if epoch > 1:
                        print('Average PSNR in Validation dataset is %.2f' 
                        % (np.mean(np.asarray(metrices_val.PSNRs))))
                    
                    print(' ')

                    errL2_dc_sum = 0

                kdatas = kdatas.to(device)
                targets = targets.to(device)
                recon_input = recon_input.to(device)
                csms = csms.to(device)
                csm_lowres = csm_lowres.to(device)
                brain_masks = brain_masks.to(device)
                brain_masks_erode = brain_masks_erode.to(device)

                # operator = Back_forward_multiEcho(csms, masks, 0)
                # test_image = operator.AtA(targets, 0).cpu().detach().numpy()
                # save_mat(rootName+'/results/test_image.mat', 'test_image', test_image)

                optimizerG_dc.zero_grad()
                if opt['temporal_pred'] == 1:
                    Xs = netG_dc(kdatas, csms, csm_lowres, masks, flip, x_input=recon_input)
                else:
                    Xs = netG_dc(kdatas, csms, csm_lowres, masks, flip)

                if lambda1 == 1:
                    # compute paremeters label
                    tmp = torch_channel_deconcate(targets)
                    mags_target = torch.sqrt(tmp[:, 0, ...]**2 + tmp[:, 1, ...]**2).permute(0, 2, 3, 1)
                    [r2s_targets, water_target] = arlo(TEs, mags_target, flag_water=1)
                    # [r2s_targets, water_target] = fit_R2_LM(tmp.permute(0, 3, 4, 1, 2), max_iter=1)
                    [p1_targets, p0_target] = fit_complex(tmp.permute(0, 3, 4, 1, 2), max_iter=1)
                lossl2_sum = 0
                for i in range(len(Xs)):
                    if opt['loss'] == 0:
                        if lambda1 == 0:
                            # ssim loss
                            lossl2_sum -= loss(Xs[i]*brain_masks, targets*brain_masks)
                        # # low rank loss
                        # lossl2_sum -= lambda0 * loss(low_rank_approx(Xs[i], kdatas, csm_lowres, k=rank)*brain_masks, targets*brain_masks)
                        elif lambda1 == 1:
                            # compute parameters
                            Xsi = torch_channel_deconcate(Xs[i])
                            mags = torch.sqrt(Xsi[:, 0, ...]**2 + Xsi[:, 1, ...]**2).permute(0, 2, 3, 1)
                            [r2s, water] = arlo(TEs, mags, flag_water=1)
                            # [r2s, water] = fit_R2_LM(Xsi.permute(0, 3, 4, 1, 2), max_iter=1)
                            [p1, p0] = fit_complex(Xsi.permute(0, 3, 4, 1, 2), max_iter=1)
                            # parameter estimation loss
                            lossl2_sum -= lambda1 * loss(r2s[:, None, ...]*brain_masks_erode[:, 0:1, ...], r2s_targets[:, None, ...]*brain_masks_erode[:, 0:1, ...])
                            # lossl2_sum -= lambda1 * loss(water[:, None, ...]*brain_masks_erode[:, 0:1, ...], water_target[:, None, ...]*brain_masks_erode[:, 0:1, ...])
                            lossl2_sum -= lambda2 * loss(p1[:, None, ...]*brain_masks_erode[:, 0:1, ...], p1_targets[:, None, ...]*brain_masks_erode[:, 0:1, ...])
                            # lossl2_sum -= lambda2 * loss(p0[:, None, ...]*brain_masks_erode[:, 0:1, ...], p0_target[:, None, ...]*brain_masks_erode[:, 0:1, ...])
                    elif opt['loss'] > 0:
                        # L1 or L2 loss
                        lossl2_sum += loss(Xs[i]*brain_masks, targets*brain_masks)
                        # lossl2_sum += lambda0 * loss(low_rank_approx(Xs[i], kdatas, csm_lowres, k=rank)*brain_masks, targets*brain_masks)
                        # # compute r2s and field
                        # Xsi = torch_channel_deconcate(Xs[i])
                        # mags = torch.sqrt(Xsi[:, 0, ...]**2 + Xsi[:, 1, ...]**2).permute(0, 2, 3, 1)
                        # r2s = arlo(TEs, mags)
                        # p1 = fit_complex(Xsi.permute(0, 3, 4, 1, 2), max_iter=0)
                        # # parameter estimation loss
                        # lossl2_sum += lambda1 * loss(r2s[:, None, ...]*brain_masks_erode[:, 0:1, ...], r2s_targets[:, None, ...]*brain_masks_erode[:, 0:1, ...])
                        # lossl2_sum += lambda2 * loss(p1[:, None, ...]*brain_masks_erode[:, 0:1, ...], p1_targets[:, None, ...]*brain_masks_erode[:, 0:1, ...])
                
                # # maximal cross entropy mask loss
                # lossl2_sum -= lambda_maskbce * loss_cem(netG_dc.Pmask_rescaled)
                lossl2_sum.backward()
                optimizerG_dc.step()

                errL2_dc_sum += lossl2_sum.item()

                # calculating metrices
                metrices_train.get_metrices(Xs[-1]*brain_masks, targets*brain_masks)
                gen_iterations += 1

            if opt['loupe'] < 1:
                scheduler.step(epoch)
            
            # validation phase
            netG_dc.eval()
            metrices_val = Metrices()
            loss_total_list = []
            with torch.no_grad():  # to solve memory exploration issue
                for idx, (kdatas, targets, recon_input, csms, csm_lowres, brain_masks, brain_masks_erode) in enumerate(valLoader):
                    kdatas = kdatas[:, :, :necho, ...]  # temporal undersampling
                    csms = csms[:, :, :necho, ...]  # temporal undersampling
                    recon_input = recon_input[:, :2*necho, ...]  # temporal undersampling
                    if opt['temporal_pred'] == 0:
                        targets = targets[:, :2*necho, ...]  # temporal undersampling
                        brain_masks = brain_masks[:, :2*necho, ...]  # temporal undersampling

                    if torch.sum(brain_masks) == 0:
                        continue

                    kdatas = kdatas.to(device)
                    targets = targets.to(device)
                    recon_input = recon_input.to(device)
                    csms = csms.to(device)
                    csm_lowres = csm_lowres.to(device)
                    brain_masks = brain_masks.to(device)

                    if opt['temporal_pred'] == 1:
                        Xs = netG_dc(kdatas, csms, csm_lowres, masks, flip, x_input=recon_input)
                    else:
                        Xs = netG_dc(kdatas, csms, csm_lowres, masks, flip)

                    metrices_val.get_metrices(Xs[-1]*brain_masks, targets*brain_masks)
                    # targets = np.asarray(targets.cpu().detach())
                    # brain_masks = np.asarray(brain_masks.cpu().detach())
                    # temp = 0
                    # for i in range(len(Xs)):
                    #     X = np.asarray(Xs[i].cpu().detach())
                    #     temp += abs(X - targets) * brain_masks
                    # X = np.asarray(Xs[-1].cpu().detach())
                    # temp += abs(X - targets) * brain_masks
                    # lossl2_sum = np.mean(temp)
                    lossl2_sum = loss(Xs[-1]*brain_masks, targets*brain_masks)
                    loss_total_list.append(lossl2_sum)

                print('\n Validation loss: %f \n' 
                    % (sum(loss_total_list) / float(len(loss_total_list))))
                Validation_loss.append(sum(loss_total_list) / float(len(loss_total_list)))
                Validation_psnr.append(np.mean(np.asarray(metrices_val.PSNRs)))
            
            # save log
            logger.print_and_save('Epoch: [%d/%d], PSNR in training: %.2f' 
            % (epoch, niter, np.mean(np.asarray(metrices_train.PSNRs))))
            logger.print_and_save('Epoch: [%d/%d], PSNR in validation: %.2f, loss in validation: %.10f' 
            % (epoch, niter, np.mean(np.asarray(metrices_val.PSNRs)), Validation_loss[-1]))

            # save weights
            if Validation_psnr[-1] == max(Validation_psnr):
                torch.save(netG_dc.state_dict(), rootName+'/'+opt['weights_dir']+'/bcrnn={}_loss={}_K={}_loupe={}_ratio={}_solver={}_unet={}.pt' \
                .format(opt['bcrnn'], opt['loss'], opt['K'], opt['loupe'], opt['samplingRatio'], opt['solver'], opt['flag_unet']))
            torch.save(netG_dc.state_dict(), rootName+'/'+opt['weights_dir']+'/bcrnn={}_loss={}_K={}_loupe={}_ratio={}_solver={}_unet={}_last.pt' \
            .format(opt['bcrnn'], opt['loss'], opt['K'], opt['loupe'], opt['samplingRatio'], opt['solver'], opt['flag_unet']))  
    
    
    # for test
    if opt['flag_train'] == 0:
        if opt['echo_cat'] == 1:
            netG_dc = Resnet_with_DC2(
                input_channels=2*necho,
                filter_channels=32*necho,
                necho=necho,
                necho_pred=necho_pred,
                lambda_dll2=lambda_dll2,
                ncoil=ncoil,
                K=K,
                U=U,
                rank=opt['rank'],
                echo_cat=1,
                flag_2D=opt['flag_2D'],
                flag_solver=opt['solver'],
                flag_precond=opt['precond'],
                flag_loupe=opt['loupe'],
                flag_temporal_pred=0,
                flag_convFT=opt['convft'],
                flag_multi_level=opt['multilevel'],
                flag_bn=opt['bn'],
                samplingRatio=opt['samplingRatio'],
                norm_last=norm_last,
                flag_temporal_conv=flag_temporal_conv,
                flag_BCRNN=flag_bcrnn,
                flag_hidden=flag_hidden,
                flag_unet=opt['flag_unet'],
                flag_att=opt['att']
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
        weights_dict = torch.load(rootName+'/'+opt['weights_dir']+'/bcrnn={}_loss={}_K={}_loupe={}_ratio={}_solver={}_unet={}.pt' \
                .format(opt['bcrnn'], opt['loss'], opt['K'], opt['loupe'], opt['samplingRatio'], opt['solver'], opt['flag_unet']))
        weights_dict['lambda_lowrank'] = torch.tensor([lambda_dll2])
        # if opt['temporal_pred'] == 1:
        #     print('Temporal Prediction with {} Echos'.format(necho))
        #     weights_dict = torch.load(rootName+'/'+opt['weights_dir']+'/bcrnn={}_loss={}_K=10_loupe=0_ratio={}_solver={}_echo={}_temporal={}_.pt'
        #                 .format(opt['bcrnn'], 0, opt['samplingRatio'], opt['solver'], necho, opt['temporal_pred']))
        netG_dc.load_state_dict(weights_dict)
        netG_dc.to(device)
        netG_dc.eval()

        Inputs = []
        Targets = []
        Targets_torch = []
        LLRs = []
        M0, R2s = [], []
        F0, P = [], []
        Recons = []
        preconds = []
        
        if opt['prosp'] == 0:
            dataLoader_test = kdata_multi_echo_CBIC(
                rootDir=rootName,
                contrast='MultiEcho', 
                split='test',
                subject=opt['test_sub'],
                normalization=opt['normalization'],
                echo_cat=opt['echo_cat']
            )
        elif opt['prosp'] == 1:
            dataLoader_test = kdata_multi_echo_CBIC_prosp(
                rootDir=rootName,
                contrast='MultiEcho', 
                split='test',
                subject=opt['test_sub'],
                loupe=opt['loupe'],
                normalization=opt['normalization'],
                echo_cat=opt['echo_cat']
            )
        testLoader = data.DataLoader(dataLoader_test, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for idx, (kdatas, targets, recon_input, csms, csm_lowres, brain_masks, brain_masks_erode) in enumerate(testLoader):
                kdatas = kdatas[:, :, :necho, ...]  # temporal undersampling
                csms = csms[:, :, :necho, ...]  # temporal undersampling
                csm_lowres = csm_lowres[:, :, :necho, ...]  # temporal undersampling
                recon_input = recon_input[:, :2*necho, ...]  # temporal undersampling

                if idx == 1 and opt['loupe'] > 0:
                    # Mask = netG_dc.Mask.cpu().detach().numpy()
                    Mask = netG_dc.Pmask.cpu().detach().numpy()
                    print('Saving sampling mask: %', np.mean(Mask)*100)
                    save_mat(rootName+'/results/Mask_bcrnn={}_loss={}_K={}_loupe={}_ratio={}_solver={}.mat' \
                            .format(opt['bcrnn'], opt['loss'], opt['K'], opt['loupe'], opt['samplingRatio'], opt['solver']), 'Mask', Mask)
                if idx == 1:
                    print('Sampling ratio: {}%'.format(torch.mean(netG_dc.Mask)*100))
                if idx % 10 == 0:
                    print('Finish slice #', idx)
                kdatas = kdatas.to(device)
                targets = targets.to(device)
                csms = csms.to(device)
                csm_lowres = csm_lowres.to(device)
                recon_input = recon_input.to(device)
                brain_masks = brain_masks.to(device)

                # inputs = backward_multiEcho(kdatas, csms, masks, flip,
                                            # opt['echo_cat'])
                if opt['temporal_pred'] == 1:
                    Xs_1 = netG_dc(kdatas, csms, csm_lowres, masks, flip, x_input=recon_input)[-1]
                else:
                    Xs_1 = netG_dc(kdatas, csms, csm_lowres, masks, flip)[-1]
                precond = netG_dc.precond
                if opt['echo_cat']:
                    targets = torch_channel_deconcate(targets)
                    recon_input = torch_channel_deconcate(recon_input)
                    # inputs = torch_channel_deconcate(inputs)
                    Xs_1 = torch_channel_deconcate(Xs_1)
                    # mags = torch.sqrt(Xs_1[:, 0, ...]**2 + Xs_1[:, 1, ...]**2).permute(0, 2, 3, 1)
                    mags_target = torch.sqrt(targets[:, 0, ...]**2 + targets[:, 1, ...]**2).permute(0, 2, 3, 1)
                    
                    # [y1_target, y2_target] = arlo(TEs, mags_target, flag_water=1)
                    # y = fit_complex(Xs_1.permute(0, 3, 4, 1, 2))
                    # y_target = fit_complex(targets.permute(0, 3, 4, 1, 2))
                    # [y1_target, y2_target] = fit_R2_LM(targets.permute(0, 3, 4, 1, 2))
                    # y = low_rank_approx(torch_channel_concate(Xs_1), kdatas, csm_lowres, k=20)
                    # y = torch_channel_deconcate(y)

                    targets_complex = r2c(targets).permute(0, 2, 3, 1)
                    # t = time.time()
                    [m0, r2s, f0, p] = fit_complex_all(targets_complex, TEs)
                    # print(time.time() - t)

                # Inputs.append(inputs.cpu().detach())
                Targets.append(targets.cpu().detach())
                Targets_torch.append(targets_complex)
                LLRs.append(recon_input.cpu().detach())
                Recons.append(Xs_1.cpu().detach())

                M0.append(m0.cpu().detach())
                R2s.append(r2s.cpu().detach())
                F0.append(f0.cpu().detach())
                P.append(p.cpu().detach())

            # write into .mat file
            Recons_ = np.squeeze(r2c(np.concatenate(Recons, axis=0), opt['echo_cat']))
            Recons_ = np.transpose(Recons_, [0, 2, 3, 1])
            if opt['lambda1'] == 1:
                save_mat(rootName+'/results_ablation2/iField_bcrnn={}_loupe={}_solver={}_sub={}_.mat' \
                    .format(opt['bcrnn'], opt['loupe'], opt['solver'], opt['test_sub']), 'Recons', Recons_)
            elif opt['lambda1'] == 0:
                save_mat(rootName+'/results_ablation2/iField_bcrnn={}_loupe={}_solver={}_sub={}_ratio={}.mat' \
                    .format(opt['bcrnn'], opt['loupe'], opt['solver'], opt['test_sub'], opt['samplingRatio']), 'Recons', Recons_)

            # M0 = np.concatenate(M0, axis=0)
            # R2s = np.concatenate(R2s, axis=0)
            # F0 = np.concatenate(F0, axis=0)
            # P = np.concatenate(P, axis=0)
            # adict = {}
            # adict['m0'], adict['r2s'], adict['f0'], adict['p'] = M0, R2s, F0, P
            # sio.savemat(rootName+'/results_ablation/four_parameters.mat', adict)

            # Targets_torch = torch.cat(Targets_torch, dim=0)
            # [M0, R2s, F0, P] = fit_complex_all(Targets_torch, TEs)
            # M0 = M0.cpu().detach().numpy()
            # R2s = R2s.cpu().detach().numpy()
            # F0 = F0.cpu().detach().numpy()
            # P = P.cpu().detach().numpy()
            # adict = {}
            # adict['m0'], adict['r2s'], adict['f0'], adict['p'] = M0, R2s, F0, P
            # sio.savemat(rootName+'/results_ablation/four_parameters_3d.mat', adict)


            # R2s_target = np.concatenate(R2s_target, axis=0)
            # save_mat(rootName+'/results_ablation/R2s_target.mat', 'R2s_target', R2s_target)
            # water_target = np.concatenate(water_target, axis=0)
            # save_mat(rootName+'/results_ablation/water_target.mat', 'water_target', water_target)

            # write into .bin file
            # (nslice, 2, 10, 206, 80) to (80, 206, nslice, 10, 2)
            print('iField size is: ', np.concatenate(Recons, axis=0).shape)
            iField = np.transpose(np.concatenate(Recons, axis=0), [4, 3, 0, 2, 1])
            iField[:, :, 1::2, :, :] = - iField[:, :, 1::2, :, :]
            iField[..., 1] = - iField[..., 1]
            print('iField size is: ', iField.shape)
            if os.path.exists(rootName+'/results_QSM/iField.bin'):
                os.remove(rootName+'/results_QSM/iField.bin')
            iField.tofile(rootName+'/results_QSM/iField.bin')
            print('Successfully save iField.bin')

            # run MEDIN
            os.system('medi ' + rootName + '/results_QSM/iField.bin' 
                    + ' --parameter ' + rootName + '/results_QSM/parameter.txt'
                    + ' --temp ' + rootName +  '/results_QSM/'
                    + ' --GPU ' + ' --device ' + opt['gpu_id'] 
                    + ' --CSF ' + ' -of QR')
            
            # read .bin files and save into .mat files
            QSM = np.fromfile(rootName+'/results_QSM/recon_QSM_10.bin', 'f4')
            QSM = np.transpose(QSM.reshape([80, 206, nslice]), [2, 1, 0])

            iMag = np.fromfile(rootName+'/results_QSM/iMag.bin', 'f4')
            iMag = np.transpose(iMag.reshape([80, 206, nslice]), [2, 1, 0])

            RDF = np.fromfile(rootName+'/results_QSM/RDF.bin', 'f4')
            RDF = np.transpose(RDF.reshape([80, 206, nslice]), [2, 1, 0])

            R2star = np.fromfile(rootName+'/results_QSM/R2star.bin', 'f4')
            R2star = np.transpose(R2star.reshape([80, 206, nslice]), [2, 1, 0])

            Mask = np.fromfile(rootName+'/results_QSM/Mask.bin', 'f4')
            Mask = np.transpose(Mask.reshape([80, 206, nslice]), [2, 1, 0]) > 0

            adict = {}
            adict['QSM'], adict['iMag'], adict['RDF'] = QSM, iMag, RDF
            adict['R2star'], adict['Mask'] = R2star, Mask
            if opt['lambda1'] == 1:
                sio.savemat(rootName+'/results_ablation2/QSM_bcrnn={}_loupe={}_solver={}_sub={}_.mat' \
                    .format(opt['bcrnn'], opt['loupe'], opt['solver'], opt['test_sub']), adict)
            elif opt['lambda1'] == 0:
                sio.savemat(rootName+'/results_ablation2/QSM_bcrnn={}_loupe={}_solver={}_sub={}_ratio={}.mat' \
                    .format(opt['bcrnn'], opt['loupe'], opt['solver'], opt['test_sub'], opt['samplingRatio']), adict)
            
            
            # # # write into .mat file
            # # Inputs = r2c(np.concatenate(Inputs, axis=0), opt['echo_cat'])
            # # Inputs = np.transpose(Inputs, [0, 2, 3, 1])
            # # Targets = r2c(np.concatenate(Targets, axis=0), opt['echo_cat'])
            # # Targets = np.transpose(Targets, [0, 2, 3, 1])
            # # Recons = r2c(np.concatenate(Recons, axis=0), opt['echo_cat'])
            # # Recons = np.transpose(Recons, [0, 2, 3, 1])

            # # save_mat(rootName+'/results/Inputs.mat', 'Inputs', Inputs)
            # # save_mat(rootName+'/results/Targets.mat', 'Targets', Targets)
            # # save_mat(rootName+'/results/Recons_echo_cat={}_solver={}_K={}_loupe={}.mat' \
            # #   .format(opt['echo_cat'], opt['solver'], opt['K'], opt['loupe']), 'Recons', Recons)


