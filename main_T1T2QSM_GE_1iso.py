"""
    Experiment on 0.75*0.75*1.0 T1w+mGRE kspace data reconstruction from GE scanner
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
from loader.kdata_T1T2QSM_CBIC_1iso import kdata_T1T2QSM_CBIC_1iso
from utils.data import r2c, save_mat, save_nii, readcfl, memory_pre_alloc, torch_channel_deconcate, torch_channel_concate, Logger
from utils.loss import lossL1, lossL2, SSIM, snr_gain, CrossEntropyMask, FittingError
from utils.test import Metrices
from utils.operators import Back_forward_MS, backward_MS, forward_MS
from models.resnet_with_dc_t1t2qsm import Resnet_with_DC2
# from models.resnet_with_dc_t1t2qsm_parallel import Resnet_with_DC2
from fits.fits import fit_R2_LM, arlo, fit_complex, fit_T1T2M0
from utils.operators import low_rank_approx, HPphase

if __name__ == '__main__':

    lrG_dc = 1e-3
    batch_size = 1
    display_iters = 10
    gen_iterations = 1
    t0 = time.time()
    epoch = 0
    errL2_dc_sum = 0
    PSNRs_val = []
    Validation_loss = []
    ncoil = 8
    nrow = 386
    ncol = 340
    lambda_dll2 = 1e-3
    TEs = [0.004428,0.009244,0.014060,0.018876,0.023692,0.028508,0.033324,0.038140,0.042956,0.047772,0.052588]

    # parameters in the MP sequence
    TR1 = 6.9
    TR2 = 37.9
    nframes1 = 128
    nframes2 = 128
    alpha1 = 8
    alpha2 = 8
    TE_T2PREP = 57.5
    TD1 = 19.1 - 16.06 + 299
    TD2 = 1000
    num_iter = 2

    # typein parameters
    parser = argparse.ArgumentParser(description='Multi_echo_GE')
    parser.add_argument('--gpu_id', type=str, default='0'), 
    parser.add_argument('--flag_train', type=int, default=1)  # 1 for training, 0 for testing
    parser.add_argument('--test_sub', type=int, default=0)  # 0: iField1, 1: iField2, 2: iField3, 3: iField4
    parser.add_argument('--K', type=int, default=2)  # number of unrolls
    parser.add_argument('--loupe', type=int, default=2)  # -2: fixed learned mask across echos
                                                         # -1: manually designed mask, 0 fixed learned mask, 
                                                         # 1: mask learning, same mask across echos, 2: mask learning, mask for each echo
    parser.add_argument('--bcrnn', type=int, default=1)  # 0: without bcrnn blcok, 1: with bcrnn block, 2: with bclstm block
    parser.add_argument('--solver', type=int, default=1)  # 0 for deep Quasi-newton, 1 for deep ADMM,
                                                          # 2 for TV Quasi-newton, 3 for TV ADMM.
    parser.add_argument('--samplingRatio', type=float, default=0.1)  # Under-sampling ratio
    parser.add_argument('--dataset_id', type=int, default=0)  # 0: new4 of T1w+mGRE+T2w dataset (#1)
    parser.add_argument('--prosp', type=int, default=0)  # flag to test on prospective data
    parser.add_argument('--mc_fusion', type=int, default=1)  # flag to fuse multi-contrast features
    parser.add_argument('--t1w_only', type=int, default=0)  # flag to reconstruct T1w+QSM
    parser.add_argument('--padding', type=int, default=1)  # flag to pad k-space data
    parser.add_argument('--diff_lambdas', type=int, default=0)  # flag to different lambdas for each contrast

    parser.add_argument('--t2w_redesign', type=int, default=0)  # flag to to redesign T2w under-sampling pattern to reduce blur
    parser.add_argument('--flag_unet', type=int, default=1)  # flag to use unet as denoiser
    parser.add_argument('--flag_complex', type=int, default=0)  # flag to use complex convolution
    parser.add_argument('--bn', type=int, default=2)  # flag to use group normalization: 2: use instance normalization    
    parser.add_argument('--necho', type=int, default=11)  # number of echos with kspace data (generalized echoes including T1w and T2w)
    parser.add_argument('--temporal_pred', type=int, default=0)  # flag to use a 2nd recon network with temporal under-sampling
    parser.add_argument('--lambda0', type=float, default=0.0)  # weighting of low rank approximation loss
    parser.add_argument('--rank', type=int, default=0)  #  rank of low rank approximation loss (e.g. 10)
    parser.add_argument('--lambda1', type=float, default=0.0)  # weighting of r2s reconstruction loss
    parser.add_argument('--lambda2', type=float, default=0.0)  # weighting of p1 reconstruction loss
    parser.add_argument('--lambda_maskbce', type=float, default=0.0)  # weighting of Maximal cross entropy in masks
    parser.add_argument('--loss', type=int, default=0)  # 0: SSIM loss, 1: L1 loss, 2: L2 loss
    parser.add_argument('--weights_dir', type=str, default='weights_ablation')
    parser.add_argument('--echo_cat', type=int, default=1)  # flag to concatenate echo dimension into channel
    parser.add_argument('--norm_last', type=int, default=0)  # 0: norm+relu, 1: relu+norm
    parser.add_argument('--temporal_conv', type=int, default=0) # 0: no temporal, 1: center, 2: begining
    parser.add_argument('--1d_type', type=str, default='shear')  # 'shear' or 'random' sampling type of 1D mask
    parser.add_argument('--precond', type=int, default=0)  # flag to use preconsitioning
    parser.add_argument('--att', type=int, default=0)  # flag to use attention-based denoiser
    parser.add_argument('--random', type=int, default=0)  # flag to multiply the input data with a random complex number
    parser.add_argument('--normalizations', type=list, default=[50, 125, 100])  # normalization factors of [mGRE, T1w, T2w] images
                                                                                # default [50, 100, 125] 
                                                       
    opt = {**vars(parser.parse_args())}
    K = opt['K']
    norm_last = opt['norm_last']
    flag_temporal_conv = opt['temporal_conv']
    lambda0 = opt['lambda0']
    lambda1 = opt['lambda1']
    lambda2 = opt['lambda2']
    lambda_maskbce = opt['lambda_maskbce']  # 0.01 too large
    rank = opt['rank']
    opt['necho'] = 11
    necho = opt['necho']
    necho_pred = 0
    # concatenate echo dimension to the channel dimension for TV regularization
    if opt['solver'] > 1:
        opt['echo_cat'] = 1
    if opt['loupe'] > 0:
        niter = 2000
    else:
        niter = 100
    
    if opt['padding'] == 0:
        nrow = 206
        ncol = 160
    else:
        nrow = 350
        ncol = 290

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
    rootName = '/data2/Jinwei/T1T2QSM'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # torch.manual_seed(0)

    if opt['loupe'] == -2:
        # load fixed loupe optimized mask across echos
        masks = np.real(readcfl(rootName+'/masks{}/mask_{}_echo'.format(opt['dataset_id']+1, opt['samplingRatio'])))
        masks = masks[..., np.newaxis] # (necho, nrow, ncol, 1)
        if opt['padding'] == 1:
            masks = np.pad(masks, ((0, 0), (72, 72), (65, 65), (0, 0)))
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

    if opt['t2w_redesign'] == 1 and opt['loupe'] != 2:
        # load fixed loupe optimized mask across echos
        masks = np.real(readcfl(rootName+'/masks{}/t2redesign=1/mask_{}_echo'.format(opt['dataset_id']+1, opt['samplingRatio'])))
        masks = masks[..., np.newaxis] # (necho, nrow, ncol, 1)
        masks = torch.tensor(masks, device=device).float()
        # to complex data
        masks = torch.cat((masks, torch.zeros(masks.shape).to(device)),-1) # (necho, nrow, ncol, 2)
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

        dataLoader = kdata_T1T2QSM_CBIC_1iso(
            rootDir=rootName,
            contrast='MultiContrast', 
            split='train',
            dataset_id=opt['dataset_id'],
            padding_flag=opt['padding'],
            normalizations=opt['normalizations'],
            echo_cat=opt['echo_cat']
        )
        trainLoader = data.DataLoader(dataLoader, batch_size=batch_size, shuffle=True, num_workers=1)

        dataLoader_val = kdata_T1T2QSM_CBIC_1iso(
            rootDir=rootName,
            contrast='MultiContrast', 
            split='val',
            dataset_id=opt['dataset_id'],
            padding_flag=opt['padding'],
            normalizations=opt['normalizations'],
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
                nrow=nrow,
                ncol=ncol,
                ncoil=ncoil,
                K=K,
                rank=opt['rank'],
                echo_cat=1,
                flag_solver=opt['solver'],
                flag_precond=opt['precond'],
                flag_loupe=opt['loupe'],
                flag_temporal_pred=0,
                samplingRatio=opt['samplingRatio'],
                norm_last=norm_last,
                flag_temporal_conv=flag_temporal_conv,
                flag_BCRNN=flag_bcrnn,
                flag_hidden=flag_hidden,
                flag_unet=opt['flag_unet'],
                flag_att=opt['att'],
                flag_cp=1,
                flag_dataset=1,
                flag_mc_fusion=opt['mc_fusion'],
                flag_t2w_redesign=opt['t2w_redesign'],
                flag_t1w_only=opt['t1w_only'],
                flag_diff_lambdas=opt['diff_lambdas']
            )
        else:
            netG_dc = Resnet_with_DC2(
                input_channels=2,
                filter_channels=32,
                lambda_dll2=lambda_dll2,
                nrow=nrow,
                ncol=ncol,
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
            weights_dict = torch.load(rootName+'/'+opt['weights_dir']+'/bcrnn={}_diff_lambdas={}_K=1_loupe=1_ratio={}_solver={}_mc_fusion={}_dataset={}_padding={}.pt'
                        .format(opt['bcrnn'], opt['diff_lambdas'], opt['samplingRatio'], opt['solver'], opt['mc_fusion'], opt['dataset_id'], opt['padding']))
            netG_dc.load_state_dict(weights_dict)
        elif opt['loupe'] == -2:
            weights_dict = torch.load(rootName+'/'+opt['weights_dir']+'/bcrnn={}_diff_lambdas={}_K=1_loupe=2_ratio={}_solver={}_mc_fusion={}_dataset={}_padding={}_last.pt'
                        .format(opt['bcrnn'], opt['diff_lambdas'], opt['samplingRatio'], opt['solver'], opt['mc_fusion'], opt['dataset_id'], opt['padding']))
            netG_dc.load_state_dict(weights_dict)

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
            for idx, (kdatas, targets, csms, brain_masks) in enumerate(trainLoader):
                kdatas = kdatas[:, :, :necho, ...]  # temporal undersampling
                csms = csms[:, :, :necho, ...]  # temporal undersampling
                if opt['temporal_pred'] == 0:
                    targets = targets[:, :2*necho, ...]  # temporal undersampling
                    brain_masks = brain_masks[:, :2*necho, ...]  # temporal undersampling

                if torch.sum(brain_masks) == 0:
                    continue

                if gen_iterations%display_iters == 0:

                    print('epochs: [%d/%d], batchs: [%d/%d], time: %ds'
                    % (epoch, niter, idx, dataLoader.nsamples//batch_size, time.time()-t0))

                    print('bcrnn: {}, loss: {}, K: {}, loupe: {}, solver: {}, rank: {}'.format( \
                            opt['bcrnn'], opt['loss'], opt['K'], opt['loupe'], opt['solver'], opt['rank']))
                    
                    if opt['loupe'] > 0:
                        print('Sampling ratio cal: %f, Sampling ratio setup: %f, Pmask: %f' 
                        % (torch.mean(netG_dc.Mask), netG_dc.samplingRatio, torch.mean(netG_dc.Pmask)))
                    else:
                        print('Sampling ratio cal: %f' % (torch.mean(netG_dc.Mask)))

                    if opt['solver'] < 3:
                        if opt['diff_lambdas'] == 0:
                            print('netG_dc --- loss_L2_dc: %f, lambda_dll2: %f, lambda_lowrank: %f'
                                % (errL2_dc_sum/display_iters, netG_dc.lambda_dll2, netG_dc.lambda_lowrank))
                        elif opt['diff_lambdas'] == 1:
                            print('netG_dc --- loss_L2_dc: %f, lambda_dll2_mGRE: %f, lambda_dll2_t1w: %f, lambda_dll2_t2w: %f'
                                % (errL2_dc_sum/display_iters, netG_dc.lambda_dll2[0], netG_dc.lambda_dll2[1], netG_dc.lambda_dll2[2]))
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
                csms = csms.to(device)
                brain_masks = brain_masks.to(device)

                optimizerG_dc.zero_grad()
                if opt['temporal_pred'] == 1:
                    Xs = netG_dc(kdatas, csms, None, masks, flip, x_input=None)
                else:
                    Xs = netG_dc(kdatas, csms, None, masks, flip)

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
                        # ssim loss
                        lossl2_sum -= loss(Xs[i]*brain_masks, targets*brain_masks)
                    elif opt['loss'] > 0:
                        # L1 or L2 loss
                        lossl2_sum += loss(Xs[i]*brain_masks, targets*brain_masks)

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
            # with torch.no_grad():  # to solve memory exploration issue
            #     for idx, (kdatas, targets, csms, brain_masks) in enumerate(valLoader):
            #         kdatas = kdatas[:, :, :necho, ...]  # temporal undersampling
            #         csms = csms[:, :, :necho, ...]  # temporal undersampling
            #         if opt['temporal_pred'] == 0:
            #             targets = targets[:, :2*necho, ...]  # temporal undersampling
            #             brain_masks = brain_masks[:, :2*necho, ...]  # temporal undersampling

            #         if torch.sum(brain_masks) == 0:
            #             continue

            #         kdatas = kdatas.to(device)
            #         targets = targets.to(device)
            #         csms = csms.to(device)
            #         brain_masks = brain_masks.to(device)

            #         if opt['temporal_pred'] == 1:
            #             Xs = netG_dc(kdatas, csms, None, masks, flip, x_input=None)
            #         else:
            #             Xs = netG_dc(kdatas, csms, None, masks, flip)

            #         metrices_val.get_metrices(Xs[-1]*brain_masks, targets*brain_masks)
            #         lossl2_sum = loss(Xs[-1]*brain_masks, targets*brain_masks)
            #         loss_total_list.append(lossl2_sum)

            #     print('\n Validation loss: %f \n' 
            #         % (sum(loss_total_list) / float(len(loss_total_list))))
            #     Validation_loss.append(sum(loss_total_list) / float(len(loss_total_list)))
            #     PSNRs_val.append(np.mean(np.asarray(metrices_val.PSNRs)))
            
            # # save log
            # logger.print_and_save('Epoch: [%d/%d], PSNR in training: %.2f' 
            # % (epoch, niter, np.mean(np.asarray(metrices_train.PSNRs))))
            # logger.print_and_save('Epoch: [%d/%d], PSNR in validation: %.2f, loss in validation: %.10f' 
            # % (epoch, niter, np.mean(np.asarray(metrices_val.PSNRs)), Validation_loss[-1]))

            # # save weights
            # if PSNRs_val[-1] == max(PSNRs_val):
            #     torch.save(netG_dc.state_dict(), rootName+'/'+opt['weights_dir']+'/bcrnn={}_diff_lambdas={}_K={}_loupe={}_ratio={}_solver={}_mc_fusion={}_dataset={}_padding={}.pt' \
            #     .format(opt['bcrnn'], opt['diff_lambdas'], opt['K'], opt['loupe'], opt['samplingRatio'], opt['solver'], opt['mc_fusion'], opt['dataset_id'], opt['padding']))
            torch.save(netG_dc.state_dict(), rootName+'/'+opt['weights_dir']+'/bcrnn={}_diff_lambdas={}_K={}_loupe={}_ratio={}_solver={}_mc_fusion={}_dataset={}_padding={}_last.pt' \
            .format(opt['bcrnn'], opt['diff_lambdas'], opt['K'], opt['loupe'], opt['samplingRatio'], opt['solver'], opt['mc_fusion'], opt['dataset_id'], opt['padding']))
    
    
    # for test
    lossl2 = lossL2()
    if opt['flag_train'] == 0:
        if opt['echo_cat'] == 1:
            netG_dc = Resnet_with_DC2(
                input_channels=2*necho,
                filter_channels=32*necho,
                necho=necho,
                necho_pred=necho_pred,
                lambda_dll2=lambda_dll2,
                nrow=nrow,
                ncol=ncol,
                ncoil=ncoil,
                K=K,
                rank=opt['rank'],
                echo_cat=1,
                flag_solver=opt['solver'],
                flag_precond=opt['precond'],
                flag_loupe=opt['loupe'],
                flag_temporal_pred=0,
                samplingRatio=opt['samplingRatio'],
                norm_last=norm_last,
                flag_temporal_conv=flag_temporal_conv,
                flag_BCRNN=flag_bcrnn,
                flag_hidden=flag_hidden,
                flag_unet=opt['flag_unet'],
                flag_att=opt['att'],
                flag_dataset=1,
                flag_mc_fusion=opt['mc_fusion'],
                flag_t2w_redesign=opt['t2w_redesign'],
                flag_t1w_only=opt['t1w_only'],
                flag_diff_lambdas=opt['diff_lambdas']
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
        weights_dict = torch.load(rootName+'/'+opt['weights_dir']+'/bcrnn={}_diff_lambdas={}_K={}_loupe={}_ratio={}_solver={}_mc_fusion={}_dataset={}_padding={}_last.pt' \
                    .format(opt['bcrnn'], opt['diff_lambdas'], opt['K'], opt['loupe'], opt['samplingRatio'], opt['solver'], opt['mc_fusion'], opt['dataset_id'], opt['padding']))
        netG_dc.load_state_dict(weights_dict)
        netG_dc.to(device)
        netG_dc.eval()

        Inputs = []
        Targets = []
        R2s, R2s_target = [], []
        water, water_target = [], []
        Recons = []
        preconds = []
        T1, M0 = [], []

        dataLoader_test = kdata_T1T2QSM_CBIC_1iso(
            rootDir=rootName,
            contrast='MultiContrast', 
            split='test',
            dataset_id=opt['dataset_id'],
            prosp_flag=opt['prosp'],
            padding_flag=opt['padding'],
            subject=opt['test_sub'],
            normalizations=opt['normalizations'],
            echo_cat=opt['echo_cat']
        )
        testLoader = data.DataLoader(dataLoader_test, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for idx, (kdatas, targets, csms, brain_masks) in enumerate(testLoader):
                kdatas = kdatas[:, :, :necho, ...]  # temporal undersampling
                csms = csms[:, :, :necho, ...]  # temporal undersampling

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
                brain_masks = brain_masks.to(device)

                if opt['temporal_pred'] == 1:
                    Xs_1 = netG_dc(kdatas, csms, None, masks, flip, x_input=None)[-1]
                else:
                    Xs_1 = netG_dc(kdatas, csms, None, masks, flip)[-1]
                precond = netG_dc.precond
                if opt['echo_cat']:
                    targets = torch_channel_deconcate(targets)
                    Xs_1 = torch_channel_deconcate(Xs_1)
                    mags_target = torch.sqrt(targets[:, 0, ...]**2 + targets[:, 1, ...]**2).permute(0, 2, 3, 1)

                # fit_T1T2M0_model = fit_T1T2M0(mags_target, TR1, TR2, nframes1, nframes2, alpha1, alpha2, TE_T2PREP, TD1, TD2, num_iter)
                # optimizer_fit = torch.optim.Adam(fit_T1T2M0_model.parameters(), lr=1e-1, betas=(0.9, 0.999))
                # for i in range(num_iter):
                #     optimizer_fit.zero_grad()
                #     [M2_pred, M3_pred] = fit_T1T2M0_model(mags_target[..., -2], mags_target[..., 0], mags_target[..., -1])
                #     loss = lossl2(M2_pred, mags_target[..., -2]) + lossl2(M3_pred, mags_target[..., 0])
                #     loss.backward()
                #     optimizer_fit.step()
                #     print("iter: {}, loss: {}", i, loss.item())
                
                # T1.append(fit_T1T2M0_model.T1.cpu().detach())
                # M0.append(fit_T1T2M0_model.M0.cpu().detach())
                Targets.append(targets.cpu().detach())
                Recons.append(Xs_1.cpu().detach())

            # write into .mat file
            Recons_ = np.squeeze(r2c(np.concatenate(Recons, axis=0), opt['echo_cat']))
            Recons_ = np.transpose(Recons_, [0, 2, 3, 1])
            save_mat(rootName+'/results/T1w_bcrnn={}_loupe={}_solver={}_sub={}_ratio={}.mat' \
                .format(opt['bcrnn'], opt['loupe'], opt['solver'], opt['test_sub'], opt['samplingRatio']), 'Recons', Recons_[..., necho-2:])

            # # write T1 into .mat file
            # T1 = np.squeeze(np.concatenate(T1, axis=0))
            # save_mat(rootName+'/results/T1_bcrnn={}_loupe={}_solver={}_sub={}_ratio={}.mat' \
            #     .format(opt['bcrnn'], opt['loupe'], opt['solver'], opt['test_sub'], opt['samplingRatio']), 'T1', T1)

            # # write M0 into .mat file
            # M0 = np.squeeze(np.concatenate(M0, axis=0))
            # save_mat(rootName+'/results/M0_bcrnn={}_loupe={}_solver={}_sub={}_ratio={}.mat' \
            #     .format(opt['bcrnn'], opt['loupe'], opt['solver'], opt['test_sub'], opt['samplingRatio']), 'M0', M0)

            # write into .bin file
            # (nslice, 2, necho, nrow, ncol) to (ncol, nrow, nslice, necho, 2)
            iField = np.transpose(np.concatenate(Recons, axis=0), [4, 3, 0, 2, 1])
            nslice = iField.shape[2]
            iField[..., 1] = - iField[..., 1]
            iField = iField[:, :, :, :necho-2, :]
            print('iField size is: ', iField.shape)
            if os.path.exists(rootName+'/results_QSM/iField.bin'):
                os.remove(rootName+'/results_QSM/iField.bin')
            iField.tofile(rootName+'/results_QSM/iField.bin')
            print('Successfully save iField.bin')

            torch.cuda.empty_cache()

            # run MEDIN
            if opt['padding'] == 1:
                os.system('medi ' + rootName + '/results_QSM/iField.bin' 
                        + ' --parameter ' + rootName + '/results_QSM/parameter_1iso.txt'
                        + ' --temp ' + rootName +  '/results_QSM/'
                        + ' --GPU ' + ' --device ' + opt['gpu_id'] 
                        + ' --CSF ' + ' -of QR  -rl 0.45')
            else:
                os.system('medi ' + rootName + '/results_QSM/iField.bin' 
                        + ' --parameter ' + rootName + '/results_QSM/parameter_ori.txt'
                        + ' --temp ' + rootName +  '/results_QSM/'
                        + ' --GPU ' + ' --device ' + opt['gpu_id'] 
                        + ' --CSF ' + ' -of QR  -rl 0.6')
            
            # read .bin files and save into .mat files
            QSM = np.fromfile(rootName+'/results_QSM/recon_QSM_09.bin', 'f4')
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
            sio.savemat(rootName+'/results/QSM_bcrnn={}_loupe={}_solver={}_sub={}_ratio={}.mat' \
                .format(opt['bcrnn'], opt['loupe'], opt['solver'], opt['test_sub'], opt['samplingRatio']), adict)


