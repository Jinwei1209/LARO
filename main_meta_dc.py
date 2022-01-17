# PYTHON_ARGCOMPLETE_OK
import os
import time
import torch
import math
import argparse
import random
import numpy as np

from torch.optim.lr_scheduler import MultiStepLR
from torch.utils import data
from loader.kdata_loader_GE import kdata_loader_GE
from loader.real_and_kdata_loader import real_and_kdata_loader
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
from models.meta_resnet_with_dc import Meta_Res_DC
from bayesOpt.sample_loss import gen_pattern
from utils.test import *


if __name__ == '__main__':

    lrG_dc = 1e-3
    niter = 500  # 500 for mask experiment
    batch_size = 1
    display_iters = 2
    lambda_dll2 = 1e-4
    lambda_tv = 1e-4
    rho_penalty = lambda_tv*100
    use_uncertainty = False
    passSigmoid = False
    fixed_mask = False  # +/-
    optimal_mask = False  # +/-
    rescale = True
    Pa = 1.8
    Pbs = np.float32(np.arange(1, 6))
    # shuffle Pbs
    random.seed(1)
    random.shuffle(Pbs)
    print('Pbs = ', Pbs)

    # typein parameters
    parser = argparse.ArgumentParser(description='LOUPE-ST')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--weight_dir', type=str, default='weights_new')
    parser.add_argument('--flag_solver', type=int, default=2)
    parser.add_argument('--contrast', type=str, default='T2')
    parser.add_argument('--K', type=int, default=10) 
    parser.add_argument('--samplingRatio', type=float, default=0.2)  # 0.1/0.2
    # argcomplete.autocomplete(parser)
    opt = {**vars(parser.parse_args())}

    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_id']
    rootName = '/data/Jinwei/{}_slice_recon_GE'.format(opt['contrast'])

    # start here
    t0 = time.time()
    epoch = 0
    gen_iterations = 1
    errL2_dc_sum = 0
    Training_loss, Validation_loss = [], []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataLoader = kdata_loader_GE(
        rootDir=rootName,
        contrast=opt['contrast'], 
        split='train'
        )
    trainLoader = data.DataLoader(dataLoader, batch_size=batch_size, shuffle=True, num_workers=1)

    dataLoader_val = kdata_loader_GE(
        rootDir=rootName,
        contrast=opt['contrast'], 
        split='val'
        )
    valLoader = data.DataLoader(dataLoader_val, batch_size=batch_size, shuffle=True, num_workers=1)

    netG_dc = Meta_Res_DC(
        input_channels=2,
        filter_channels=32,
        lambda_dll2=lambda_dll2,
        lambda_tv=lambda_tv,
        rho_penalty=rho_penalty,
        flag_solver=opt['flag_solver'],
        ncoil=32,
        nrow=256,
        ncol=192,
        K=opt['K'],
        contrast=opt['contrast'],
        samplingRatio=opt['samplingRatio']
    )

    # print(netG_dc)
    netG_dc.to(device)

    # optimizer
    optimizerG_dc = optim.Adam(netG_dc.parameters(), lr=lrG_dc, betas=(0.9, 0.999))
    ms = [0.2, 0.4, 0.6, 0.8]
    ms = [np.floor(m * niter).astype(int) for m in ms]
    # scheduler = MultiStepLR(optimizerG_dc, milestones = ms, gamma = 0.2)

    # # logger
    # logger = Logger(rootName, opt)
    
    while epoch < niter:
        epoch += 1 

        # training phase
        netG_dc.train()
        lossl1 = lossL1()
        metrices_train = Metrices()
        loss_total_list = []
        for idx, (inputs, targets, csms, brain_masks) in enumerate(trainLoader):
            # generate mask
            Pb = Pbs[idx % len(Pbs)]
            pmask_BO = gen_pattern(Pa, Pb)
            Pb = torch.tensor(Pb)[None, ...].to(device)

            if gen_iterations%display_iters == 0:

                print('epochs: [%d/%d], batchs: [%d/%d], time: %ds, K=%d, Solver=%d'
                % (epoch, niter, idx, 300//batch_size+1, time.time()-t0, netG_dc.K, netG_dc.flag_solver))

                if opt['flag_solver'] > 0:
                    print('Lambda_dll2: %f, Sampling ratio cal: %f, Sampling ratio setup: %f'
                        % (netG_dc.lambda_dll2, torch.mean(netG_dc.masks), \
                            netG_dc.samplingRatio))
                else:
                    print('Lambda_tv: %f, Rho_penalty: %f, Sampling ratio cal: %f, Sampling ratio setup: %f'
                        % (netG_dc.lambda_tv, netG_dc.rho_penalty, torch.mean(netG_dc.masks), \
                            netG_dc.samplingRatio))

                print('netG_dc --- loss_L2_dc: %f'
                    % (errL2_dc_sum/display_iters))
                loss_total_list.append(errL2_dc_sum/display_iters)

                print('Average PSNR in Training dataset is %.2f' 
                % (np.mean(np.asarray(metrices_train.PSNRs[-1-display_iters*batch_size:]))))
                if epoch > 1:
                    print('Average PSNR in Validation dataset is %.2f' 
                    % (np.mean(np.asarray(metrices_val.PSNRs))))

                errL2_dc_sum = 0
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            csms = csms.to(device)
            
            # training
            optimizerG_dc.zero_grad()
            Xs = netG_dc(inputs, csms, pmask_BO, Pb)
            lossl2_sum = 0
            for i in range(len(Xs)):
                lossl2_sum += lossl1(Xs[i], targets)
            X = Xs[-1]
            lossl2_sum.backward()
            optimizerG_dc.step()

            # calculating metrices
            errL2_dc_sum += lossl2_sum.item()
            if opt['contrast'] == 'T1':
                metrices_train.get_metrices(X*brain_masks, targets*brain_masks)
            elif opt['contrast'] == 'T2':
                metrices_train.get_metrices(X, targets)

            del(inputs)
            del(targets)
            del(csms)
            del(brain_masks)
            gen_iterations += 1
            
        Training_loss.append(sum(loss_total_list) / float(len(loss_total_list)))

        # scheduler.step(epoch)
        
        # validation phase
        netG_dc.eval()
        metrices_val = Metrices()
        loss_total_list = []
        with torch.no_grad():  # to solve memory exploration issue
            for idx, (inputs, targets, csms, brain_masks) in enumerate(valLoader):
                # generate mask
                Pb = Pbs[idx % len(Pbs)]
                pmask_BO = gen_pattern(Pa, Pb)
                Pb = torch.tensor(Pb)[None, ...].to(device)

                inputs = inputs.to(device)
                targets = targets.to(device)
                csms = csms.to(device)
                brain_masks = brain_masks.to(device)

                # calculating metrices
                Xs = netG_dc(inputs, csms, pmask_BO, Pb)
                metrices_val.get_metrices(Xs[-1]*brain_masks, targets*brain_masks)

                targets = np.asarray(targets.cpu().detach())
                brain_masks = np.asarray(brain_masks.cpu().detach())
                lossl2_sum = loss_unc_sum = 0

                for i in range(len(Xs)):
                    Xs_i = np.asarray(Xs[i].cpu().detach())
                    temp = abs(Xs_i - targets) * brain_masks
                    lossl2_sum += np.mean(temp)
                loss_total_list.append(lossl2_sum)
            print('\n Validation loss: %f \n' 
                % (sum(loss_total_list) / float(len(loss_total_list))))
            Validation_loss.append(sum(loss_total_list) / float(len(loss_total_list)))

        # # save log
        # logger.print_and_save('Epoch: [%d/%d], PSNR in training: %.2f, loss in training: %.10f, Pmask: %f' 
        # % (epoch, niter, np.mean(np.asarray(metrices_train.PSNRs)), Training_loss[-1], torch.mean(netG_dc.Pmask)))
        # logger.print_and_save('Epoch: [%d/%d], PSNR in validation: %.2f, loss in validation: %.10f' 
        # % (epoch, niter, np.mean(np.asarray(metrices_val.PSNRs)), Validation_loss[-1]))

        # save weights
        if Validation_loss[-1] == min(Validation_loss):
            torch.save(netG_dc.state_dict(), rootName+'/{0}/Solver={1}_K={2}_ratio={3}_meta.pt'.format(
                    opt['weight_dir'], opt['flag_solver'], opt['K'], opt['samplingRatio']))

