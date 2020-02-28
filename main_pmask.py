import os
import time
import torch
import math
import argparse
import numpy as np

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
from models.dc_st_pmask import *
from utils.test import *


if __name__ == '__main__':

    lrG_dc = 1e-3
    niter = 500
    batch_size = 1
    display_iters = 10
    lambda_dll2 = 1e-4
    lambda_tv = 1e-4
    rho_penalty = lambda_tv*100
    use_uncertainty = False
    passSigmoid = True
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
    opt = {**vars(parser.parse_args())}

    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_id']
    rootName = '/data/Jinwei/{}_slice_recon_GE'.format(opt['contrast'])

    # # pre-occupy the memory
    # total, used = os.popen(
    #     '"nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader'
    #         ).read().split('\n')[int(opt['gpu_id'])].split(',')
    
    # total = int(total)
    # used = int(used)

    # print('Total memory is {0} MB'.format(total))
    # print('Used memory is {0} MB'.format(used))

    # max_mem = int(total*0.8)
    # block_mem = max_mem - used
    
    # x = torch.rand((256, 1024, block_mem)).cuda()
    # x = torch.rand((2, 2)).cuda()

    # start here
    t0 = time.time()
    epoch = 0
    gen_iterations = 1
    errL2_dc_sum = Pmask_ratio = 0
    PSNRs_val = []
    Validation_loss = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataLoader = kdata_loader_GE(
        rootDir=rootName,
        contrast=opt['contrast'], 
        split='train'
        )
    # dataLoader = real_and_kdata_loader(
    #     rootDir='/data/Jinwei/T2_slice_recon_GE/',
    #     contrast=opt['contrast'], 
    #     split='train'
    #     )
    trainLoader = data.DataLoader(dataLoader, batch_size=batch_size, shuffle=True, num_workers=1)

    dataLoader_val = kdata_loader_GE(
        rootDir=rootName,
        contrast=opt['contrast'], 
        split='val'
        )
    # dataLoader_val = real_and_kdata_loader(
    #     rootDir='/data/Jinwei/T2_slice_recon_GE/',
    #     contrast=opt['contrast'], 
    #     split='val'
    #     )
    valLoader = data.DataLoader(dataLoader_val, batch_size=batch_size, shuffle=True, num_workers=1)
    
    # netG_dc = DC_with_Prop_Mask(
    #     input_channels=2, 
    #     filter_channels=32, 
    #     lambda_dll2=lambda_dll2,
    #     K=K, 
    #     unc_map=use_uncertainty,
    #     fixed_mask=fixed_mask,
    #     rescale=rescale
    # )

    # netG_dc = DC_with_Straight_Through_Pmask(
    #     input_channels=2, 
    #     filter_channels=32, 
    #     lambda_dll2=lambda_dll2,
    #     K=K_model, 
    #     unc_map=use_uncertainty,
    #     passSigmoid=passSigmoid,
    #     fixed_mask=fixed_mask,
    #     optimal_mask=optimal_mask,
    #     rescale=rescale,
    #     samplingRatio=samplingRatio,
    #     contrast=contrast
    # )

    netG_dc = DC_ST_Pmask(
        input_channels=2, 
        filter_channels=8, 
        lambda_dll2=lambda_dll2,
        lambda_tv=lambda_tv,
        rho_penalty=rho_penalty,
        flag_ND=opt['flag_ND'],
        flag_solver=opt['flag_solver'],
        flag_TV=opt['flag_TV'],
        K=opt['K'], 
        unc_map=use_uncertainty,
        passSigmoid=passSigmoid,
        rescale=rescale,
        samplingRatio=opt['samplingRatio'],
        nrow=256,
        ncol=192,
        ncoil=32
    )

    print(netG_dc)
    netG_dc.to(device)

    # # load pre-trained weights with pmask
    # netG_dc.load_state_dict(torch.load(rootName+'/'+folderName+'/weights/{}'.format(math.floor(samplingRatio*100))+
    #             '/weights_ratio_pmask={}%_optimal_ST.pt'.format(math.floor(samplingRatio*100))))
    # netG_dc.eval()
    # print('Load Pmask weights')

    # optimizer
    optimizerG_dc = optim.Adam(netG_dc.parameters(), lr=lrG_dc, betas=(0.9, 0.999))

    # logger
    logger = Logger(rootName, opt)
    
    while epoch < niter:
        epoch += 1 

        # training phase
        netG_dc.train()
        metrices_train = Metrices()
        for idx, (inputs, targets, csms, brain_masks) in enumerate(trainLoader):
            
            if gen_iterations%display_iters == 0:

                print('epochs: [%d/%d], batchs: [%d/%d], time: %ds, K=%d, Solver=%d'
                % (epoch, niter, idx, 300//batch_size+1, time.time()-t0, netG_dc.K, netG_dc.flag_solver))

                if opt['flag_solver'] < 1:
                    print('Lambda_dll2: %f, Sampling ratio cal: %f, Sampling ratio setup: %f, Pmask: %f' 
                        % (netG_dc.lambda_dll2, torch.mean(netG_dc.masks), \
                            netG_dc.samplingRatio, torch.mean(netG_dc.Pmask)))
                elif 0 < opt['flag_solver'] < 3:
                    print('Lambda_tv: %f, Rho_penalty: %f, Sampling ratio cal: %f, Sampling ratio setup: %f, Pmask: %f' 
                        % (netG_dc.lambda_tv, netG_dc.rho_penalty, torch.mean(netG_dc.masks), \
                            netG_dc.samplingRatio, torch.mean(netG_dc.Pmask)))
                else:
                    print('Sampling ratio cal: %f, Sampling ratio setup: %f, Pmask: %f' % (torch.mean(netG_dc.masks), netG_dc.samplingRatio, torch.mean(netG_dc.Pmask)))

                print('netG_dc --- loss_L2_dc: %f'
                    % (errL2_dc_sum/display_iters))

                print('Average PSNR in Training dataset is %.2f' 
                % (np.mean(np.asarray(metrices_train.PSNRs[-1-display_iters*batch_size:]))))
                if epoch > 1:
                    print('Average PSNR in Validation dataset is %.2f' 
                    % (np.mean(np.asarray(metrices_val.PSNRs))))

                errL2_dc_sum = Pmask_ratio = 0
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            csms = csms.to(device)
            brain_masks = brain_masks.to(device)

            errL2_dc, loss_Pmask = netG_dc_train_pmask(
                inputs, 
                targets, 
                csms,
                brain_masks, 
                netG_dc, 
                optimizerG_dc, 
                use_uncertainty,
                lambda_Pmask=0
            )
            errL2_dc_sum += errL2_dc 
            Pmask_ratio += loss_Pmask

            # calculating metrices
            Xs = netG_dc(inputs, csms)
            metrices_train.get_metrices(Xs[-1]*brain_masks, targets*brain_masks)

            gen_iterations += 1
            
        # validation phase
        netG_dc.eval()
        metrices_val = Metrices()
        loss_total_list = []
        with torch.no_grad():  # to solve memory exploration issue
            for idx, (inputs, targets, csms, brain_masks) in enumerate(valLoader):

                inputs = inputs.to(device)
                targets = targets.to(device)
                csms = csms.to(device)
                brain_masks = brain_masks.to(device)

                # calculating metrices
                Xs = netG_dc(inputs, csms)
                metrices_val.get_metrices(Xs[-1]*brain_masks, targets*brain_masks)

                targets = np.asarray(targets.cpu().detach())
                brain_masks = np.asarray(brain_masks.cpu().detach())
                lossl2_sum = loss_unc_sum = 0

                # for i in range(len(Xs)):
                #     Xs_i = np.asarray(Xs[i].cpu().detach())
                #     temp = abs(Xs_i - targets) * brain_masks
                #     lossl2_sum += np.mean(temp)
                Xs_i = np.asarray(Xs[-1].cpu().detach())
                temp = abs(Xs_i - targets) * brain_masks
                lossl2_sum += np.mean(temp)

                temp = np.asarray(netG_dc.Pmask.cpu().detach())
                loss_Pmask = 0*np.mean(temp)
                loss_total = lossl2_sum + loss_Pmask
                loss_total_list.append(loss_total)
            print('\n Validation loss: %f \n' 
                % (sum(loss_total_list) / float(len(loss_total_list))))
            Validation_loss.append(sum(loss_total_list) / float(len(loss_total_list)))

        # save log
        logger.print_and_save('Epoch: [%d/%d], PSNR in training: %.2f, Pmask: %f' 
        % (epoch, niter, np.mean(np.asarray(metrices_train.PSNRs)), torch.mean(netG_dc.Pmask)))
        logger.print_and_save('Epoch: [%d/%d], PSNR in validation: %.2f, loss in validation: %.5f' 
        % (epoch, niter, np.mean(np.asarray(metrices_val.PSNRs)), Validation_loss[-1]))

        # save weights
        if Validation_loss[-1] == min(Validation_loss):
            torch.save(netG_dc.state_dict(), rootName+'/{0}/Solver={1}_K={2}_flag_ND={3}_ratio={4}.pt'.format(
                       opt['weight_dir'], opt['flag_solver'], opt['K'], opt['flag_ND'], opt['samplingRatio']))

