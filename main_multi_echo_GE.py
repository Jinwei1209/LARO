"""
    Experiment on multi-echo kspace data reconstruction from GE scanner
"""
import os
import time
import torch
import math
import argparse
import numpy as np

from torch.utils import data
from loader.kdata_multi_echo_GE import kdata_multi_echo_GE
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
from models.resnet_with_dc import *
from utils.test import *
from utils.operators import *

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
    ncoil = 12
    nrow = 206
    ncol = 80
    necho = 10
    K = 5
    lambda_dll2 = 0.001
    
    # typein parameters
    parser = argparse.ArgumentParser(description='CardiacQSM')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--flag_train', type=int, default=1)  # 1 for training, 0 for testing
    parser.add_argument('--flag_model', type=int, default=0)  # 0 for vanilla MoDL
    parser.add_argument('--normalization', type=int, default=1)  # 0 for no normalization
    parser.add_argument('--echo_cat', type=int, default=1)  # flag to concatenate echo dimension into channel
    opt = {**vars(parser.parse_args())}

    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_id']
    rootName = '/data/Jinwei/Multi_echo_slice_recon_GE'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load mask
    masks = np.real(readcfl(rootName+'/megre_slice_GE/mask'))
    # masks = np.ones(masks.shape)
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

    # training
    if opt['flag_train'] == 1:
        lossl1 = lossL1()

        dataLoader = kdata_multi_echo_GE(
            rootDir=rootName,
            contrast='MultiEcho', 
            split='train',
            normalization=opt['normalization'],
            echo_cat=opt['echo_cat']
        )
        trainLoader = data.DataLoader(dataLoader, batch_size=batch_size, shuffle=True, num_workers=1)

        dataLoader_val = kdata_multi_echo_GE(
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
                lambda_dll2=lambda_dll2,
                K=K,
                echo_cat=1
            )
        else:
            netG_dc = Resnet_with_DC2(
                input_channels=2,
                filter_channels=32,
                lambda_dll2=lambda_dll2,
                K=K,
                echo_cat=0
            )
        netG_dc.to(device)

        # optimizer
        optimizerG_dc = optim.Adam(netG_dc.parameters(), lr=lrG_dc, betas=(0.9, 0.999))

        while epoch < niter:

            epoch += 1

            # training phase
            netG_dc.train()
            metrices_train = Metrices()
            for idx, (kdatas, targets, csms, brain_masks) in enumerate(trainLoader):
                # print(idx)

                # kdatas = kdatas[:, :, :necho, ...]
                # csms = csms[:, :, :necho, ...]
                # targets = targets[:, :2*necho, ...]
                # brain_masks = brain_masks[:, :2*necho, ...]

                if torch.sum(brain_masks) == 0:
                    continue

                if gen_iterations%display_iters == 0:

                    print('epochs: [%d/%d], batchs: [%d/%d], time: %ds'
                    % (epoch, niter, idx, 600//batch_size, time.time()-t0))

                    print('netG_dc --- loss_L2_dc: %f, lambda_dll2: %f, K: %d, necho: %d'
                        % (errL2_dc_sum/display_iters, netG_dc.lambda_dll2, K, necho))

                    print('Average PSNR in Training dataset is %.2f' 
                    % (np.mean(np.asarray(metrices_train.PSNRs[-1-display_iters*batch_size:]))))
                    if epoch > 1:
                        print('Average PSNR in Validation dataset is %.2f' 
                        % (np.mean(np.asarray(metrices_val.PSNRs))))

                    errL2_dc_sum = 0
                
                kdatas = kdatas.to(device)
                targets = targets.to(device)
                csms = csms.to(device)
                brain_masks = brain_masks.to(device)

                # operator = Back_forward_multiEcho(csms, masks, 0)
                # test_image = operator.AtA(targets, 0).cpu().detach().numpy()
                # save_mat(rootName+'/results/test_image.mat', 'test_image', test_image)


                inputs = backward_multiEcho(kdatas, csms, masks, flip, 
                                            opt['echo_cat'])
                optimizerG_dc.zero_grad()
                Xs = netG_dc(inputs, csms, masks, flip)

                lossl2_sum = 0
                for i in range(len(Xs)):
                    lossl2_sum += lossl1(Xs[i]*brain_masks, targets*brain_masks)
                lossl2_sum.backward()
                optimizerG_dc.step()

                errL2_dc_sum += lossl2_sum.item()

                # calculating metrices
                metrices_train.get_metrices(Xs[-1]*brain_masks, targets*brain_masks)
                gen_iterations += 1

            
            # validation phase
            netG_dc.eval()
            metrices_val = Metrices()
            loss_total_list = []
            with torch.no_grad():  # to solve memory exploration issue
                for idx, (kdatas, targets, csms, brain_masks) in enumerate(valLoader):
                    # kdatas = kdatas[:, :, :necho, ...]
                    # csms = csms[:, :, :necho, ...]
                    # targets = targets[:, :2*necho, ...]
                    # brain_masks = brain_masks[:, :2*necho, ...]

                    if torch.sum(brain_masks) == 0:
                        continue

                    kdatas = kdatas.to(device)
                    targets = targets.to(device)
                    csms = csms.to(device)
                    brain_masks = brain_masks.to(device)

                    inputs = backward_multiEcho(kdatas, csms, masks, flip,
                                                opt['echo_cat'])
                    Xs = netG_dc(inputs, csms, masks, flip)

                    metrices_val.get_metrices(Xs[-1]*brain_masks, targets*brain_masks)
                    targets = np.asarray(targets.cpu().detach())
                    brain_masks = np.asarray(brain_masks.cpu().detach())
                    temp = 0
                    for i in range(len(Xs)):
                        X = np.asarray(Xs[i].cpu().detach())
                        temp += abs(X - targets) * brain_masks
                    lossl2_sum = np.mean(temp)
                    loss_total_list.append(lossl2_sum)

                print('\n Validation loss: %f \n' 
                    % (sum(loss_total_list) / float(len(loss_total_list))))
                Validation_loss.append(sum(loss_total_list) / float(len(loss_total_list)))

            # save weights
            if Validation_loss[-1] == min(Validation_loss):
                if opt['echo_cat'] == 1:
                    torch.save(netG_dc.state_dict(), rootName+'/weights/weight_MoDL2.pt')
                else:
                    torch.save(netG_dc.state_dict(), rootName+'/weights/weight_MoDL_3D.pt')
    
    # for test
    if opt['flag_train'] == 0:
        if opt['echo_cat'] == 1:
            netG_dc = Resnet_with_DC2(
                input_channels=2*necho,
                filter_channels=32*necho,
                lambda_dll2=lambda_dll2,
                K=K,
                echo_cat=1
            )
            weights_dict = torch.load(rootName+'/weights/weight_MoDL2.pt')
        else:
            netG_dc = Resnet_with_DC2(
                input_channels=2,
                filter_channels=32,
                lambda_dll2=lambda_dll2,
                K=K,
                echo_cat=0
            )
            weights_dict = torch.load(rootName+'/weights/weight_MoDL_3D.pt')
        netG_dc.to(device)
        netG_dc.load_state_dict(weights_dict)
        netG_dc.eval()

        Inputs = []
        Targets = []
        Recons = []

        dataLoader_test = kdata_multi_echo_GE(
            rootDir=rootName,
            contrast='MultiEcho', 
            split='test',
            normalization=opt['normalization'],
            echo_cat=opt['echo_cat']
        )
        testLoader = data.DataLoader(dataLoader_test, batch_size=batch_size, shuffle=False)

        for idx, (kdatas, targets, csms, brain_masks) in enumerate(testLoader):
            print(idx)
            kdatas = kdatas[:, :, :necho, ...]
            csms = csms[:, :, :necho, ...]
            targets = targets[:, :2*necho, ...]
            brain_masks = brain_masks[:, :2*necho, ...]

            kdatas = kdatas.to(device)
            targets = targets.to(device)
            csms = csms.to(device)
            brain_masks = brain_masks.to(device)

            inputs = backward_multiEcho(kdatas, csms, masks, flip,
                                        opt['echo_cat'])
            Xs = netG_dc(inputs, csms, masks, flip)

            Inputs.append(inputs.cpu().detach())
            Targets.append(targets.cpu().detach())
            Recons.append(Xs[-1].cpu().detach())

        Inputs = r2c(np.concatenate(Inputs, axis=0), opt['echo_cat'])
        Inputs = np.transpose(Inputs, [0, 2, 3, 1])
        Targets = r2c(np.concatenate(Targets, axis=0), opt['echo_cat'])
        Targets = np.transpose(Targets, [0, 2, 3, 1])
        Recons = r2c(np.concatenate(Recons, axis=0), opt['echo_cat'])
        Recons = np.transpose(Recons, [0, 2, 3, 1])

        save_mat(rootName+'/results/Inputs.mat', 'Inputs', Inputs)
        save_mat(rootName+'/results/Targets.mat', 'Targets', Targets)
        save_mat(rootName+'/results/Recons_echo_cat={}.mat'.format( \
                opt['echo_cat']), 'Recons', Recons)


