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
from models.resnet_with_dc import *
from utils.test import *

if __name__ == '__main__':

    lrG_dc = 1e-3
    niter = 3000
    batch_size = 4
    display_iters = 10
    gen_iterations = 1
    t0 = time.time()
    epoch = 0
    errL2_dc_sum = 0
    PSNRs_val = []
    Validation_loss = []
    ncoil = 16
    nrow = 192
    ncol = 140
    
    # typein parameters
    parser = argparse.ArgumentParser(description='CardiacQSM')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--flag_train', type=int, default=0)  # 0 for training, 1 for testing, 2 for FINE.
    parser.add_argument('--flag_model', type=int, default=0)  # 0 for Unet, 1 for MoDL.
    opt = {**vars(parser.parse_args())}

    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_id']
    rootName = '/data/Jinwei/T2_slice_recon_GE'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load mask
    masks = load_mat(rootName+'/Total_slices_CardiacQSM/Mask_CardiacQSM.mat', 'Mask')  # LOUPE/VD/Adjoint
    masks = masks[np.newaxis, ..., np.newaxis]
    masks = torch.tensor(masks, device=device).float()
    # to complex data
    masks = torch.cat((masks, torch.zeros(masks.shape).to(device)),-1)
    # add coil dimension
    masks = torch.cat(ncoil*[masks])
    masks = masks[None, ...]

    # flip matrix
    flip = torch.ones([nrow, ncol, 1]) 
    flip = torch.cat((flip, torch.zeros(flip.shape)), -1).to(device)
    flip[::2, ...] = - flip[::2, ...] 
    flip[:, ::2, ...] = - flip[:, ::2, ...]

    # for Unet training
    if opt['flag_train'] == 0:
        lossl1 = lossL1()

        dataLoader = kdata_loader_GE(
            rootDir=rootName,
            contrast='CardiacQSM', 
            split='train'
            )
        trainLoader = data.DataLoader(dataLoader, batch_size=batch_size, shuffle=True, num_workers=1)

        dataLoader_val = kdata_loader_GE(
            rootDir=rootName,
            contrast='CardiacQSM', 
            split='val'
            )
        valLoader = data.DataLoader(dataLoader_val, batch_size=batch_size, shuffle=True, num_workers=1)

        if opt['flag_model'] == 0:
            netG_dc = Unet(
                input_channels=2, 
                output_channels=2, 
                num_filters=[2**i for i in range(5, 10)]
            )
        elif opt['flag_model'] == 1:
            netG_dc = Resnet_with_DC(
                input_channels=2,
                filter_channels=32,
                lambda_dll2=0.001,
                K=5
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
                
                if gen_iterations%display_iters == 0:

                    print('epochs: [%d/%d], batchs: [%d/%d], time: %ds'
                    % (epoch, niter, idx, 90//batch_size+1, time.time()-t0))

                    print('netG_dc --- loss_L2_dc: %f'
                        % (errL2_dc_sum/display_iters))

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

                inputs = backward_CardiacQSM(kdatas, csms, masks, flip)
                inputs = inputs.to(device)

                optimizerG_dc.zero_grad()
                if opt['flag_model'] == 0:
                    X = netG_dc(inputs)
                elif opt['flag_model'] == 1:
                    X = netG_dc(inputs, csms, masks)

                lossl2_sum = lossl1(X*brain_masks, targets*brain_masks)
                lossl2_sum.backward()
                optimizerG_dc.step()

                errL2_dc_sum += lossl2_sum.item()

                # calculating metrices
                if opt['flag_model'] == 0:
                    X = netG_dc(inputs)
                elif opt['flag_model'] == 1:
                    X = netG_dc(inputs, csms, masks)
                metrices_train.get_metrices(X*brain_masks, targets*brain_masks)
                gen_iterations += 1

            
            # validation phase
            netG_dc.eval()
            metrices_val = Metrices()
            loss_total_list = []
            with torch.no_grad():  # to solve memory exploration issue
                for idx, (kdatas, targets, csms, brain_masks) in enumerate(valLoader):

                    kdatas = kdatas.to(device)
                    targets = targets.to(device)
                    csms = csms.to(device)
                    brain_masks = brain_masks.to(device)

                    inputs = backward_CardiacQSM(kdatas, csms, masks, flip)
                    inputs = inputs.to(device)

                    # calculating metrices
                    if opt['flag_model'] == 0:
                        X = netG_dc(inputs)
                    elif opt['flag_model'] == 1:
                        X = netG_dc(inputs, csms, masks)

                    metrices_val.get_metrices(X*brain_masks, targets*brain_masks)
                    targets = np.asarray(targets.cpu().detach())
                    brain_masks = np.asarray(brain_masks.cpu().detach())
                    X = np.asarray(X.cpu().detach())
                    temp = abs(X - targets) * brain_masks
                    lossl2_sum = np.mean(temp)
                    loss_total_list.append(lossl2_sum)

                print('\n Validation loss: %f \n' 
                    % (sum(loss_total_list) / float(len(loss_total_list))))
                Validation_loss.append(sum(loss_total_list) / float(len(loss_total_list)))

            # save weights
            if Validation_loss[-1] == min(Validation_loss):
                if opt['flag_model'] == 0:
                    torch.save(netG_dc.state_dict(), rootName+'/weights/weight_CardiacQSM.pt')
                elif opt['flag_model'] == 1:
                    torch.save(netG_dc.state_dict(), rootName+'/weights/weight_CardiacQSM_MoDL.pt')
    

    # for Unet test
    if opt['flag_train'] == 1:
        if opt['flag_model'] == 0:
            netG_dc = Unet(
                input_channels=2, 
                output_channels=2, 
                num_filters=[2**i for i in range(5, 10)]
            )
        elif opt['flag_model'] == 1:
            netG_dc = Resnet_with_DC(
                input_channels=2,
                filter_channels=32,
                lambda_dll2=0.001,
                K=5
            )
        netG_dc.to(device)
        weights_dict = torch.load(rootName+'/weights/weight_CardiacQSM.pt')
        netG_dc.load_state_dict(weights_dict)
        netG_dc.eval()

        Inputs = []
        Targets = []
        Recons = []

        dataLoader_test = kdata_loader_GE(
        rootDir=rootName,
        contrast='CardiacQSM', 
        split='test'
        )
        testLoader = data.DataLoader(dataLoader_test, batch_size=batch_size, shuffle=False)

        for idx, (kdatas, targets, csms, brain_masks) in enumerate(testLoader):
            print(idx)

            kdatas = kdatas.to(device)
            targets = targets.to(device)
            csms = csms.to(device)
            brain_masks = brain_masks.to(device)

            inputs = backward_CardiacQSM(kdatas, csms, masks, flip)
            inputs = inputs.to(device)
            if opt['flag_model'] == 0:
                X = netG_dc(inputs)
            elif opt['flag_model'] == 1:
                X = netG_dc(inputs, csms, masks)

            Inputs.append(inputs.cpu().detach())
            Targets.append(targets.cpu().detach())
            Recons.append(X.cpu().detach())

        Inputs = r2c(np.concatenate(Inputs, axis=0))
        Inputs = np.transpose(Inputs, [1,2,0])
        Targets = r2c(np.concatenate(Targets, axis=0))
        Targets = np.transpose(Targets, [1,2,0])
        Recons = r2c(np.concatenate(Recons, axis=0))
        Recons = np.transpose(Recons, [1,2,0])

        adict = {}
        adict['Inputs'] = Inputs
        sio.savemat(rootName+'/results/Inputs_CardiacQSM.mat', adict)

        adict = {}
        adict['Targets'] = Targets
        sio.savemat(rootName+'/results/Targets_CardiacQSM.mat', adict)

        adict = {}
        adict['Recons'] = Recons
        sio.savemat(rootName+'/results/Recons_CardiacQSM.mat', adict)

    # for Unet FINE
    if opt['flag_train'] == 2:
        lossl2 = lossL2()

        netG_dc = Unet(
            input_channels=2, 
            output_channels=2, 
            num_filters=[2**i for i in range(5, 10)]
        )
        netG_dc.to(device)
        weights_dict = torch.load(rootName+'/weights/weight_CardiacQSM.pt')
        netG_dc.load_state_dict(weights_dict)
        netG_dc.eval()

        Kunders = []
        FINEs = []

        dataLoader_test = kdata_loader_GE(
            rootDir=rootName,
            contrast='CardiacQSM', 
            split='test'
        )
        testLoader = data.DataLoader(dataLoader_test, batch_size=batch_size, shuffle=False)

        # optimizer
        optimizerG_dc = optim.Adam(netG_dc.parameters(), lr=lrG_dc, betas=(0.9, 0.999))

        for idx, (kdatas, targets, csms, brain_masks) in enumerate(testLoader):

            kdatas = kdatas.to(device)
            targets = targets.to(device)
            csms = csms.to(device)
            brain_masks = brain_masks.to(device)

            for i in range(1000):

                optimizerG_dc.zero_grad()

                inputs = backward_CardiacQSM(kdatas, csms, masks, flip)
                X = netG_dc(inputs)
                K_measured = cplx_mlpy(kdatas, masks)
                kunder = forward_CardiacQSM(X, csms, masks, flip)
                kspace_loss = lossl2(K_measured, kunder)

                kspace_loss.backward()

                optimizerG_dc.step()
                
                if i % 10 == 0:
                    print('Fidelity loss: {0}'.format(kspace_loss.item()))
                    FINEs.append(X.cpu().detach())

            FINEs = r2c(np.concatenate(FINEs, axis=0))
            print(FINEs.shape)
            FINEs = np.transpose(FINEs, [1,2,0])

            adict = {}
            adict['FINEs'] = FINEs
            sio.savemat(rootName+'/results/FINEs_CardiacQSM.mat', adict)





        #     Kunders.append(kunder.cpu().detach())

        # Kunders = np.concatenate(Kunders, axis=0)
        # Kunders_ = np.zeros(Kunders.shape[:4], dtype=np.complex64)
        # Kunders_ = Kunders[..., 0] + 1j*Kunders[..., 1]
        # Kunders_ = np.transpose(Kunders_, [2, 3, 0, 1])

        # adict = {}
        # adict['Kunders'] = Kunders_
        # sio.savemat(rootName+'/results/Kunders_CardiacQSM.mat', adict)


