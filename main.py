import os
import time
import torch
import numpy as np
from torch.utils import data
from loader.real_data_loader import real_data_loader
from utils.data import *
from models.unet import Unet
from models.initialization import *
from models.discriminator import Basic_D
from utils.train import *
from IPython.display import clear_output
from utils.loss import *
from models.dc_blocks import *
from models.unet_with_dc import *
from models.resnet_with_dc import *
from utils.test import *


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    lrG = lrD = lrG_dc = 2e-4
    niter = 50
    batch_size = 6
    display_iters = 10
    lambda_l1 = 1000
    lambda_dll2 = 0.01
    lambda_dc = 1000
    K = 10
    use_uncertainty = False
    folderName = '{0}_rolls'.format(K)
    rootName = '/data/Jinwei/T2_slice_recon'

    epoch = 0
    gen_iterations = 1
    errD_real_sum = errD_fake_sum = 0
    errL1_sum = errG_sum = errdc_sum = 0
    errL1_dc_sum = errG_dc_sum = 0
    PSNRs_val = []

    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataLoader = real_data_loader(split='train')
    trainLoader = data.DataLoader(dataLoader, batch_size=batch_size, shuffle=True)

    dataLoader_val = real_data_loader(split='val')
    valLoader = data.DataLoader(dataLoader_val, batch_size=batch_size//2, shuffle=True)

    # netG = Unet(input_channels=2, output_channels=2, num_filters=[2**i for i in range(5, 10)])
    # netD = Basic_D(input_channels=2, output_channels=2, num_filters=[2**i for i in range(3, 9)])
    
    netG_dc = Resnet_with_DC(
        input_channels=2, 
        filter_channels=32, 
        lambda_dll2=lambda_dll2,
        K=K, 
        unc_map=use_uncertainty
    )
    # netG_dc = Unet_with_DC(
    #     input_channels=2, 
    #     output_channels=2, 
    #     num_filters=[2**i for i in range(5, 10)],
    #     lambda_dll2=lambda_dll2,
    #     K=K)

    print(netG_dc)

    # netG.to(device)
    # netD.to(device)
    netG_dc.to(device)

    # optimizerD = optim.Adam(netD.parameters(), lr = lrD, betas=(0.5, 0.999))
    # optimizerG = optim.Adam(netG.parameters(), lr = lrG, betas=(0.5, 0.999))
    optimizerG_dc = optim.Adam(netG_dc.parameters(), lr = lrG_dc, betas=(0.5, 0.999))

    logger = Logger(folderName, rootName)
    
    while epoch < niter:
        
        epoch += 1 
        # training phase
        metrices_train = Metrices()
        for idx, (inputs, targets, csms, masks) in enumerate(trainLoader):
            
            if gen_iterations%display_iters == 0:
                if gen_iterations%(5*display_iters) == 0:           
                    clear_output()
                
                sampling = True
                # inputs_show, idxs = showImage(inputs, sampling=sampling)
                
                sampling = False
                # targets_show, idxs = showImage(targets, idxs=idxs, sampling=sampling)

                inputs = inputs.to(device)
                targets = targets.to(device)
                csms = csms.to(device)
                masks = masks.to(device)
                
                # outputs = netG(inputs)
                outputs = netG_dc(inputs, csms, masks)
                # outputs_np = np.squeeze(np.asarray(outputs.cpu().detach()))
                # outputs_show, idxs = showImage(outputs_np, idxs=idxs, sampling=sampling)

                print('epochs: [%d/%d], batchs: [%d/%d], time: %ds'
                % (epoch, niter, idx, 8800//batch_size+1, time.time()-t0))

                print('Lambda_dll2: %f' % (netG_dc.lambda_dll2))

                # print('Discriminator --- Loss_D_real: %f, Loss_D_fake: %f'
                # % (errD_real_sum/display_iters, errD_fake_sum/display_iters))

                # print('Unet --- Loss_G: %f, loss_L1: %f, loss_fidelity: %f'
                # % (errG_sum/display_iters, errL1_sum/display_iters, errdc_sum/display_iters))

                # print('netG_dc --- Loss_G_dc: %f, loss_L1_dc: %f'
                # % (errG_dc_sum/display_iters, errL1_dc_sum/display_iters))

                print('netG_dc --- loss_L1_dc: %f' % (errL1_dc_sum/display_iters))

                print('Average PSNR in Training dataset is %.2f' 
                % (np.mean(np.asarray(metrices_train.PSNRs[-1-display_iters*batch_size:]))))
                if epoch > 1:
                    print('Average PSNR in Validation dataset is %.2f' 
                    % (np.mean(np.asarray(metrices_val.PSNRs))))

                # errD_real_sum = errD_fake_sum = 0
                # errL1_sum = errG_sum = errdc_sum = 0
                errL1_dc_sum = errG_dc_sum = 0
                
                # A = Back_forward(csms, masks, lambda_dll2)
                # rhs = lambda_dll2*outputs + inputs
                # dc_layer = DC_layer(A, rhs)
                # tmp_images = dc_layer.CG_iter()
                # tmp_images_np = np.squeeze(np.asarray(tmp_images.cpu().detach()))
                # tmp_show, idxs = showImage(tmp_images_np, idxs=idxs, sampling=sampling)
                
            inputs = inputs.to(device)
            targets = targets.to(device)
            csms = csms.to(device)
            masks = masks.to(device)

            # # train discriminator
            # errD_real, errD_fake = netD_train(inputs, targets, csms, masks, \
            #                                   netD, netG_dc, optimizerD, dc_layer=True)
            # errD_real_sum += errD_real
            # errD_fake_sum += errD_fake

            ## train generator without dc layer, but with fidelity loss
            #AtA = Back_forward(csms, masks, lambda_dll2).AtA
            #errG, errL1, errdc = netG_train(inputs, targets, AtA, \
            #                                netD, netG, optimizerG, lambda_l1, lambda_dc)
            #errG_sum += errG
            #errL1_sum += errL1
            #errdc_sum += errdc

            # # train generator with dc layer, but without fidelity loss
            # errG_dc, errL1_dc = netG_dc_train(inputs, targets, csms, masks, \
            #                                   netD, netG_dc, optimizerG_dc, lambda_l1)

            # train generator with dc layer, but without fidelity loss
            errL1_dc = netG_dc_train_intermediate(inputs, targets, csms, masks, netG_dc, \
                                                  optimizerG_dc, use_uncertainty)
            errL1_dc_sum += errL1_dc 

            # calculating metrices
            outputs = netG_dc(inputs, csms, masks)
            # outputs = netG(inputs)
            metrices_train.get_metrices(outputs[-1], targets)

            gen_iterations += 1
            
        # validation phase
        metrices_val = Metrices()
        for idx, (inputs, targets, csms, masks) in enumerate(valLoader):

            inputs = inputs.to(device)
            targets = targets.to(device)
            csms = csms.to(device)
            masks = masks.to(device)

            # calculating metrices
            outputs = netG_dc(inputs, csms, masks)
            # outputs = netG(inputs, csms, masks)
            metrices_val.get_metrices(outputs[-1], targets)

        # save log
        logger.print_and_save('Epoch: [%d/%d], PSNR in Training: %.2f' 
        % (epoch, niter, np.mean(np.asarray(metrices_train.PSNRs))))
        logger.print_and_save('Epoch: [%d/%d], PSNR in Validation: %.2f' 
        % (epoch, niter, np.mean(np.asarray(metrices_val.PSNRs))))

        # save weights
        PSNRs_val.append(np.mean(np.asarray(metrices_val.PSNRs)))
        if PSNRs_val[-1] == max(PSNRs_val):
            torch.save(netG_dc.state_dict(), logger.logPath+'/weights_sigma=0.01_new2.pt')

        logger.close()

