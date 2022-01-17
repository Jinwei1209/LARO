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
    lrG_dc = 2e-4
    niter = 50
    batch_size = 6
    display_iters = 10
    lambda_l1 = 1000
    lambda_dll2 = 0.01
    lambda_dc = 1000
    K = 10
    use_uncertainty = True
    folderName = '{0}_rolls'.format(K)
    rootName = '/data/Jinwei/T2_slice_recon'

    epoch = 0
    gen_iterations = 1
    errL2_dc_sum = errL2_unc_sum = 0
    PSNRs_val = []

    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataLoader = real_data_loader(split='train')
    trainLoader = data.DataLoader(dataLoader, batch_size=batch_size, shuffle=True)

    dataLoader_val = real_data_loader(split='val')
    valLoader = data.DataLoader(dataLoader_val, batch_size=batch_size//2, shuffle=True)
    
    netG_dc = Resnet_with_DC(
        input_channels=2, 
        filter_channels=32, 
        lambda_dll2=lambda_dll2,
        K=K, 
        unc_map=use_uncertainty
    )
    print(netG_dc)

    netG_dc.to(device)
    optimizerG_dc = optim.Adam(netG_dc.parameters(), lr = lrG_dc, betas=(0.5, 0.999))
    logger = Logger(folderName, rootName)
    
    while epoch < niter:
        epoch += 1 
        # training phase
        metrices_train = Metrices()
        for idx, (inputs, targets, csms, masks) in enumerate(trainLoader):
            
            if gen_iterations%display_iters == 0:

                print('epochs: [%d/%d], batchs: [%d/%d], time: %ds'
                % (epoch, niter, idx, 8800//batch_size+1, time.time()-t0))

                print('Lambda_dll2: %f' % (netG_dc.lambda_dll2))

                print('netG_dc --- loss_L2_dc: %f, loss_uncertainty: %f'
                    % (errL2_dc_sum/display_iters, errL2_unc_sum/display_iters))

                print('Average PSNR in Training dataset is %.2f' 
                % (np.mean(np.asarray(metrices_train.PSNRs[-1-display_iters*batch_size:]))))
                if epoch > 1:
                    print('Average PSNR in Validation dataset is %.2f' 
                    % (np.mean(np.asarray(metrices_val.PSNRs))))

                errL2_dc_sum = errL2_unc_sum = 0
                
            inputs = inputs.to(device)
            targets = targets.to(device)
            csms = csms.to(device)
            masks = masks.to(device)

            errL2_dc, loss_unc = netG_dc_train_intermediate(inputs, targets, csms, masks,
                                                            netG_dc, optimizerG_dc, use_uncertainty)
            errL2_dc_sum += errL2_dc 
            errL2_unc_sum += loss_unc

            # calculating metrices
            Xs, Unc_maps = netG_dc(inputs, csms, masks)
            metrices_train.get_metrices(Xs[-1], targets)

            gen_iterations += 1
            
        # validation phase
        metrices_val = Metrices()
        for idx, (inputs, targets, csms, masks) in enumerate(valLoader):

            inputs = inputs.to(device)
            targets = targets.to(device)
            csms = csms.to(device)
            masks = masks.to(device)

            # calculating metrices
            Xs, Unc_maps = netG_dc(inputs, csms, masks)
            metrices_val.get_metrices(Xs[-1], targets)

        # save log
        logger.print_and_save('Epoch: [%d/%d], PSNR in Training: %.2f' 
        % (epoch, niter, np.mean(np.asarray(metrices_train.PSNRs))))
        logger.print_and_save('Epoch: [%d/%d], PSNR in Validation: %.2f' 
        % (epoch, niter, np.mean(np.asarray(metrices_val.PSNRs))))

        # save weights
        PSNRs_val.append(np.mean(np.asarray(metrices_val.PSNRs)))
        if PSNRs_val[-1] == max(PSNRs_val):
            torch.save(netG_dc.state_dict(), logger.logPath+'/weights_sigma=0.01_uncertain.pt')

        logger.close()

