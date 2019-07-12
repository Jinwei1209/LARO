import os
import time
import torch
import numpy as np
from torch.utils import data
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
from utils.test import *


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    lrG_dc = 1e-3
    niter = 8800
    batch_size = 12
    display_iters = 10
    lambda_Pmask = 10
    lambda_dll2 = 0.01
    K = 3
    use_uncertainty = True
    fixed_mask = True
    folderName = '{0}_rolls'.format(K)
    rootName = '/data/Jinwei/T2_slice_recon'

    epoch = 0
    gen_iterations = 1
    errL2_dc_sum = errL2_unc_sum = Pmask_ratio = 0
    PSNRs_val = []
    Validation_loss = []

    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataLoader = real_and_kdata_loader(split='train')
    trainLoader = data.DataLoader(dataLoader, batch_size=batch_size, shuffle=True)

    dataLoader_val = real_and_kdata_loader(split='val')
    valLoader = data.DataLoader(dataLoader_val, batch_size=batch_size//2, shuffle=True)
    
    netG_dc = DC_with_Prop_Mask(
        input_channels=2, 
        filter_channels=32, 
        lambda_dll2=lambda_dll2,
        K=K, 
        unc_map=use_uncertainty,
        fixed_mask=fixed_mask
    )
    print(netG_dc)
    netG_dc.to(device)

    netG_dc.load_state_dict(torch.load(rootName+'/'+folderName+
                            '/weights_sigma=0.01_lambda_pmask={}.pt'.format(lambda_Pmask)))
    netG_dc.eval()

    adict = {}
    adict['Thresh'] = np.asarray(netG_dc.thresh_const.cpu().detach()) 
    sio.savemat(rootName+'/'+folderName+
                '/Thresh_{}.mat'.format(lambda_Pmask), adict)

    optimizerG_dc = optim.Adam(netG_dc.parameters(), lr=lrG_dc, betas=(0.9, 0.999))
    logger = Logger(folderName, rootName)
    
    while epoch < niter:
        epoch += 1 
        # training phase
        metrices_train = Metrices()
        for idx, (inputs, targets, csms) in enumerate(trainLoader):
            
            if gen_iterations%display_iters == 0:

                print('epochs: [%d/%d], batchs: [%d/%d], time: %ds'
                % (epoch, niter, idx, 200//batch_size+1, time.time()-t0))

                print('Lambda_dll2: %f, Sampling ratio: %f, lambda_Pmask: %d' 
                    % (netG_dc.lambda_dll2, torch.mean(netG_dc.Pmask), lambda_Pmask))

                print('netG_dc --- loss_L2_dc: %f, loss_uncertainty: %f'
                    % (errL2_dc_sum/display_iters, errL2_unc_sum/display_iters))

                print('Average PSNR in Training dataset is %.2f' 
                % (np.mean(np.asarray(metrices_train.PSNRs[-1-display_iters*batch_size:]))))
                if epoch > 1:
                    print('Average PSNR in Validation dataset is %.2f' 
                    % (np.mean(np.asarray(metrices_val.PSNRs))))

                errL2_dc_sum = errL2_unc_sum = Pmask_ratio = 0
                
            inputs = inputs.to(device)
            targets = targets.to(device)
            csms = csms.to(device)

            errL2_dc, loss_unc, loss_Pmask = netG_dc_train_pmask(
                inputs, 
                targets, 
                csms, 
                netG_dc, 
                optimizerG_dc, 
                use_uncertainty,
                lambda_Pmask
            )
            errL2_dc_sum += errL2_dc 
            errL2_unc_sum += loss_unc
            Pmask_ratio += loss_Pmask

            # calculating metrices
            Xs, Unc_maps = netG_dc(inputs, csms)
            metrices_train.get_metrices(Xs[-1], targets)

            gen_iterations += 1
            
        # validation phase
        metrices_val = Metrices()
        loss_total_list = []
        for idx, (inputs, targets, csms) in enumerate(valLoader):

            inputs = inputs.to(device)
            targets = targets.to(device)
            csms = csms.to(device)

            # calculating metrices
            Xs, Unc_maps = netG_dc(inputs, csms)
            metrices_val.get_metrices(Xs[-1], targets)

            targets = np.asarray(targets.cpu().detach())
            lossl2_sum = loss_unc_sum = 0
            for i in range(len(Xs)):
                Xs_i = np.asarray(Xs[i].cpu().detach())
                Unc_maps_i = np.asarray(Unc_maps[i].cpu().detach())
                temp = (Xs_i - targets)**2
                lossl2_sum += np.mean(np.sum(temp, axis=1)/np.exp(Unc_maps_i))
                loss_unc_sum += np.mean(Unc_maps_i)
            temp = np.asarray(netG_dc.Pmask.cpu().detach())
            loss_Pmask = lambda_Pmask*np.mean(temp)
            loss_total = lossl2_sum + loss_unc_sum + loss_Pmask
            loss_total_list.append(loss_total)
        print('\n Validation loss: %f \n' 
            % (sum(loss_total_list) / float(len(loss_total_list))))
        Validation_loss.append(sum(loss_total_list) / float(len(loss_total_list)))
        # save log
        logger.print_and_save('Epoch: [%d/%d], PSNR in Training: %.2f' 
        % (epoch, niter, np.mean(np.asarray(metrices_train.PSNRs))))
        logger.print_and_save('Epoch: [%d/%d], PSNR in Validation: %.2f' 
        % (epoch, niter, np.mean(np.asarray(metrices_val.PSNRs))))

        # save weights
        PSNRs_val.append(np.mean(np.asarray(metrices_val.PSNRs)))
        if Validation_loss[-1] == min(Validation_loss):
            torch.save(netG_dc.state_dict(), logger.logPath+
                       '/weights_sigma=0.01_lambda_pmask={}_optimal_fixed.pt'.format(lambda_Pmask))
        torch.save(netG_dc.state_dict(), logger.logPath+
                   '/weights_sigma=0.01_lambda_pmask={}_last_fixed.pt'.format(lambda_Pmask))

        logger.close()

