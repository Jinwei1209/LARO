import os
import time
import torch
import numpy as np
from torch.utils import data
from loader.kdata_loader_GE import kdata_loader_GE
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
from utils.test import *


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    lrG_dc = 1e-3
    niter = 8800
    batch_size = 4
    display_iters = 10
    lambda_Pmask = 0  # 0.01
    lambda_dll2 = 0.01 
    K = 2
    samplingRatio = 0.1
    use_uncertainty = False
    fixed_mask = False
    testing = False
    rescale = True
    folderName = '{0}_rolls'.format(K)
    rootName = '/data/Jinwei/T1_slice_recon_GE'

    epoch = 0
    gen_iterations = 1
    errL2_dc_sum = Pmask_ratio = 0
    PSNRs_val = []
    Validation_loss = []

    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataLoader = kdata_loader_GE(
        rootDir=rootName,
        contrast='T1', 
        split='train'
        )
    trainLoader = data.DataLoader(dataLoader, batch_size=batch_size, shuffle=True)

    dataLoader_val = kdata_loader_GE(
        rootDir=rootName,
        contrast='T1', 
        split='val'
        )
    valLoader = data.DataLoader(dataLoader_val, batch_size=batch_size, shuffle=True)
    
    netG_dc = DC_with_Prop_Mask(
        input_channels=2, 
        filter_channels=32, 
        lambda_dll2=lambda_dll2,
        K=K, 
        unc_map=use_uncertainty,
        fixed_mask=fixed_mask,
        testing=testing,
        rescale=rescale
    )

    # netG_dc = DC_with_Straight_Through_Pmask(
    #     input_channels=2, 
    #     filter_channels=32, 
    #     lambda_dll2=lambda_dll2,
    #     K=K, 
    #     unc_map=use_uncertainty,
    #     fixed_mask=fixed_mask,
    #     testing=testing,
    #     rescale=rescale
    # )

    print(netG_dc)
    netG_dc.to(device)

    # # load pre-trained weights with pmask
    # netG_dc.load_state_dict(torch.load(rootName+'/'+folderName+
    #                         '/weights_lambda_pmask={}_optimal.pt'.format(lambda_Pmask)))
    # netG_dc.eval()

    # # save thresh_const for network training with fixed optimal mask training 
    # adict = {}
    # adict['Thresh'] = np.asarray(netG_dc.thresh_const.cpu().detach()) 
    # sio.savemat(rootName+'/'+folderName+
    #             '/Thresh_{}.mat'.format(lambda_Pmask), adict)

    optimizerG_dc = optim.Adam(netG_dc.parameters(), lr=lrG_dc, betas=(0.9, 0.999))
    logger = Logger(folderName, rootName)
    
    while epoch < niter:
        epoch += 1 
        # training phase
        metrices_train = Metrices()
        for idx, (inputs, targets, csms) in enumerate(trainLoader):
            
            if gen_iterations%display_iters == 0:

                print('epochs: [%d/%d], batchs: [%d/%d], time: %ds, Lambda: %f'
                % (epoch, niter, idx, 1050//batch_size+1, time.time()-t0, lambda_Pmask))

                print('Lambda_dll2: %f, Sampling ratio cal: %f, Sampling ratio setup: %f, Pmask: %f' 
                    % (netG_dc.lambda_dll2, torch.mean(netG_dc.masks), \
                        netG_dc.samplingRatio, torch.mean(netG_dc.Pmask)))

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

            errL2_dc, loss_Pmask = netG_dc_train_pmask(
                inputs, 
                targets, 
                csms, 
                netG_dc, 
                optimizerG_dc, 
                use_uncertainty,
                lambda_Pmask
            )
            errL2_dc_sum += errL2_dc 
            Pmask_ratio += loss_Pmask

            # calculating metrices
            Xs = netG_dc(inputs, csms)
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
            Xs = netG_dc(inputs, csms)
            metrices_val.get_metrices(Xs[-1], targets)

            targets = np.asarray(targets.cpu().detach())
            lossl2_sum = loss_unc_sum = 0
            for i in range(len(Xs)):
                Xs_i = np.asarray(Xs[i].cpu().detach())
                temp = abs(Xs_i - targets)
                lossl2_sum += np.mean(temp)
            temp = np.asarray(netG_dc.Pmask.cpu().detach())
            loss_Pmask = lambda_Pmask*np.mean(temp)
            loss_total = lossl2_sum + loss_Pmask
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
                       '/weights_lambda_pmask={}_optimal.pt'.format(lambda_Pmask))
        torch.save(netG_dc.state_dict(), logger.logPath+
                   '/weights_lambda_pmask={}_last.pt'.format(lambda_Pmask))

        logger.close()
