import os
import time
import torch
import numpy as np
from torch.utils import data
from loader.kdata_loader_GE import kdata_loader_GE
from models.unet import Unet
from models.initialization import *
from models.dc_blocks import *
from models.unet_with_dc import *
from models.resnet_with_dc import *
from models.dc_with_prop_mask import *     
from utils.test import *
from utils.data import *
from utils.test import *

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    K = 2
    use_uncertainty = False
    fixed_mask = True
    testing = True
    lambda_Pmask = 0.015  
    lambda_dll2 = 0.01
    batch_size = 2
    folderName = '{0}_rolls'.format(K)
    rootName = '/data/Jinwei/T2_slice_recon_GE'

    dataLoader_test = kdata_loader_GE(split='val')
    testLoader = data.DataLoader(dataLoader_test, batch_size=batch_size, shuffle=False)

    netG_dc = DC_with_Prop_Mask(
        input_channels=2, 
        filter_channels=32, 
        lambda_dll2=lambda_dll2, 
        K=K,
        unc_map=use_uncertainty,
        fixed_mask=fixed_mask,
        testing=testing
    )
    print(netG_dc)
    netG_dc.to(device)
    # netG_dc.load_state_dict(torch.load(rootName+'/'+folderName+
    #     '/weights_lambda_pmask={}_optimal.pt'.format(lambda_Pmask)))
    netG_dc.load_state_dict(torch.load(rootName+'/'+folderName+
        '/weights_lambda_pmask={}_fixed_optimal.pt'.format(lambda_Pmask)))
    netG_dc.eval()
    print(netG_dc.lambda_dll2)
    metrices_test = Metrices()

    Recons = []
    for idx, (inputs, targets, csms) in enumerate(testLoader):
        print(idx)
        inputs = inputs.to(device)
        targets = targets.to(device)
        csms = csms.to(device)
        # calculating metrices
        Xs = netG_dc(inputs, csms)
        Recons.append(Xs[-1].cpu().detach())
        metrices_test.get_metrices(Xs[-1], targets)
        if idx == 0:
            print(torch.mean(netG_dc.masks))
            adict = {}
            adict['Mask'] = np.squeeze(np.asarray(netG_dc.Pmask.cpu().detach()))
            sio.savemat(rootName+'/'+folderName+
                        '/Optimal_mask_{}.mat'.format(lambda_Pmask), adict)

    print(np.mean(np.asarray(metrices_test.PSNRs)))
    Recons = np.concatenate(Recons, axis=0)

    adict = {}
    adict['Recons'] = Recons
    sio.savemat(rootName+'/'+folderName+
                '/Recons_{}.mat'.format(lambda_Pmask), adict)



