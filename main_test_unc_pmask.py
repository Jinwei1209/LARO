import os
import time
import torch
import numpy as np
from torch.utils import data
from loader.real_and_kdata_loader import real_and_kdata_loader
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

    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    K = 3
    use_uncertainty = True
    slope_threshold = 12
    lambda_Pmask = 1
    lambda_dll2 = 0.01
    batch_size = 12
    folderName = '{0}_rolls'.format(K)
    rootName = '/data/Jinwei/T2_slice_recon'

    dataLoader_test = real_and_kdata_loader(split='test')
    testLoader = data.DataLoader(dataLoader_test, batch_size=batch_size, shuffle=False)

    netG_dc = DC_with_Prop_Mask(
        input_channels=2, 
        filter_channels=32, 
        lambda_dll2=lambda_dll2, 
        K=K,
        unc_map=use_uncertainty,
        slope_threshold=slope_threshold
    )
    print(netG_dc)
    netG_dc.to(device)
    netG_dc.load_state_dict(torch.load(rootName+'/'+folderName+
                            '/weights_sigma=0.01_lambda_pmask={}.pt'.format(lambda_Pmask)))
    netG_dc.eval()
    print(netG_dc.lambda_dll2)
    print(torch.mean(netG_dc.Pmask))
    metrices_test = Metrices()

    for idx, (inputs, targets, csms) in enumerate(testLoader):
        print(idx)
        inputs = inputs.to(device)
        targets = targets.to(device)
        csms = csms.to(device)
        # calculating metrices
        Xs, Unc_maps = netG_dc(inputs, csms)
        metrices_test.get_metrices(Xs[-1], targets)
    print(np.mean(np.asarray(metrices_test.PSNRs)))



