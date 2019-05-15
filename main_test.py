import os
import time
import torch
import numpy as np
from torch.utils import data
from loader.real_data_loader import real_data_loader
from models.unet import Unet
from models.initialization import *
from models.dc_blocks import *
from models.unet_with_dc import *
from models.resnet_with_dc import *     
from utils.test import *
from utils.data import *
from utils.test import *

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    K = 10
    lambda_dll2 = 0.01
    batch_size = 6
    folderName = '{0}_rolls'.format(K)
    rootName = '/data/Jinwei/T2_slice_recon'

    dataLoader_test = real_data_loader(split='test')
    testLoader = data.DataLoader(dataLoader_test, batch_size=batch_size, shuffle=False)

    netG_dc = Resnet_with_DC(input_channels=2, filter_channels=32, lambda_dll2=lambda_dll2, K=K)
    netG_dc.to(device)
    netG_dc.load_state_dict(torch.load(rootName+'/'+folderName+'/weights.pt'))
    netG_dc.eval()
    print(netG_dc.lambda_dll2)
    metrices_test = Metrices()

    for idx, (inputs, targets, csms, masks) in enumerate(testLoader):
        print(idx)
        inputs = inputs.to(device)
        targets = targets.to(device)
        csms = csms.to(device)
        masks = masks.to(device)
        # calculating metrices
        outputs = netG_dc(inputs, csms, masks)
        # outputs = netG(inputs, csms, masks)
        metrices_test.get_metrices(outputs, targets)
    print(np.mean(np.asarray(metrices_test.PSNRs)))



