import os
import numpy as np
from torch.utils import data
from utils.data import *
from utils.loss import *
from utils.operators import *


class kdata_loader_GE(data.Dataset):


    folderMatcher = {
        'T1': '/Total_slices_T1/',
        'T2': '/Total_slices_T2/', 
        'T2FLAIR': '/Total_slices_T2FLAIR/',
        'CardiacQSM': '/Total_slices_CardiacQSM/',
        'CardiacSub6': '/Total_slices_multi_echo_sub6/'
    }

    # dataRangeT1 = {
    #     # 'train': ['150', '1200'],
    #     'train': ['150', '600'],   
    #     'val': ['1200', '1350'],  # 1200 1350
    #     'test': ['1350', '1650']  # 1350 1650
    #     # 'test': ['1400', '1401']
    # }

    dataRangeT1 = {
        'train': ['0', '300'],   
        'val': ['300', '400'],
        # 'test': ['400', '600']
        'test': ['440', '441']
    }

    dataRangeT2 = {
        'train': ['0', '300'],   
        'val': ['300', '400'],
        'test': ['400', '500']
        # 'test': ['440', '441']
    }

    # # for the phantom study
    # dataRangeT2 = {
    #     'train': ['1000', '1100'],   
    #     'val': ['1000', '1100'],
    #     'test': ['1000', '1100']
    # }
    
    dataRangeCardiac = {
        'train': ['0', '90'],   
        'val': ['90', '108'],
        # 'test': ['108', '126']
        'test': ['98', '99']
    }

    # for cardiac sub6 with multi-echo data
    dataRangeCardiacSub6 = {
        'test': ['0', '90']
    }


    def __init__(self,
        rootDir = '/data/Jinwei/T2_slice_recon_GE',
        contrast = 'T2',
        split = 'train',
        batchSize = 1,
        augmentations = [None],
        SNR = 0,  # used for BO project
        flag_BO = 0,  # used for BO project
        slice_spacing = 25  # used for BO project
    ):

        self.rootDir = rootDir
        self.dataFD = rootDir + self.folderMatcher[contrast]
        self.contrast = contrast
        if contrast == 'T1':
            self.startIdx = int(self.dataRangeT1[split][0])
            self.endIdx = int(self.dataRangeT1[split][1])
        elif contrast == 'T2':
            self.startIdx = int(self.dataRangeT2[split][0])
            self.endIdx = int(self.dataRangeT2[split][1])
        elif contrast == 'CardiacQSM':
            self.startIdx = int(self.dataRangeCardiac[split][0])
            self.endIdx = int(self.dataRangeCardiac[split][1])
        elif contrast == 'CardiacSub6':
            self.startIdx = int(self.dataRangeCardiacSub6[split][0])
            self.endIdx = int(self.dataRangeCardiacSub6[split][1])

        self.nsamples = self.endIdx - self.startIdx
        self.augmentations = augmentations
        self.augmentation = self.augmentations[0]
        self.augSize = len(self.augmentations)
        self.augIndex = 0
        self.batchSize = batchSize
        self.batchIndex = 0
        self.SNR = SNR
        self.flag_BO = flag_BO
        self.slice_spacing = slice_spacing

        if self.flag_BO:
            self.startIdx = 400
            self.endIdx = 600
            self.nsamples = (self.endIdx - self.startIdx) // self.slice_spacing


    def __len__(self):

        return self.nsamples * self.augSize


    def __getitem__(self, idx):
        if self.flag_BO:
            idx = int(idx / self.augSize) * self.slice_spacing + self.startIdx
        else:
            idx = int(idx / self.augSize) + self.startIdx

        if (self.batchIndex == self.batchSize):

            self.batchIndex = 0
            self.augIndex += 1
            self.augIndex = self.augIndex % self.augSize
            self.augmentation = self.augmentations[self.augIndex]

        self.batchIndex += 1
        org = load_mat(self.dataFD + 'fully_slice_%d.mat' %(idx), 'fully_slice')
        org =  c2r(org)
        csm = load_mat(self.dataFD + 'sensMaps_slice_%d.mat' %(idx), 'sensMaps_slice')
        csm = np.transpose(csm, (2, 0, 1))
        csm = c2r_kdata(csm)
        kdata = load_mat(self.dataFD + 'kdata_slice_%d.mat' %(idx), 'kdata_slice')
        kdata = np.transpose(kdata, (2, 0, 1))
        kdata = c2r_kdata(kdata)
        # add gaussian noise in kdata, SNR = 0 for not adding noise, otherwise referring to desired linear SNR
        if self.SNR != 0:
            var_n = np.mean(np.sqrt(kdata[..., 0]**2 + kdata[..., 1]**2).flatten()) / self.SNR
            kdata += np.random.normal(0, np.sqrt(var_n/2), size=len(kdata.flatten())).reshape(kdata.shape)

        if self.contrast == 'T1':
            # tmp = load_mat(self.dataFD + 'brain_mask_slice_%d.mat' %(idx), 'brain_mask_slice')
            # brain_mask = np.zeros(org.shape, dtype=np.float32)
            # brain_mask[0, ...] = tmp
            # brain_mask[1, ...] = tmp
            brain_mask = np.ones(org.shape, dtype=np.float32) 
        else:
            brain_mask = np.ones(org.shape, dtype=np.float32)
        return kdata, org, csm, brain_mask




        

        


    
