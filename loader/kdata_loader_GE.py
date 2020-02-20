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
        'T2FLAIR': '/Total_slices_T2FLAIR/'
    }

    dataRangeT1 = {
        'train': ['150', '1200'],   
        'val': ['1200', '1350'],  # 1200 1350
        # 'test': ['1350', '1650']  # 1350 1650
        'test': ['1400', '1401']
    }

    dataRangeT2 = {
        'train': ['0', '300'],   
        'val': ['300', '400'],
        # 'test': ['400', '500']
        'test': ['450', '451']
    }


    def __init__(self,
        rootDir = '/data/Jinwei/T2_slice_recon_GE',
        contrast = 'T2',
        split = 'train',
        batchSize = 1,
        augmentations = [None]
    ):

        self.rootDir = rootDir
        self.dataFD = rootDir + self.folderMatcher[contrast]
        self.contrast = contrast
        if contrast == 'T1':
            self.startIdx = int(self.dataRangeT1[split][0])
            self.endIdx = int(self.dataRangeT1[split][1])
        else:
            self.startIdx = int(self.dataRangeT2[split][0])
            self.endIdx = int(self.dataRangeT2[split][1])
        self.nsamples = self.endIdx - self.startIdx

        self.augmentations = augmentations
        self.augmentation = self.augmentations[0]
        self.augSize = len(self.augmentations)
        self.augIndex = 0
        self.batchSize = batchSize
        self.batchIndex = 0


    def __len__(self):

        return self.nsamples * self.augSize


    def __getitem__(self, idx):

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
        if self.contrast == 'T1':
            tmp = load_mat(self.dataFD + 'brain_mask_slice_%d.mat' %(idx), 'brain_mask_slice')
            brain_mask = np.zeros(org.shape, dtype=np.float32)
            brain_mask[0, ...] = tmp
            brain_mask[1, ...] = tmp 
        else:
            brain_mask = np.ones(org.shape, dtype=np.float32)
        return kdata, org, csm, brain_mask

        

        


    
