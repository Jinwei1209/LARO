'''
    Dataloader of multi-echo GRE data from GE scanner
'''

import os
import numpy as np
from torch.utils import data
from utils.data import *
from utils.loss import *
from utils.operators import *


class kdata_multi_echo_GE(data.Dataset):


    folderMatcher = {
        'MultiEcho': '/megre_slice_GE/'
    }

    dataRange = {
        'train': ['200', '800'],   
        'val': ['0', '200'],
        'test': ['800', '1000']
    }
    

    def __init__(self,
        rootDir = '/data/Jinwei/Multi_echo_slice_recon_GE',
        contrast = 'MultiEcho',
        necho = 10, # number of echos
        split = 'train',
        batchSize = 1,
        augmentations = [None]
    ):

        self.rootDir = rootDir
        self.dataFD = rootDir + self.folderMatcher[contrast]
        self.contrast = contrast
        self.necho = necho
        if contrast == 'MultiEcho':
            self.startIdx = int(self.dataRange[split][0])
            self.endIdx = int(self.dataRange[split][1])
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

        org = readcfl(self.dataFD + 'fully_slice_{}'.format(idx))
        org =  c2r(org, 1)  # (2*echo, row, col) with first dimension real&imag concatenated for all echos 

        csm = readcfl(self.dataFD + 'sensMaps_slice_{}'.format(idx))
        csm = np.transpose(csm, (2, 0, 1))[:, np.newaxis, ...]  # (coil, 1, row, col)
        csm = np.repeat(csm, self.necho, axis=1)  # (coil, echo, row, col)
        csm = c2r_kdata(csm) # (coil, echo, row, col, 2) with last dimension real&imag

        kdata = readcfl(self.dataFD + 'kdata_slice_{}'.format(idx))
        kdata = np.transpose(kdata, (2, 3, 0, 1))  # (coil, echo, row, col)
        kdata = c2r_kdata(kdata) # (coil, echo, row, col, 2) with last dimension real&imag

        brain_mask = np.ones(org.shape, dtype=np.float32)
        
        return kdata, org, csm, brain_mask




        

        


    