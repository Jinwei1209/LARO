import os
import numpy as np
from torch.utils import data
from utils.data import *
from utils.loss import *


class real_and_kdata_loader(data.Dataset):


    folderMatcher = {
        'T1': 'T1_slice_mat/',
        'T2': 'T2_slice_mat/',
        'T2FLAIR': 'T2FLAIR_slice_mat/'
    }

    dataRange = {
        'train': ['0', '200'],
        'val': ['8800', '8850'],
        'test': ['11000', '11857']
    }


    def __init__(self,
        rootDir = '/data/Jinwei/T2_slice_recon/',
        contrast = 'T2',
        split = 'train',
        batchSize = 1,
        augmentations = [None],
        nrow = 256,
        ncol = 184,
        ncoil = 1
    ):

        self.rootDir = rootDir
        self.dataFD = rootDir + self.folderMatcher[contrast]
        self.dataFD_T1 = rootDir + self.folderMatcher['T1']
        self.startIdx = int(self.dataRange[split][0])
        self.endIdx = int(self.dataRange[split][1])
        self.nsamples = self.endIdx - self.startIdx

        self.augmentations = augmentations
        self.augmentation = self.augmentations[0]
        self.augSize = len(self.augmentations)
        self.augIndex = 0
        self.batchSize = batchSize
        self.batchIndex = 0

        self.ncoil = ncoil  # just one coil for real-valued images
        self.nrow = nrow
        self.ncol = ncol


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

        org = load_mat(self.dataFD + '%d.mat' %(idx), 'a')
        org = np.float32(org[np.newaxis, ...]) 

        csm = np.ones((self.ncoil, self.nrow, self.ncol))
        # csm = abs(org) > 1e-1
        csm = np.float32(csm)

        kdata = self.generateKdata(org, csm, sigma=0.00)
        # for real-valued images
        org = np.float32(np.concatenate((org, np.zeros(org.shape)), axis=0))
        csm = np.float32(np.concatenate((csm[..., np.newaxis], np.zeros(csm.shape+(1,))), axis=-1))
        brain_mask = np.ones(org.shape, dtype=np.float32)

        return kdata, org, csm, brain_mask


    def generateKdata(self, org, csm, sigma=0.01):

        noise = np.random.randn(self.ncoil, self.nrow, self.ncol) \
                + 1j*np.random.randn(self.ncoil, self.nrow, self.ncol)
        noise = noise * (sigma / np.sqrt(2.))
        y = np.fft.fft2(org) / np.sqrt(self.nrow*self.ncol) + noise
        kdata = c2r_kdata(y)

        return kdata




        

        


    
