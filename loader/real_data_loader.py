import os
import numpy as np
from torch.utils import data
from utils.data import *
from utils.loss import *


class real_data_loader(data.Dataset):


    folderMatcher = {
        'T1': 'T1_slice_mat/',
        'T2': 'T2_slice_mat/',
        'T2FLAIR': 'T2FLAIR_slice_mat/'
    }

    dataRange = {
        'train': ['0', '8800'],
        'val': ['8800', '11000'],
        'test': ['11000', '11857']
    }


    def __init__(self,
        rootDir = '/home/sdc/Jinwei/Contrast_Transform/Data/',
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

        mask = load_mat(self.rootDir + 'mask/random_30.mat', 'mask')
        mask = np.fft.fftshift(mask)
        mask = np.tile(mask, (self.ncoil, 1, 1))
        mask = np.float32(mask)

        atb = self.generateUndersampled(org, csm, mask, sigma=0.)

        # for real-valued images
        org = np.float32(np.concatenate((org, np.zeros(org.shape)), axis=0))
        csm = np.float32(np.concatenate((csm[..., np.newaxis], np.zeros(csm.shape+(1,))), axis=-1))
        mask = np.float32(np.concatenate((mask[..., np.newaxis], np.zeros(mask.shape+(1,))), axis=-1))
        
        return atb, org, csm, mask


    def generateUndersampled(self, org, csm, mask, sigma=0.):

        A = lambda z: forward_operator(z, csm, mask, self.ncoil, self.nrow, self.ncol)
        At = lambda z: backward_operator(z, csm, mask, self.ncoil, self.nrow, self.ncol)

        sidx = np.where(mask.ravel()!=0)[0]
        nSIDX = len(sidx)
        noise = np.random.randn(nSIDX*self.ncoil,) + 1j*np.random.randn(nSIDX*self.ncoil,)
        noise = noise * (sigma / np.sqrt(2.))
        y = A(org) + noise
        atb = At(y)
        atb = c2r(atb)

        return atb




        

        


    
