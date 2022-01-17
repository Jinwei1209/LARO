import os
import numpy as np
from torch.utils import data
from utils.data import *
from utils.loss import *


class prosp_kdata_loader_GE(data.Dataset):


    def __init__(self,
        rootDir = '/data/Jinwei/T1_slice_recon_GE/prospective_data',
        subject = 'sub1',
        mask = 'LOUPE_10',
        batchSize = 1,
        augmentations = [None]
    ):

        self.dataFD = rootDir + '/' + subject + '_' + mask + '/'
        self.startIdx = 0
        self.endIdx = 206
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

        brain_mask = np.ones(org.shape, dtype=np.float32)
        
        return kdata, org, csm, brain_mask




        

        


    
