import os
import numpy as np
from torch.utils import data
from utils.data import *
from utils.loss import *
from utils.operators import *


class kdata_multi_echo_CBIC(data.Dataset):
    '''
        Dataloader of multi-echo GRE data from GE scanner (CBIC scanner)
    '''

    def __init__(self,
        rootDir = '/data/Jinwei/QSM_raw_CBIC',
        contrast = 'MultiEcho',
        necho = 10, # number of echos
        split = 'train',
        normalization = 0,  # flag to normalize the data
        echo_cat = 1, # flag to concatenate echo dimension into channel
        batchSize = 1,
        augmentations = [None]
    ):

        self.rootDir = rootDir
        self.contrast = contrast
        self.necho = necho
        self.normalization = normalization
        self.echo_cat = echo_cat
        self.split = split
        if contrast == 'MultiEcho':
            if split == 'train':
                self.nsamples = 600
            elif split == 'val':
                self.nsamples = 200
            elif split == 'test':
                self.nsamples = 200
        self.augmentations = augmentations
        self.augmentation = self.augmentations[0]
        self.augSize = len(self.augmentations)
        self.augIndex = 0
        self.batchSize = batchSize
        self.batchIndex = 0


    def __len__(self):

        return self.nsamples * self.augSize


    def __getitem__(self, idx):

        if self.split == 'train':
            subject = 0
            while (True):
                if (idx - 200 < 0):
                    break
                else:
                    idx -= 200
                    subject += 1
            if subject == 0:
                dataFD = self.rootDir + '/data_cfl/thanh/full_cc_slices/'
            elif subject == 1:
                dataFD = self.rootDir + '/data_cfl/jinwei/full_cc_slices/'
            elif subject == 2:
                dataFD = self.rootDir + '/data_cfl/qihao/full_cc_slices/'

        elif self.split == 'val':
            dataFD = self.rootDir + '/data_cfl/jiahao/full_cc_slices/'
        
        elif self.split == 'test':
            dataFD = self.rootDir + '/data_cfl/jiahao/full_cc_slices/'

        idx += 30

        if (self.batchIndex == self.batchSize):
            self.batchIndex = 0
            self.augIndex += 1
            self.augIndex = self.augIndex % self.augSize
            self.augmentation = self.augmentations[self.augIndex]
        self.batchIndex += 1

        org = readcfl(dataFD + 'fully_slice_{}'.format(idx))  # (row, col, echo)
        org =  c2r(org, self.echo_cat)  # echo_cat == 1: (2*echo, row, col) with first dimension real&imag concatenated for all echos 
                                        # echo_cat == 0: (2, row, col, echo)

        org_gen = org

        if self.echo_cat == 0:
            org = np.transpose(org, (0, 3, 1, 2)) # (2, echo, row, col)
            org_gen = np.transpose(org_gen, (0, 3, 1, 2))
            
        csm = readcfl(dataFD + 'sensMaps_slice_{}'.format(idx))
        csm = np.transpose(csm, (2, 0, 1))[:, np.newaxis, ...]  # (coil, 1, row, col)
        csm = np.repeat(csm, self.necho, axis=1)  # (coil, echo, row, col)
        csm = c2r_kdata(csm) # (coil, echo, row, col, 2) with last dimension real&imag

        kdata = readcfl(dataFD + 'kdata_slice_{}'.format(idx))
        kdata = np.transpose(kdata, (2, 3, 0, 1))  # (coil, echo, row, col)
        kdata = c2r_kdata(kdata) # (coil, echo, row, col, 2) with last dimension real&imag

        brain_mask = np.real(readcfl(dataFD + 'mask_slice_{}'.format(idx)))  # (row, col)
        if self.echo_cat:
            brain_mask = np.repeat(brain_mask[np.newaxis, ...], self.necho*2, axis=0) # (2*echo, row, col)
        else:
            brain_mask = np.repeat(brain_mask[np.newaxis, ...], 2, axis=0) # (2, row, col)
            brain_mask = np.repeat(brain_mask[:, np.newaxis, ...], self.necho, axis=1)# (2, echo, row, col)
        
        if self.normalization == 0:
            return kdata, org, org_gen, csm, brain_mask





        

        


    
