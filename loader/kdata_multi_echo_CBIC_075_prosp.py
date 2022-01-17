import os
import numpy as np
from torch.utils import data
from utils.data import *
from utils.loss import *
from utils.operators import *


class kdata_multi_echo_CBIC_075_prosp(data.Dataset):
    '''
        Dataloader of multi-echo GRE data from GE scanner with prospective under-sampling (CBIC scanner)
    '''

    def __init__(self,
        rootDir = '/data/Jinwei/QSM_raw_CBIC',
        contrast = 'MultiEcho',
        necho = 7, # number of echos
        nrow = 258,
        ncol = 112,
        split = 'train',
        subject = 0,  # 0: junghun, 1: chao, 2: alexey
        loupe = 0,  # 0: variable density; 1: optimal
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
        self.nrow = nrow
        self.ncol = ncol
        self.loupe = loupe
        if contrast == 'MultiEcho':
            if split == 'train':
                self.nsamples = 320 * 4
            elif split == 'val':
                self.nsamples = 320
            elif split == 'test':
                self.nsamples = 320
                if subject == 0:
                    self.subject = 'jiahao'
                # elif subject == 1:
                #     self.subject = 'chao2'
                # elif subject == 2:
                #     self.subject = 'alexey2'
                # elif subject == 3:
                #     self.subject = 'liangdong2'
                # elif subject == 4:
                #     self.subject = 'fenglei2'
                # elif subject == 5:
                #     self.subject = 'jiahao2'
                # elif subject == 6:
                #     self.subject = 'dom2'
                # elif subject == 7:
                #     self.subject = 'hangwei2'
                # elif subject == 8:
                #     self.subject = 'wenxin2'
                # elif subject == 9:
                #     self.subject = 'hanxuan2'
                # elif subject == 10:
                #     self.subject = 'qihao2'
                print("Test on {} with prospective under-sampling".format(self.subject))
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
                if (idx - 320 < 0):
                    break
                else:
                    idx -= 320
                    subject += 1
            if subject == 0:
                dataFD = self.rootDir + '/data_cfl/thanh2/full_cc_slices/'
            elif subject == 1:
                dataFD = self.rootDir + '/data_cfl/jinwei2/full_cc_slices/'
            elif subject == 2:
                dataFD = self.rootDir + '/data_cfl/qihao2/full_cc_slices/'

        elif self.split == 'val':
            dataFD = self.rootDir + '/data_cfl/jiahao2/full_cc_slices/'
        
        elif self.split == 'test':
            dataFD_prosp = self.rootDir + '/data_cfl/' + self.subject + '/10_loupe={}_cc_slices_sense_echo/'.format(self.loupe)

        if (self.batchIndex == self.batchSize):
            self.batchIndex = 0
            self.augIndex += 1
            self.augIndex = self.augIndex % self.augSize
            self.augmentation = self.augmentations[self.augIndex]
        self.batchIndex += 1

        org = readcfl(dataFD_prosp + 'fully_slice_{}'.format(idx))  # (row, col, echo)
        org =  c2r(org, self.echo_cat)  # echo_cat == 1: (2*echo, row, col) with first dimension real&imag concatenated for all echos 
                                        # echo_cat == 0: (2, row, col, echo)

        if self.echo_cat == 0:
            org = np.transpose(org, (0, 3, 1, 2)) # (2, echo, row, col)
        
        # Coil sensitivity maps
        csm = readcfl(dataFD_prosp + 'sensMaps_slice_{}'.format(idx))
        # csm = np.transpose(csm, (2, 0, 1))[:, np.newaxis, ...]  # (coil, 1, row, col)
        # csm = np.repeat(csm, self.necho, axis=1)  # (coil, echo, row, col)
        csm = np.transpose(csm, (2, 3, 0, 1))  # (coil, echo, row, col)
        for i in range(self.necho):
            csm[:, i, :, :] = csm[:, i, :, :] * np.exp(-1j * np.angle(csm[0:1, i, :, :]))
        csm = c2r_kdata(csm) # (coil, echo, row, col, 2) with last dimension real&imag

        # Fully sampled kspace data
        kdata = readcfl(dataFD_prosp + 'kdata_slice_{}'.format(idx))
        kdata = np.transpose(kdata, (2, 3, 0, 1))  # (coil, echo, row, col)
        kdata = c2r_kdata(kdata) # (coil, echo, row, col, 2) with last dimension real&imag

        # brain tissue mask (not used for prospective testing)
        brain_mask = abs(org) > 0 

        if self.normalization == 0:
            return kdata, org, csm, brain_mask





        

        


    
