import os
import numpy as np
from torch.utils import data
from utils.data import *
from utils.loss import *
from utils.operators import *


class kdata_multi_echo_CBIC_075(data.Dataset):
    '''
        Dataloader of multi-echo GRE data from GE scanner (CBIC scanner)
    '''

    def __init__(self,
        rootDir = '/data2/Jinwei/QSM_raw_CBIC',
        contrast = 'MultiEcho',
        necho = 7, # number of echos
        nrow = 206,
        ncol = 80,
        split = 'train',
        subject = 0,  # 0: junghun, 1: chao, 2: alexey
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
        if contrast == 'MultiEcho':
            if split == 'train':
                self.nsamples = 320 * 4
            elif split == 'val':
                self.nsamples = 320
            elif split == 'test':
                self.nsamples = 320
                if subject == 0:
                    self.subject = 'thanh'
                elif subject == 1:
                    self.subject = 'jiahao'
                elif subject == 2:
                    self.subject = 'chao'
                elif subject == 3:
                    self.subject = 'liangdong2'
                print("Test on {}".format(self.subject))
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
                dataFD_sense_echo = self.rootDir + '/data_cfl/hangwei/full_cc_slices_sense_echo/'
            elif subject == 1:
                dataFD_sense_echo = self.rootDir + '/data_cfl/dom/full_cc_slices_sense_echo/'
            elif subject == 2:
                dataFD_sense_echo = self.rootDir + '/data_cfl/alexey/full_cc_slices_sense_echo/'
            elif subject == 3:
                dataFD_sense_echo = self.rootDir + '/data_cfl/qihao/full_cc_slices_sense_echo/'
            # elif subject == 4:
            #     dataFD_sense_echo = self.rootDir + '/data_cfl/dominick2/full_cc_slices_sense_echo/'
            # elif subject == 5:
            #     dataFD_sense_echo = self.rootDir + '/data_cfl/hangwei2/full_cc_slices_sense_echo/'
            # elif subject == 6:
            #     dataFD_sense_echo = self.rootDir + '/data_cfl/kelly2/full_cc_slices_sense_echo/'
            # elif subject == 7:
            #     dataFD_sense_echo = self.rootDir + '/data_cfl/feng2/full_cc_slices_sense_echo/'

        elif self.split == 'val':
            dataFD_sense_echo = self.rootDir + '/data_cfl/thanh/full_cc_slices_sense_echo/'
        
        elif self.split == 'test':
            dataFD_sense_echo = self.rootDir + '/data_cfl/' + self.subject + '/full_cc_slices_sense_echo/'

        if (self.batchIndex == self.batchSize):
            self.batchIndex = 0
            self.augIndex += 1
            self.augIndex = self.augIndex % self.augSize
            self.augmentation = self.augmentations[self.augIndex]
        self.batchIndex += 1

        org = readcfl(dataFD_sense_echo + 'fully_slice_{}'.format(idx))  # (row, col, echo)
        org =  c2r(org, self.echo_cat)  # echo_cat == 1: (2*echo, row, col) with first dimension real&imag concatenated for all echos 
                                        # echo_cat == 0: (2, row, col, echo)

        if self.echo_cat == 0:
            org = np.transpose(org, (0, 3, 1, 2)) # (2, echo, row, col)

        # Option 2: csms estimated from each echo
        csm = readcfl(dataFD_sense_echo + 'sensMaps_slice_{}'.format(idx))
        csm = np.transpose(csm, (2, 3, 0, 1))  # (coil, echo, row, col)
        for i in range(self.necho):
            csm[:, i, :, :] = csm[:, i, :, :] * np.exp(-1j * np.angle(csm[0:1, i, :, :]))
    
        csm = c2r_kdata(csm) # (coil, echo, row, col, 2) with last dimension real&imag

        # Fully sampled kspace data
        kdata = readcfl(dataFD_sense_echo + 'kdata_slice_{}'.format(idx))
        kdata = np.transpose(kdata, (2, 3, 0, 1))  # (coil, echo, row, col)
        kdata = c2r_kdata(kdata) # (coil, echo, row, col, 2) with last dimension real&imag

        # brain tissue mask
        brain_mask = np.real(readcfl(dataFD_sense_echo + 'mask_slice_{}'.format(idx)))  # (row, col)
        if self.echo_cat:
            brain_mask = np.repeat(brain_mask[np.newaxis, ...], self.necho*2, axis=0) # (2*echo, row, col)
        else:
            brain_mask = np.repeat(brain_mask[np.newaxis, ...], 2, axis=0) # (2, row, col)
            brain_mask = np.repeat(brain_mask[:, np.newaxis, ...], self.necho, axis=1)# (2, echo, row, col)
        
        if self.normalization == 0:
            return kdata, org, csm, brain_mask





        

        


    
