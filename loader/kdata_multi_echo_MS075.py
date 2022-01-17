import os
import numpy as np
from torch.utils import data
from utils.data import *
from utils.loss import *
from utils.operators import *


class kdata_multi_echo_MS075(data.Dataset):
    '''
        Dataloader of MS lesion multi-echo GRE data from Siemens scanner (0.75*0.75*3 mm3 resolution) with 
        kdata generated from iField
    '''

    def __init__(self,
        rootDir = '/data/Jinwei/QSM_iField_MS',
        contrast = 'MultiEcho',
        necho = 1, # number of echos
        nrow = 210,
        ncol = 56,
        split = 'train',
        subject = 0,  # 0: iField1, 1: iField2, 2: iField3
        test_ID = None,
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
                self.nsamples = 260 * 10
            elif split == 'val':
                self.nsamples = 260
            elif split == 'test':
                self.nsamples = 260
                if subject == 0:
                    self.subject = '21'
                elif subject == 1:
                    self.subject = '22'
                elif subject == 2:
                    self.subject = '23'
                if test_ID is not None:
                    self.subject = str(test_ID+1)
                print("Test on {}".format(self.subject))
        self.augmentations = augmentations
        self.augmentation = self.augmentations[0]
        self.augSize = len(self.augmentations)
        self.augIndex = 0
        self.batchSize = batchSize
        self.batchIndex = 0
        self.iField = np.zeros((self.nsamples, self.nrow, self.ncol, self.necho)) + 1j * np.zeros((self.nsamples, self.nrow, self.ncol, self.necho))
        self.Mask = np.zeros((self.nsamples, self.nrow, self.ncol, self.necho))
        self.Mask_erode = np.zeros((self.nsamples, self.nrow, self.ncol, self.necho))
        if split == 'train':
            print('Loading training dataset')
            for i in range(1, 11):
                print('case: {}'.format(i))
                iField = load_mat(self.rootDir+'/data_MS075/iField{}.mat'.format(i), 'iField')
                iField = np.transpose(iField, (1, 0, 2, 3))[30:290, 25:235, ...]
                Mask = load_mat(self.rootDir+'/data_MS075/iField{}.mat'.format(i), 'Mask')
                Mask = np.transpose(Mask, (1, 0, 2))[30:290, 25:235, ...]
                Mask = np.repeat(Mask[..., np.newaxis], self.necho, axis=-1)
                self.iField[260*(i-1):260*i, ...] = iField[:, :, :, 4:5]
                self.Mask[260*(i-1):260*i, ...] = Mask
        elif split == 'val':
            print('Loading validation dataset')
            print('case: 11')
            iField = load_mat(self.rootDir+'/data_MS075/iField11.mat', 'iField')
            iField = np.transpose(iField, (1, 0, 2, 3))[30:290, 25:235, ...]
            Mask = load_mat(self.rootDir+'/data_MS075/iField11.mat', 'Mask')
            Mask = np.transpose(Mask, (1, 0, 2))[30:290, 25:235, ...]
            Mask = np.repeat(Mask[..., np.newaxis], self.necho, axis=-1)
            self.iField[:260, ...] = iField[:, :, :, 4:5]
            self.Mask[:260, ...] = Mask
        elif split == 'test':
            print('Loading test dataset {}'.format(self.subject))
            iField = load_mat(self.rootDir+'/data_MS075/iField{}.mat'.format(self.subject), 'iField')
            iField = np.transpose(iField, (1, 0, 2, 3))[30:290, 25:235, ...]
            Mask = load_mat(self.rootDir+'/data_MS075/iField{}.mat'.format(self.subject), 'Mask')
            Mask = np.transpose(Mask, (1, 0, 2))[30:290, 25:235, ...]
            Mask = np.repeat(Mask[..., np.newaxis], self.necho, axis=-1)
            self.iField[:260, ...] = iField[:, :, :, 4:5]
            self.Mask[:260, ...] = Mask


    def __len__(self):

        return self.nsamples * self.augSize


    def __getitem__(self, idx):
        if (self.batchIndex == self.batchSize):
            self.batchIndex = 0
            self.augIndex += 1
            self.augIndex = self.augIndex % self.augSize
            self.augmentation = self.augmentations[self.augIndex]
        self.batchIndex += 1

        org = self.iField[idx, ...]  # (row, col, echo)
        kdata = self.generateKdata(org[np.newaxis, ...]) # (coil, echo, row, col, 2) with last dimension real&imag
        org = c2r(org, self.echo_cat)  # echo_cat == 1: (2*echo, row, col) with first dimension real&imag concatenated for all echos 
                                        # echo_cat == 0: (2, row, col, echo)
        recon_input = org
        if self.echo_cat == 0:
            org = np.transpose(org, (0, 3, 1, 2)) # (2, echo, row, col)
            recon_input = np.transpose(recon_input, (0, 3, 1, 2))
        # Coil sensitivity maps
        csm = np.ones((1, 1, self.nrow, self.ncol))  # (coil, 1, row, col)
        csm = np.repeat(csm, self.necho, axis=1)  # (coil, echo, row, col)
        csm = c2r_kdata(csm) # (coil, echo, row, col, 2) with last dimension real&imag

        # Coil sensitivity maps from central kspace data
        csm_lowres = np.ones((1, 1, 25, 25))  # (coil, 1, row, col)
        csm_lowres = np.repeat(csm_lowres, self.necho, axis=1)  # (coil, echo, row, col)
        csm_lowres = c2r_kdata(csm_lowres) # (coil, echo, row, col, 2) with last dimension real&imag

        # brain tissue mask
        brain_mask = np.transpose(self.Mask[idx, ...], (2, 0, 1))  # (echo, row, col)
        brain_mask_erode = np.transpose(self.Mask_erode[idx, ...], (2, 0, 1))  # (echo, row, col)
        if self.echo_cat:
            brain_mask = np.repeat(brain_mask, 2, axis=0) # (2*echo, row, col)
            brain_mask_erode = np.repeat(brain_mask_erode, 2, axis=0) # (2*echo, row, col)
        else:
            brain_mask = np.repeat(brain_mask[np.newaxis, ...], 2, axis=0) # (2, echo, row, col)
            brain_mask_erode = np.repeat(brain_mask_erode[np.newaxis, ...], 2, axis=0) # (2, echo, row, col)
        return kdata, org, recon_input, csm, csm_lowres, brain_mask, brain_mask_erode


    def generateKdata(self, org, sigma=0.01):
        '''
            org: (coil, echo, row, col)
        '''
        noise = np.random.randn(1, self.nrow, self.ncol, self.necho) \
                + 1j*np.random.randn(1, self.nrow, self.ncol, self.necho)
        noise = noise * (sigma / np.sqrt(2.)) * np.sqrt(self.nrow*self.ncol)
        y = np.fft.fft2(org, axes=(1, 2)) + noise
        y = np.fft.fftshift(y, axes=(1, 2))
        kdata = c2r_kdata(np.transpose(y, (0, 3, 1, 2)))  # (coil, echo, row, col, 2) with last dimension real&imag
        return kdata




        

        


    
