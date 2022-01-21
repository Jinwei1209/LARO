import os
import numpy as np
from torch.utils import data
from utils.data import *
from utils.loss import *
from utils.operators import *


class kdata_T1T2QSM_MS(data.Dataset):
    '''
        Dataloader of MS lesion multi-echo GRE data from Siemens scanner (0.75*0.75*3 mm3 resolution) with 
        kdata generated from iField
    '''

    def __init__(self,
        rootDir = '/data2/Jinwei/T1T2QSM_MS/data',
        contrast = 'MultiContrast',
        necho = 13, # number of echos (generalized echo dimension, including T1w and T2w contrasts as the first two dimensions)
        nrow = 256,
        ncol = 50,
        split = 'train',
        subject_test = 0,  # subject for testing
        subject_val = 7,
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
        if contrast == 'MultiContrast':
            if split == 'test':
                self.subject_test = subject_test
                print("Test on {}".format(self.subject_test))
        self.augmentations = augmentations
        self.augmentation = self.augmentations[0]
        self.augSize = len(self.augmentations)
        self.augIndex = 0
        self.batchSize = batchSize
        self.batchIndex = 0
        self.imgs_all, self.Masks = [], []
        if split == 'train':
            print('Loading training dataset')
            for i in range(1, subject_val):
                print('case: {}'.format(i))
                img_all = load_mat(self.rootDir+'/data/img_all_{}.mat'.format(i), 'img_all')  # (nrow, nslice, ncol, necho)
                padding_dim = (self.ncol - img_all.shape[2]) // 2
                if padding_dim != 0:
                    img_all = np.pad(img_all, [(0, 0), (0, 0), (padding_dim, padding_dim), (0, 0)], mode='constant')
                img_all = np.transpose(img_all, (1, 0, 2, 3))
                Mask = load_mat(self.rootDir+'/data/img_all_{}.mat'.format(i), 'Mask') # (nrow, nslice, ncol)
                if padding_dim != 0:
                    Mask = np.pad(Mask, [(0, 0), (0, 0), (padding_dim, padding_dim)], mode='constant')
                Mask = np.transpose(Mask, (1, 0, 2))
                Mask = np.repeat(Mask[..., np.newaxis], self.necho, axis=-1)
                self.imgs_all.append(img_all / np.max(abs(img_all[Mask==1])) / 2)  # divided by 2 is a magic number
                self.Masks.append(Mask)
        elif split == 'val':
            print('Loading validation dataset')
            print('case: {}'.format(subject_val))
            img_all = load_mat(self.rootDir+'/data/img_all_{}.mat'.format(subject_val), 'img_all')  # (nrow, nslice, ncol, necho)
            padding_dim = (self.ncol - img_all.shape[2]) // 2
            if padding_dim != 0:
                img_all = np.pad(img_all, [(0, 0), (0, 0), (padding_dim, padding_dim), (0, 0)], mode='constant')
            img_all = np.transpose(img_all, (1, 0, 2, 3))
            Mask = load_mat(self.rootDir+'/data/img_all_{}.mat'.format(subject_val), 'Mask') # (nrow, nslice, ncol)
            if padding_dim != 0:
                Mask = np.pad(Mask, [(0, 0), (0, 0), (padding_dim, padding_dim)], mode='constant')
            Mask = np.transpose(Mask, (1, 0, 2))
            Mask = np.repeat(Mask[..., np.newaxis], self.necho, axis=-1)
            self.imgs_all.append(img_all / np.max(abs(img_all[Mask==1])) / 2)  # divided by 2 is a magic number
            self.Masks.append(Mask)
        elif split == 'test':
            print('Loading test dataset {}'.format(self.subject_test))
            img_all = load_mat(self.rootDir+'/data/img_all_{}.mat'.format(self.subject_test), 'img_all')  # (nrow, nslice, ncol, necho)
            padding_dim = (self.ncol - img_all.shape[2]) // 2
            if padding_dim != 0:
                img_all = np.pad(img_all, [(0, 0), (0, 0), (padding_dim, padding_dim), (0, 0)], mode='constant')
            img_all = np.transpose(img_all, (1, 0, 2, 3))
            Mask = load_mat(self.rootDir+'/data/img_all_{}.mat'.format(self.subject_test), 'Mask') # (nrow, nslice, ncol)
            if padding_dim != 0:
                Mask = np.pad(Mask, [(0, 0), (0, 0), (padding_dim, padding_dim)], mode='constant')
            Mask = np.transpose(Mask, (1, 0, 2))
            Mask = np.repeat(Mask[..., np.newaxis], self.necho, axis=-1)
            self.imgs_all.append(img_all / np.max(abs(img_all[Mask==1])) / 2)  # divided by 2 is a magic number
            self.Masks.append(Mask)
        self.imgs_all = np.concatenate(self.imgs_all, axis=0)
        self.Masks = np.concatenate(self.Masks, axis=0)
        self.nsamples = self.imgs_all.shape[0]
        print('{} samples in total'.format(self.nsamples))


    def __len__(self):

        return self.nsamples * self.augSize


    def __getitem__(self, idx):
        if (self.batchIndex == self.batchSize):
            self.batchIndex = 0
            self.augIndex += 1
            self.augIndex = self.augIndex % self.augSize
            self.augmentation = self.augmentations[self.augIndex]
        self.batchIndex += 1

        org = self.imgs_all[idx, ...]  # (row, col, echo)
        kdata = self.generateKdata(org[np.newaxis, ...]) # (coil, echo, row, col, 2) with last dimension real&imag
        org = c2r(org, self.echo_cat)  # echo_cat == 1: (2*echo, row, col) with first dimension real&imag concatenated for all echos 
                                        # echo_cat == 0: (2, row, col, echo)
        if self.echo_cat == 0:
            org = np.transpose(org, (0, 3, 1, 2)) # (2, echo, row, col)
        # Coil sensitivity maps
        csm = np.ones((1, 1, self.nrow, self.ncol))  # (coil, 1, row, col)
        csm = np.repeat(csm, self.necho, axis=1)  # (coil, echo, row, col)
        csm = c2r_kdata(csm) # (coil, echo, row, col, 2) with last dimension real&imag

        # brain tissue mask
        brain_mask = np.transpose(self.Masks[idx, ...], (2, 0, 1))  # (echo, row, col)
        if self.echo_cat:
            brain_mask = np.repeat(brain_mask, 2, axis=0) # (2*echo, row, col)
        else:
            brain_mask = np.repeat(brain_mask[np.newaxis, ...], 2, axis=0) # (2, echo, row, col)
        return kdata, org, csm, brain_mask


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




        

        


    
