import os
import numpy as np
from torch.utils import data
from utils.data import *
from utils.loss import *
from utils.operators import *


class kdata_multi_echo_GE(data.Dataset):
    '''
        Dataloader of multi-echo GRE data from GE scanner (12-coil scanner)
    '''

    folderMatcher = {
        'MultiEcho': '/megre_slice_GE/'
    }

    dataRange = {
        'train': ['200', '800'], 
        'val': ['0', '200'],
        # 'test': ['800', '1000']
        'test': ['200', '400']
    }
    

    def __init__(self,
        rootDir = '/data/Jinwei/Multi_echo_slice_recon_GE',
        contrast = 'MultiEcho',
        necho = 10, # number of echos
        split = 'train',
        normalization = 0,  # flag to normalize the data
        echo_cat = 1, # flag to concatenate echo dimension into channel
        batchSize = 1,
        augmentations = [None]
    ):

        self.rootDir = rootDir
        self.dataFD = rootDir + self.folderMatcher[contrast]
        self.contrast = contrast
        self.necho = necho
        self.normalization = normalization
        self.echo_cat = echo_cat
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

        # self.gen_target()


    def gen_target(self):
        '''
            generate target from calculated m0, R2s, f0 and p
        '''
        rootDir = '/data/Jinwei/Multi_echo_kspace/data_parameters'
        subject_IDs = ['sub1', 'sub2', 'sub3', 'sub4', 'sub5']
        TEs = [0.003224, 0.007108, 0.010992, 0.014876, 0.018760, 0.022644, 0.026528, 0.030412, 0.034296, 0.038180]
        self.orgs_gen = np.zeros([1000, 206, 80, self.necho]) + 1j * np.zeros([1000, 206, 80, self.necho])

        for idx, subject_ID in enumerate(subject_IDs):
            print('Processing {}'.format(subject_ID))
            dataFD = rootDir + '/' + subject_ID
            if subject_ID == 'sub1':
                # for the fully-sampled parameters
                m0 = load_mat(dataFD+'/m0.mat', 'm0')[np.newaxis, 15:215, ...]
                r2s = load_mat(dataFD+'/R2s.mat', 'R2s')[np.newaxis, 15:215, ...]
                f0 = load_mat(dataFD+'/f0.mat', 'f0')[np.newaxis, 15:215, ...]
                p = load_mat(dataFD+'/p.mat', 'p')[np.newaxis, 15:215, ...]
                # for the fully-sampled parameters
                for i, te in enumerate(TEs):
                    self.orgs_gen[idx*200:(idx+1)*200, :, :, i] = - m0 * np.exp(- r2s * te) * np.exp(1j * (f0 + p * te))

                # m0 = load_mat(dataFD+'/0.2/m0_10.mat', 'm0')
                # r2s = load_mat(dataFD+'/0.2/R2s_10.mat', 'R2s')
                # f0 = load_mat(dataFD+'/0.2/f0_10.mat', 'f0')
                # p = load_mat(dataFD+'/0.2/p_10.mat', 'p')

                # for i, te in enumerate(TEs):
                #     self.orgs_gen[idx*200:(idx+1)*200, :, :, i] = m0 * np.exp(- r2s * te) * np.exp(1j * (f0 + p * te))

            else:
                # for the fully-sampled parameters
                m0 = load_mat(dataFD+'/m0.mat', 'm0')[np.newaxis, 30:230, ...]
                r2s = load_mat(dataFD+'/R2s.mat', 'R2s')[np.newaxis, 30:230, ...]
                f0 = load_mat(dataFD+'/f0.mat', 'f0')[np.newaxis, 30:230, ...]
                p = load_mat(dataFD+'/p.mat', 'p')[np.newaxis, 30:230, ...]

                # m0 = load_mat(dataFD+'/0.2/m0_10.mat', 'm0')
                # r2s = load_mat(dataFD+'/0.2/R2s_10.mat', 'R2s')
                # f0 = load_mat(dataFD+'/0.2/f0_10.mat', 'f0')
                # p = load_mat(dataFD+'/0.2/p_10.mat', 'p')
                
                for i, te in enumerate(TEs):
                    self.orgs_gen[idx*200:(idx+1)*200, :, :, i] = m0 * np.exp(- r2s * te) * np.exp(1j * (f0 + p * te))

        self.orgs_gen[0::2, ...] = - self.orgs_gen[0::2, ...]


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

        org = readcfl(self.dataFD + 'fully_slice_{}'.format(idx))  # (row, col, echo)
        org =  c2r(org, self.echo_cat)  # echo_cat == 1: (2*echo, row, col) with first dimension real&imag concatenated for all echos 
                                        # echo_cat == 0: (2, row, col, echo)
        # org_gen = self.orgs_gen[idx, ...]
        # org_gen = c2r(org_gen, self.echo_cat)
        org_gen = org

        if self.echo_cat == 0:
            org = np.transpose(org, (0, 3, 1, 2)) # (2, echo, row, col)
            org_gen = np.transpose(org_gen, (0, 3, 1, 2))
            
        csm = readcfl(self.dataFD + 'sensMaps_slice_{}'.format(idx))
        csm = np.transpose(csm, (2, 0, 1))[:, np.newaxis, ...]  # (coil, 1, row, col)
        csm = np.repeat(csm, self.necho, axis=1)  # (coil, echo, row, col)
        csm = c2r_kdata(csm) # (coil, echo, row, col, 2) with last dimension real&imag

        kdata = readcfl(self.dataFD + 'kdata_slice_{}'.format(idx))
        kdata = np.transpose(kdata, (2, 3, 0, 1))  # (coil, echo, row, col)
        kdata = c2r_kdata(kdata) # (coil, echo, row, col, 2) with last dimension real&imag

        brain_mask = np.real(readcfl(self.dataFD + 'mask_slice_{}'.format(idx)))  # (row, col)
        if self.echo_cat:
            brain_mask = np.repeat(brain_mask[np.newaxis, ...], self.necho*2, axis=0) # (2*echo, row, col)
        else:
            brain_mask = np.repeat(brain_mask[np.newaxis, ...], 2, axis=0) # (2, row, col)
            brain_mask = np.repeat(brain_mask[:, np.newaxis, ...], self.necho, axis=1)# (2, echo, row, col)

        return kdata*self.normalization, org*self.normalization, csm, brain_mask


        

        


    
