import os
import numpy as np
from torch.utils import data
from utils.data import *
from utils.loss import *
from utils.operators import *


class kdata_multi_echo_CBIC_prosp(data.Dataset):
    '''
        Dataloader of multi-echo GRE data from GE scanner with prospective under-sampling (CBIC scanner)
    '''

    def __init__(self,
        rootDir = '/data/Jinwei/QSM_raw_CBIC',
        contrast = 'MultiEcho',
        necho = 10, # number of echos
        nrow = 206,
        ncol = 80,
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
                self.nsamples = 800
            elif split == 'val':
                self.nsamples = 200
            elif split == 'test':
                self.nsamples = 200
                if subject == 0:
                    self.subject = 'junghun2'
                elif subject == 1:
                    self.subject = 'chao2'
                elif subject == 2:
                    self.subject = 'alexey2'
                elif subject == 3:
                    self.subject = 'liangdong2'
                elif subject == 4:
                    self.subject = 'fenglei2'
                elif subject == 5:
                    self.subject = 'jiahao2'
                elif subject == 6:
                    self.subject = 'dom2'
                elif subject == 7:
                    self.subject = 'hangwei2'
                elif subject == 8:
                    self.subject = 'wenxin2'
                elif subject == 9:
                    self.subject = 'hanxuan2'
                elif subject == 10:
                    self.subject = 'qihao2'
                print("Test on {} with prospective under-sampling".format(self.subject))
        self.augmentations = augmentations
        self.augmentation = self.augmentations[0]
        self.augSize = len(self.augmentations)
        self.augIndex = 0
        self.batchSize = batchSize
        self.batchIndex = 0
        self.recon_inputs = np.zeros((self.nsamples, self.nrow, self.ncol, 6)) + 1j * np.zeros((self.nsamples, self.nrow, self.ncol, 6))
        if split == 'train':
            self.recon_inputs[:200, ...] = load_mat(rootDir+'/data_cfl/20%train2/iField_bcrnn=1_loupe=0_solver=1_sub=0_train2.mat', 'Recons')
            self.recon_inputs[200:400, ...] = load_mat(rootDir+'/data_cfl/20%train2/iField_bcrnn=1_loupe=0_solver=1_sub=1_train2.mat', 'Recons')
            self.recon_inputs[400:600, ...] = load_mat(rootDir+'/data_cfl/20%train2/iField_bcrnn=1_loupe=0_solver=1_sub=2_train2.mat', 'Recons')
        elif split == 'val':
            self.recon_inputs[:200, ...] = load_mat(rootDir+'/data_cfl/20%train2/iField_bcrnn=1_loupe=0_solver=1_sub=0_val2.mat', 'Recons')
        elif split == 'test':
            # self.recon_inputs[:200, ...] = load_mat(rootDir+'/data_cfl/20%train2/iField_bcrnn=1_loupe=0_solver=1_sub={}_test2.mat'.format(subject), 'Recons')
            self.recon_inputs[:200, ...] = load_mat(rootDir+'/data_cfl/20%train2/iField_bcrnn=1_loupe=0_solver=1_sub=2_test2.mat', 'Recons')
            # # to reconstruct LLR QSM
            # self.iField = load_mat(rootDir+'/data_cfl/{}/iField_llr_loupe=-1_sub={}.mat'.format(self.subject, subject), 'iField_llr')

    def __len__(self):

        return self.nsamples * self.augSize


    def __getitem__(self, idx):
        recon_input = self.recon_inputs[idx, ...]
        # # to reconstruct LLR QSM
        # recon_input = self.iField[idx, ...]
        recon_input =  c2r(recon_input, self.echo_cat)  # echo_cat == 1: (2*echo, row, col) with first dimension real&imag concatenated for all echos 
                                        # echo_cat == 0: (2, row, col, echo)

        if self.split == 'train':
            subject = 0
            while (True):
                if (idx - 200 < 0):
                    break
                else:
                    idx -= 200
                    subject += 1
            if subject == 0:
                dataFD_prosp = self.rootDir + '/data_cfl/fenglei2/10_loupe={}_cc_slices_sense_echo/'
            elif subject == 1:
                dataFD_prosp = self.rootDir + '/data_cfl/jiahao2/10_loupe={}_cc_slices_sense_echo/'
            elif subject == 2:
                dataFD_prosp = self.rootDir + '/data_cfl/wenxin2/10_loupe={}_cc_slices_sense_echo/'
            elif subject == 3:
                dataFD_prosp = self.rootDir + '/data_cfl/hanxuan2/10_loupe={}_cc_slices_sense_echo/'
            dataFD = self.rootDir + '/data_cfl/jiahao2/full_cc_slices_sense_echo/'

        elif self.split == 'val':
            dataFD = self.rootDir + '/data_cfl/jiahao2/full_cc_slices_sense_echo/'
        
        elif self.split == 'test':
            # dataFD_prosp = self.rootDir + '/data_cfl/' + self.subject + '/10_loupe={}_cc_slices_sense_echo/'.format(self.loupe)
            # dataFD_prosp = self.rootDir + '/data_cfl/' + self.subject + '/10_loupe={}_cc_slices_sense_echo_FA=25/'.format(self.loupe)
            dataFD_prosp = self.rootDir + '/data_cfl/' + self.subject + '/10_loupe={}_cc_slices_sense_echo_Necho=7/'.format(self.loupe)
            # dataFD = self.rootDir + '/data_cfl/' + self.subject + '/full_cc_slices/'
            dataFD = self.rootDir + '/data_cfl/jiahao2/full_cc_slices/'

        idx += 30

        if (self.batchIndex == self.batchSize):
            self.batchIndex = 0
            self.augIndex += 1
            self.augIndex = self.augIndex % self.augSize
            self.augmentation = self.augmentations[self.augIndex]
        self.batchIndex += 1

        org = readcfl(dataFD + 'fully_slice_{}'.format(idx))  # (row, col, echo)
        org = org[..., :self.necho]
        org =  c2r(org, self.echo_cat)  # echo_cat == 1: (2*echo, row, col) with first dimension real&imag concatenated for all echos 
                                        # echo_cat == 0: (2, row, col, echo)

        if self.echo_cat == 0:
            org = np.transpose(org, (0, 3, 1, 2)) # (2, echo, row, col)
            recon_input = np.transpose(recon_input, (0, 3, 1, 2))
        
        # Coil sensitivity maps
        csm = readcfl(dataFD_prosp + 'sensMaps_slice_{}'.format(idx))
        # csm = np.transpose(csm, (2, 0, 1))[:, np.newaxis, ...]  # (coil, 1, row, col)
        # csm = np.repeat(csm, self.necho, axis=1)  # (coil, echo, row, col)
        csm = np.transpose(csm, (2, 3, 0, 1))  # (coil, echo, row, col)
        for i in range(self.necho):
            csm[:, i, :, :] = csm[:, i, :, :] * np.exp(-1j * np.angle(csm[0:1, i, :, :]))
        csm = c2r_kdata(csm) # (coil, echo, row, col, 2) with last dimension real&imag

        # Coil sensitivity maps from central kspace data
        csm_lowres = readcfl(dataFD + 'sensMaps_lowres_slice_{}'.format(idx))
        csm_lowres = np.transpose(csm_lowres, (2, 0, 1))[:, np.newaxis, ...]  # (coil, 1, row, col)
        csm_lowres = np.repeat(csm_lowres, self.necho, axis=1)  # (coil, echo, row, col)
        csm_lowres = c2r_kdata(csm_lowres) # (coil, echo, row, col, 2) with last dimension real&imag

        # Fully sampled kspace data
        kdata = readcfl(dataFD_prosp + 'kdata_slice_{}'.format(idx))
        kdata = np.transpose(kdata, (2, 3, 0, 1))  # (coil, echo, row, col)
        kdata = c2r_kdata(kdata) # (coil, echo, row, col, 2) with last dimension real&imag

        # brain tissue mask
        brain_mask = np.real(readcfl(dataFD + 'mask_slice_{}'.format(idx)))  # (row, col)
        brain_mask_erode = np.real(readcfl(dataFD + 'mask_erode_slice_{}'.format(idx)))  # (row, col)
        if self.echo_cat:
            brain_mask = np.repeat(brain_mask[np.newaxis, ...], self.necho*2, axis=0) # (2*echo, row, col)
            brain_mask_erode = np.repeat(brain_mask_erode[np.newaxis, ...], self.necho*2, axis=0) # (2*echo, row, col)
        else:
            brain_mask = np.repeat(brain_mask[np.newaxis, ...], 2, axis=0) # (2, row, col)
            brain_mask = np.repeat(brain_mask[:, np.newaxis, ...], self.necho, axis=1)# (2, echo, row, col)
            brain_mask_erode = np.repeat(brain_mask_erode[np.newaxis, ...], 2, axis=0) # (2, row, col)
            brain_mask_erode = np.repeat(brain_mask_erode[:, np.newaxis, ...], self.necho, axis=1)# (2, echo, row, col)
        
        if self.normalization == 0:
            return kdata, org, recon_input, csm, csm_lowres, brain_mask, brain_mask_erode





        

        


    
