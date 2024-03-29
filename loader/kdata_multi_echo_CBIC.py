import os
import numpy as np
from torch.utils import data
from utils.data import *
from utils.loss import *
from utils.operators import *


class kdata_multi_echo_CBIC(data.Dataset):
    '''
        Dataloader of multi-echo GRE data from CBIC scanners (scanner: 0 (GE) / 1 (Siemens))
    '''

    def __init__(self,
        rootDir = '/data/Jinwei/QSM_raw_CBIC',
        contrast = 'MultiEcho',
        necho = 10, # number of echos
        nrow = 206,
        ncol = 80,
        split = 'train',
        subject = 0,  # 0: junghun, 1: chao, 2: alexey
        normalization = 0,  # flag to normalize the data
        echo_cat = 1, # flag to concatenate echo dimension into channel
        batchSize = 1,
        augmentations = [None],
        scanner = 0,
    ):

        self.rootDir = rootDir
        self.contrast = contrast
        self.necho = necho
        self.normalization = normalization
        self.echo_cat = echo_cat
        self.split = split
        self.nrow = nrow
        self.ncol = ncol
        self.scanner = scanner
        scanners = ['GE', 'Siemens']
        print("kspace data on {} scanner".format(scanners[self.scanner]))
        if contrast == 'MultiEcho':
            if split == 'train':
                self.nsamples = 1600
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
                elif subject == 5:
                    self.subject = 'jiahao2'
                print("Test on {}".format(self.subject))
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
            # self.iField = load_mat(rootDir+'/data_cfl/{}/iField_llr_full_sub={}.mat'.format(self.subject[:-1], subject), 'iField_llr')

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
                dataFD = self.rootDir + '/data_cfl/jiahao2/full_cc_slices/'
                dataFD_sense_echo = self.rootDir + '/data_cfl/thanh2/full_cc_slices_sense_echo/'
            elif subject == 1:
                dataFD = self.rootDir + '/data_cfl/jiahao2/full_cc_slices/'
                dataFD_sense_echo = self.rootDir + '/data_cfl/jinwei2/full_cc_slices_sense_echo/'
            elif subject == 2:
                dataFD = self.rootDir + '/data_cfl/jiahao2/full_cc_slices/'
                dataFD_sense_echo = self.rootDir + '/data_cfl/qihao2/full_cc_slices_sense_echo/'
            elif subject == 3:
                dataFD = self.rootDir + '/data_cfl/jiahao2/full_cc_slices/'
                dataFD_sense_echo = self.rootDir + '/data_cfl/hang2/full_cc_slices_sense_echo/'
            elif subject == 4:
                dataFD = self.rootDir + '/data_cfl/jiahao2/full_cc_slices/'
                dataFD_sense_echo = self.rootDir + '/data_cfl/dominick2/full_cc_slices_sense_echo/'
            elif subject == 5:
                dataFD = self.rootDir + '/data_cfl/jiahao2/full_cc_slices/'
                dataFD_sense_echo = self.rootDir + '/data_cfl/hangwei2/full_cc_slices_sense_echo/'
            elif subject == 6:
                dataFD = self.rootDir + '/data_cfl/jiahao2/full_cc_slices/'
                dataFD_sense_echo = self.rootDir + '/data_cfl/kelly2/full_cc_slices_sense_echo/'
            elif subject == 7:
                dataFD = self.rootDir + '/data_cfl/jiahao2/full_cc_slices/'
                dataFD_sense_echo = self.rootDir + '/data_cfl/feng2/full_cc_slices_sense_echo/'
            # if subject == 0:
            #     dataFD = self.rootDir + '/data_cfl/jiahao2/full_cc_slices/'
            #     dataFD_sense_echo = self.rootDir + '/data_cfl/feng2/full_cc_slices_sense_echo/'
            # elif subject == 1:
            #     dataFD = self.rootDir + '/data_cfl/jiahao2/full_cc_slices/'
            #     dataFD_sense_echo = self.rootDir + '/data_cfl/junghun2/full_cc_slices_sense_echo/'
            # elif subject == 2:
            #     dataFD = self.rootDir + '/data_cfl/jiahao2/full_cc_slices/'
            #     dataFD_sense_echo = self.rootDir + '/data_cfl/chao2/full_cc_slices_sense_echo/'
            # elif subject == 3:
            #     dataFD = self.rootDir + '/data_cfl/jiahao2/full_cc_slices/'
            #     dataFD_sense_echo = self.rootDir + '/data_cfl/alexey2/full_cc_slices_sense_echo/'

        elif self.split == 'val':
            dataFD = self.rootDir + '/data_cfl/jiahao2/full_cc_slices/'
            dataFD_sense_echo = self.rootDir + '/data_cfl/jiahao2/full_cc_slices_sense_echo/'
        
        elif self.split == 'test':
            # dataFD = self.rootDir + '/data_cfl/' + self.subject + '/full_cc_slices/'
            dataFD = self.rootDir + '/data_cfl/jiahao2/full_cc_slices/'
            # dataFD_sense_echo = self.rootDir + '/data_cfl/' + self.subject + '/full_cc_slices_sense_echo/'
            dataFD_sense_echo = self.rootDir + '/data_cfl/' + self.subject + '/full_cc_slices_sense_echo_FA=25/'
            if self.scanner == 1:
                dataFD_sense_echo = self.rootDir + '/data_cfl/' + self.subject + '/full_cc_slices_sense_echo_Siemens2/'
            # dataFD_sense_echo = self.rootDir + '/data_cfl/' + self.subject + '/full_cc_slices_sense_echo/'

        idx += 30

        if self.scanner == 1:
            idx += 120  # for Siemens scanner, +150 in total for idx

        if (self.batchIndex == self.batchSize):
            self.batchIndex = 0
            self.augIndex += 1
            self.augIndex = self.augIndex % self.augSize
            self.augmentation = self.augmentations[self.augIndex]
        self.batchIndex += 1

        org = readcfl(dataFD_sense_echo + 'fully_slice_{}'.format(idx))  # (row, col, echo)
        org = org[..., :self.necho]
        org =  c2r(org, self.echo_cat)  # echo_cat == 1: (2*echo, row, col) with first dimension real&imag concatenated for all echos 
                                        # echo_cat == 0: (2, row, col, echo)

        if self.echo_cat == 0:
            org = np.transpose(org, (0, 3, 1, 2)) # (2, echo, row, col)
            recon_input = np.transpose(recon_input, (0, 3, 1, 2))
        
        # # Option 1: coil sensitivity maps from the 1st echo
        # csm = readcfl(dataFD + 'sensMaps_slice_{}'.format(idx))
        # csm = np.transpose(csm, (2, 0, 1))[:, np.newaxis, ...]  # (coil, 1, row, col)
        # csm = np.repeat(csm, self.necho, axis=1)  # (coil, echo, row, col)
        # for i in range(self.necho):
        #     csm[:, i, :, :] = csm[:, i, :, :] * np.exp(-1j * np.angle(csm[0:1, i, :, :]))

        # Option 2: csms estimated from each echo
        csm = readcfl(dataFD_sense_echo + 'sensMaps_slice_{}'.format(idx))
        csm = csm[..., :self.necho]
        csm = np.transpose(csm, (2, 3, 0, 1))  # (coil, echo, row, col)
        for i in range(self.necho):
            csm[:, i, :, :] = csm[:, i, :, :] * np.exp(-1j * np.angle(csm[0:1, i, :, :]))
    
        csm = c2r_kdata(csm) # (coil, echo, row, col, 2) with last dimension real&imag

        # Coil sensitivity maps from central kspace data
        csm_lowres = readcfl(dataFD + 'sensMaps_lowres_slice_{}'.format(idx%256))
        csm_lowres = csm_lowres[..., :self.necho]
        csm_lowres = np.transpose(csm_lowres, (2, 0, 1))[:, np.newaxis, ...]  # (coil, 1, row, col)
        csm_lowres = np.repeat(csm_lowres, self.necho, axis=1)  # (coil, echo, row, col)
        csm_lowres = c2r_kdata(csm_lowres) # (coil, echo, row, col, 2) with last dimension real&imag

        # Fully sampled kspace data
        kdata = readcfl(dataFD_sense_echo + 'kdata_slice_{}'.format(idx))
        kdata = kdata[..., :self.necho]
        kdata = np.transpose(kdata, (2, 3, 0, 1))  # (coil, echo, row, col)
        kdata = c2r_kdata(kdata) # (coil, echo, row, col, 2) with last dimension real&imag

        # brain tissue mask
        brain_mask = np.real(readcfl(dataFD_sense_echo + 'mask_slice_{}'.format(idx)))  # (row, col)
        brain_mask = brain_mask[..., :self.necho]
        brain_mask_erode = np.real(readcfl(dataFD + 'mask_erode_slice_{}'.format(idx%256)))  # (row, col)
        brain_mask_erode = brain_mask_erode[..., :self.necho]
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
        else:
            # return kdata*0.5e7, org*0.5e7, recon_input, csm, csm_lowres, brain_mask, brain_mask_erode  # for Siemens
            return kdata*0.62e7, org*0.62e7, recon_input, csm, csm_lowres, brain_mask, brain_mask_erode  # for Siemens2






        

        


    
