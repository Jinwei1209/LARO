import os
import numpy as np
from torch.utils import data
from utils.data import *
from utils.loss import *
from utils.operators import *


class kdata_T1T2QSM_CBIC(data.Dataset):
    '''
        Dataloader of T1w+T2w+mGRE data from GE scanner (CBIC scanner)
    '''

    def __init__(self,
        rootDir = '/data4/Jinwei/T1T2QSM',
        contrast = 'MultiContrast',
        necho = 7, # number of echos in total
        necho_mGRE = 5,  # number of echos of mGRE data
        nrow = 206,
        ncol = 80,
        split = 'train',
        dataset_id = 0,  # 0: bipolar with vd9 sampling, 1: bipolar with vd11 sampling, 2: unipolar with vd11.
        subject = 0,  # 0: junghun, 1: chao, 2: alexey
        prosp_flag = 0,
        t2w_redesign_flag = 0,
        normalizations = [5, 10, 10],  # normalization factor for mGRE, T1w and T2w data
        echo_cat = 1, # flag to concatenate echo dimension into channel
        batchSize = 1,
        augmentations = [None]
    ):

        self.rootDir = rootDir
        self.contrast = contrast
        self.necho = necho
        self.necho_mGRE = necho_mGRE
        self.normalizations = normalizations
        self.echo_cat = echo_cat
        self.split = split
        self.dataset_id = dataset_id
        if self.dataset_id == 0:
            self.id = '1'
            self.echo_stride = 2
            self.necho = 7
            self.necho_mGRE = 5
        elif self.dataset_id == 1:
            self.id = '2'
            self.echo_stride = 2
            self.necho = 7
            self.necho_mGRE = 5
        elif self.dataset_id == 2:
            self.id = '3'
            self.echo_stride = 1
            self.necho = 11
            self.necho_mGRE = 9
        elif self.dataset_id == 3:
            self.id = '4'
            self.echo_stride = 2
            self.necho = 7
            self.necho_mGRE = 5
        elif self.dataset_id == 5:
            self.id = '6'
            self.echo_stride = 2
            self.necho = 7
            self.necho_mGRE = 5
        self.prosp_flag = prosp_flag
        self.t2w_redesign_flag = t2w_redesign_flag
        if self.t2w_redesign_flag == 1:
            self.id += '_t2w_redesign'
        self.nrow = nrow
        self.ncol = ncol
        if contrast == 'MultiContrast':
            if split == 'train':
                self.nsamples = 256*4
            elif split == 'val':
                self.nsamples = 256
            elif split == 'test':
                self.nsamples = 256
                if subject == 0:
                    self.subject = 'qihao'
                elif subject == 1:
                    self.subject = 'jiahao'
                elif subject == 2:
                    self.subject = 'chao'
                elif subject == 3:
                    self.subject = 'qihao'
                print("Test on {}".format(self.subject))
        self.augmentations = augmentations
        self.augmentation = self.augmentations[0]
        self.augSize = len(self.augmentations)
        self.augIndex = 0
        self.batchSize = batchSize
        self.batchIndex = 0
        print("Use dataset: {}".format(self.dataset_id))
        if self.prosp_flag:
            print("Test on {} with prospective under-sampling".format(self.subject))


    def __len__(self):

        return self.nsamples * self.augSize


    def __getitem__(self, idx):

        if self.split == 'train':
            subject = 0
            while (True):
                if (idx - 256 < 0):
                    break
                else:
                    idx -= 256
                    subject += 1
            if subject == 0:
                dataFD_sense_echo = self.rootDir + '/data_cfl{}/chao/full_cc_slices_sense_echo/'.format(self.id)
            elif subject == 1:
                dataFD_sense_echo = self.rootDir + '/data_cfl{}/jiahao/full_cc_slices_sense_echo/'.format(self.id)
            elif subject == 2:
                dataFD_sense_echo = self.rootDir + '/data_cfl{}/hangwei/full_cc_slices_sense_echo/'.format(self.id)
            elif subject == 3:
                dataFD_sense_echo = self.rootDir + '/data_cfl{}/dom/full_cc_slices_sense_echo/'.format(self.id)
            dataFD_sense_echo_mask = dataFD_sense_echo
        elif self.split == 'val':
            dataFD_sense_echo = self.rootDir + '/data_cfl{}/qihao/full_cc_slices_sense_echo/'.format(self.id)
            dataFD_sense_echo_mask = dataFD_sense_echo
        elif self.split == 'test':
            if not self.prosp_flag:
                dataFD_sense_echo = self.rootDir + '/data_cfl{}/'.format(self.id) + self.subject + '/full_cc_slices_sense_echo/'
                dataFD_sense_echo_mask = dataFD_sense_echo
            else:
                dataFD_sense_echo = self.rootDir + '/data_cfl{}/'.format(self.id) + self.subject + '/kt_cc_slices_sense_echo/'
                dataFD_sense_echo_mask = self.rootDir + '/data_cfl{}/'.format(self.id) + self.subject + '/full_cc_slices_sense_echo/'

        if (self.batchIndex == self.batchSize):
            self.batchIndex = 0
            self.augIndex += 1
            self.augIndex = self.augIndex % self.augSize
            self.augmentation = self.augmentations[self.augIndex]
        self.batchIndex += 1

        org = readcfl(dataFD_sense_echo + 'fully_slice_{}'.format(idx))  # (row, col, echo)
        org = np.concatenate((org[..., 0:9:self.echo_stride], org[..., 9:11]), axis=-1)  # use only odd echoes
        org =  c2r(org, self.echo_cat)  # echo_cat == 1: (2*echo, row, col) with first dimension real&imag concatenated for all echos 
                                        # echo_cat == 0: (2, row, col, echo)

        if self.echo_cat == 0:
            org = np.transpose(org, (0, 3, 1, 2)) # (2, echo, row, col)

        # Option 2: csms estimated from each echo
        csm = readcfl(dataFD_sense_echo + 'sensMaps_slice_{}'.format(idx))
        csm = np.concatenate((csm[..., 0:9:self.echo_stride], csm[..., 9:11]), axis=-1)  # use only odd echoes
        csm = np.transpose(csm, (2, 3, 0, 1))  # (coil, echo, row, col)
        for i in range(self.necho):
            csm[:, i, :, :] = csm[:, i, :, :] * np.exp(-1j * np.angle(csm[0:1, i, :, :]))
    
        csm = c2r_kdata(csm) # (coil, echo, row, col, 2) with last dimension real&imag

        # Fully sampled kspace data
        kdata = readcfl(dataFD_sense_echo + 'kdata_slice_{}'.format(idx))
        kdata = np.concatenate((kdata[..., 0:9:self.echo_stride], kdata[..., 9:11]), axis=-1)  # use only odd echoes
        kdata = np.transpose(kdata, (2, 3, 0, 1))  # (coil, echo, row, col)
        kdata = c2r_kdata(kdata) # (coil, echo, row, col, 2) with last dimension real&imag

        # brain tissue mask
        brain_mask = np.real(readcfl(dataFD_sense_echo_mask + 'mask_slice_{}'.format(idx)))  # (row, col)
        if self.echo_cat:
            brain_mask = np.repeat(brain_mask[np.newaxis, ...], self.necho*2, axis=0) # (2*echo, row, col)
        else:
            brain_mask = np.repeat(brain_mask[np.newaxis, ...], 2, axis=0) # (2, row, col)
            brain_mask = np.repeat(brain_mask[:, np.newaxis, ...], self.necho, axis=1)# (2, echo, row, col)

        # normalize
        kdata[:, :self.necho_mGRE, ...] = kdata[:, :self.necho_mGRE, ...] * self.normalizations[0]
        kdata[:, self.necho_mGRE:self.necho_mGRE+1, ...] = kdata[:, self.necho_mGRE:self.necho_mGRE+1, ...] * self.normalizations[1]
        kdata[:, self.necho_mGRE+1:self.necho_mGRE+2, ...] = kdata[:, self.necho_mGRE+1:self.necho_mGRE+2, ...] * self.normalizations[2]

        org[:self.necho_mGRE*2, ...] = org[:self.necho_mGRE*2, ...] * self.normalizations[0]
        org[self.necho_mGRE*2:(self.necho_mGRE+1)*2, ...] = org[self.necho_mGRE*2:(self.necho_mGRE+1)*2, ...] * self.normalizations[1]
        org[(self.necho_mGRE+1)*2:(self.necho_mGRE+2)*2, ...] = org[(self.necho_mGRE+1)*2:(self.necho_mGRE+2)*2, ...] * self.normalizations[2]
        
        return kdata, org, csm, brain_mask





        

        


    
