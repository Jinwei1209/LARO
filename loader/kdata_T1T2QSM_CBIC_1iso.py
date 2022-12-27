import os
import numpy as np
from torch.utils import data
from utils.data import *
from utils.loss import *
from utils.operators import *


class kdata_T1T2QSM_CBIC_1iso(data.Dataset):
    '''
        Dataloader of 1.0*1.0*1.0 T1w+mGRE+T2w data from GE scanner (CBIC scanner)
    '''

    def __init__(self,
        rootDir = '/data4/Jinwei/T1T2QSM',
        contrast = 'MultiContrast',
        necho = 7, # number of echos in total
        necho_mGRE = 5,  # number of echos of mGRE data
        necho_t1w = 1,  # number of T1w images
        necho_t2w = 1,  # number of T2w images
        nrow = 206,
        ncol = 80,
        split = 'train',
        dataset_id = 0,  # 0: new2 of T1w+mGRE dataset
        subject = 0,  # 0: junghun, 1: chao, 2: alexey
        prosp_flag = 0,
        padding_flag = 0,
        normalizations = [50, 100, 125],  # normalization factor for mGRE, T1w and T2w data
        echo_cat = 1, # flag to concatenate echo dimension into channel
        batchSize = 1,
        augmentations = [None]
    ):

        self.rootDir = rootDir
        self.rootDir2 = '/data3/Jinwei/T1T2QSM'
        self.contrast = contrast
        self.necho = necho
        self.necho_t1w = necho_t1w
        self.necho_t2w = necho_t2w
        self.necho_mGRE = necho_mGRE
        self.normalizations = normalizations
        # self.scales = [3, 1.5, 3, 0.75]  # order: [chao8, hangwei8, dom8, jiahao8]
        self.echo_cat = echo_cat
        self.split = split
        self.dataset_id = dataset_id
        # if self.dataset_id == 0:
        if padding_flag == 1:
            self.id = 'new4'
        else:
            self.id = 'new4_no_padding'
        self.echo_stride = 1
        self.necho = 11+necho_t1w-1
        self.necho_mGRE = 9
        self.prosp_flag = prosp_flag
        self.padding_flag = padding_flag
        if self.padding_flag:
            self.nslice = 436
        else:
            self.nslice = 256
        self.nrow = nrow
        self.ncol = ncol
        if contrast == 'MultiContrast':
            if split == 'train':
                self.nsamples = self.nslice * 8
            elif split == 'val':
                self.nsamples = self.nslice
            elif split == 'test':
                self.nsamples = self.nslice
                if subject == 0:
                    self.subject = 'liangdong13'
                elif subject == 1:
                    self.subject = 'chao13'
                elif subject == 2:
                    self.subject = 'hangwei13'
                elif subject == 3:
                    self.subject = 'jiahao13'
                elif subject == 4:
                    self.subject = 'dom13'
                elif subject == 5:
                    self.subject = 'alexey13'
                elif subject == 6:
                    self.subject = 'qihao13'
                elif subject == 7:
                    self.subject = 'carly13'
                elif subject == 8:
                    self.subject = 'mert13'
                elif subject == 9:
                    self.subject = 'kelly13'
                elif subject == 10:
                    self.subject = 'thanh13'
                elif subject == 11:
                    self.subject = 'daniel13'
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
                if (idx - self.nslice < 0):
                    break
                else:
                    idx -= self.nslice
                    subject += 1
            if subject == 0:
                dataFD_sense_echo = self.rootDir + '/data_cfl/{}/chao13/full_cc_slices_sense_echo'.format(self.id)
            elif subject == 1:
                dataFD_sense_echo = self.rootDir + '/data_cfl/{}/hangwei13/full_cc_slices_sense_echo'.format(self.id)
            elif subject == 2:
                dataFD_sense_echo = self.rootDir + '/data_cfl/{}/jiahao13/full_cc_slices_sense_echo'.format(self.id)
            elif subject == 3:
                dataFD_sense_echo = self.rootDir + '/data_cfl/{}/dom13/full_cc_slices_sense_echo'.format(self.id)
            elif subject == 4:
                dataFD_sense_echo = self.rootDir + '/data_cfl/{}/qihao13/full_cc_slices_sense_echo'.format(self.id)
            elif subject == 5:
                dataFD_sense_echo = self.rootDir + '/data_cfl/{}/liangdong13/full_cc_slices_sense_echo'.format(self.id)
            elif subject == 6:
                dataFD_sense_echo = self.rootDir + '/data_cfl/{}/kelly13/full_cc_slices_sense_echo'.format(self.id)
            elif subject == 7:
                dataFD_sense_echo = self.rootDir + '/data_cfl/{}/mert13/full_cc_slices_sense_echo'.format(self.id)
            dataFD_sense_echo_mask = dataFD_sense_echo
            scale = 1
        elif self.split == 'val':
            dataFD_sense_echo = self.rootDir + '/data_cfl/{}/qihao8/full_cc_slices_sense_echo'.format(self.id)
            dataFD_sense_echo_mask = dataFD_sense_echo
            scale = 1
        elif self.split == 'test':
            if not self.prosp_flag:
                dataFD_sense_echo = self.rootDir + '/data_cfl/{}/'.format(self.id) + self.subject + '/full_cc_slices_sense_echo'
                dataFD_sense_echo_mask = dataFD_sense_echo
            else:
                dataFD_sense_echo = self.rootDir + '/data_cfl/{}/'.format(self.id) + self.subject + '/under_cc_slices_sense_echo'
                dataFD_sense_echo_mask = self.rootDir2 + '/data_cfl/{}/'.format(self.id) + 'jiahao13' + '/full_cc_slices_sense_echo'
            scale = 1

        if (self.batchIndex == self.batchSize):
            self.batchIndex = 0
            self.augIndex += 1
            self.augIndex = self.augIndex % self.augSize
            self.augmentation = self.augmentations[self.augIndex]
        self.batchIndex += 1

        if self.padding_flag:
            org1 = readcfl(dataFD_sense_echo + '_pad_1-4/fully_slice_{}'.format(idx))  # (row, col, echo)
            brain_mask_large = abs(org1[..., 0]) > 0
            org2 = readcfl(dataFD_sense_echo + '_pad_5-8/fully_slice_{}'.format(idx))  # (row, col, echo)
            org3 = readcfl(dataFD_sense_echo + '_pad_9-11/fully_slice_{}'.format(idx))  # (row, col, echo)
            org = np.concatenate((org1, org2, org3), axis=-1)  # concatenate data split into two folders
        else:
            org = readcfl(dataFD_sense_echo + '/fully_slice_{}'.format(idx))  # (row, col, echo)
        org =  c2r(org, self.echo_cat)  # echo_cat == 1: (2*echo, row, col) with first dimension real&imag concatenated for all echos 
                                        # echo_cat == 0: (2, row, col, echo)

        if self.echo_cat == 0:
            org = np.transpose(org, (0, 3, 1, 2)) # (2, echo, row, col)

        # Option 2: csms estimated from each echo
        if self.padding_flag:
            csm1 = readcfl(dataFD_sense_echo + '_pad_1-4/sensMaps_slice_{}'.format(idx))
            csm2 = readcfl(dataFD_sense_echo + '_pad_5-8/sensMaps_slice_{}'.format(idx))
            csm3 = readcfl(dataFD_sense_echo + '_pad_9-11/sensMaps_slice_{}'.format(idx))
            csm = np.concatenate((csm1, csm2, csm3), axis=-1)  # concatenate data split into two folders
        else:
            csm = readcfl(dataFD_sense_echo + '/sensMaps_slice_{}'.format(idx))
        csm = np.transpose(csm, (2, 3, 0, 1))  # (coil, echo, row, col)
        for i in range(self.necho):
            csm[:, i, :, :] = csm[:, i, :, :] * np.exp(-1j * np.angle(csm[0:1, i, :, :]))
    
        csm = c2r_kdata(csm) # (coil, echo, row, col, 2) with last dimension real&imag

        # Fully sampled kspace data
        if self.padding_flag:
            kdata1 = readcfl(dataFD_sense_echo + '_pad_1-4/kdata_slice_{}'.format(idx))
            kdata2 = readcfl(dataFD_sense_echo + '_pad_5-8/kdata_slice_{}'.format(idx))
            kdata3 = readcfl(dataFD_sense_echo + '_pad_9-11/kdata_slice_{}'.format(idx))
            kdata = np.concatenate((kdata1, kdata2, kdata3), axis=-1)  # concatenate data split into two folders
        else:
            kdata = readcfl(dataFD_sense_echo + '/kdata_slice_{}'.format(idx))
        kdata = np.transpose(kdata, (2, 3, 0, 1))  # (coil, echo, row, col)
        kdata = c2r_kdata(kdata) # (coil, echo, row, col, 2) with last dimension real&imag

        # brain tissue mask
        if self.padding_flag:
            brain_mask = np.real(readcfl(dataFD_sense_echo_mask + '_pad_1-4/mask_slice_{}'.format(idx)))  # (row, col)
            brain_mask = brain_mask_large
        else:
            brain_mask = np.real(readcfl(dataFD_sense_echo_mask + '/mask_slice_{}'.format(idx)))  # (row, col)
        if self.echo_cat:
            brain_mask = np.repeat(brain_mask[np.newaxis, ...], self.necho*2, axis=0) # (2*echo, row, col)
        else:
            brain_mask = np.repeat(brain_mask[np.newaxis, ...], 2, axis=0) # (2, row, col)
            brain_mask = np.repeat(brain_mask[:, np.newaxis, ...], self.necho, axis=1)# (2, echo, row, col)

        # normalize
        kdata[:, :self.necho_mGRE, ...] = kdata[:, :self.necho_mGRE, ...] * self.normalizations[0] * scale
        kdata[:, self.necho_mGRE:self.necho_mGRE+self.necho_t1w, ...] = kdata[:, self.necho_mGRE:self.necho_mGRE+self.necho_t1w, ...] * self.normalizations[1] * scale
        kdata[:, self.necho_mGRE+self.necho_t1w:self.necho_mGRE+self.necho_t1w+self.necho_t2w, ...] = kdata[:, self.necho_mGRE+self.necho_t1w:self.necho_mGRE+self.necho_t1w+self.necho_t2w, ...] * self.normalizations[2] * scale

        org[:self.necho_mGRE*2, ...] = org[:self.necho_mGRE*2, ...] * self.normalizations[0] * scale
        org[self.necho_mGRE*2:(self.necho_mGRE+self.necho_t1w)*2, ...] = org[self.necho_mGRE*2:(self.necho_mGRE+self.necho_t1w)*2, ...] * self.normalizations[1] * scale
        org[(self.necho_mGRE+self.necho_t1w)*2:(self.necho_mGRE+self.necho_t1w+self.necho_t2w)*2, ...] = org[(self.necho_mGRE+self.necho_t1w)*2:(self.necho_mGRE+self.necho_t1w+self.necho_t2w)*2, ...] * self.normalizations[2] * scale

        return kdata, org, csm, brain_mask





        

        


    
