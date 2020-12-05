import os
import numpy as np

from numpy import pi
from torch.utils import data
from utils.data import *
from utils.loss import *
from utils.operators import *



class MultiEchoTemp(data.Dataset):
    '''
        Dataloader of temporal undersampled multi-echo GRE data from deep learning predicted iField data
    '''
    def __init__(self,
        rootDir = '/data/Jinwei/Multi_echo_kspace/data_parameters',
        subject_IDs = ['sub1'],
        num_echos = 6,
        ratio = 0.2,
        plane = 'coronal'  # 'axial', 'coronal' or 'sagittal'
    ):
        self.rootDir = rootDir
        self.subject_IDs = subject_IDs
        self.num_echos = num_echos
        self.ratio = ratio
        self.plane = plane
        self.num_coils = 1
        self.load_subjects()

    def load_subjects(self):
        M_0, R_2, phi_0, phi_1, iField, mask = [], [], [], [], [], []
        M_0_under, R_2_under, phi_0_under, phi_1_under = [], [], [], []
        self.parameters_means, self.parameters_stds = [], []
        self.num_subjects = len(self.subject_IDs)
        self.num_slices = np.zeros(self.num_subjects)
        for idx, subject_ID in enumerate(self.subject_IDs):
            print('Loading case: {0}'.format(idx))
            dataFD = self.rootDir + '/' + subject_ID
            dataFD_under = dataFD + '/' + str(self.ratio)
            dataFD_iField = dataFD + '/iField_recon'
            # load labels
            if subject_ID == 'sub1':
                M_0.append(np.real(load_mat(dataFD+'/m0.mat', 'm0'))[np.newaxis, 15:215, ...])
                R_2.append(load_mat(dataFD+'/R2s.mat', 'R2s')[np.newaxis, 15:215, ...])
                phi_0.append(load_mat(dataFD+'/f0.mat', 'f0')[np.newaxis, 15:215, ...])
                phi_1.append(load_mat(dataFD+'/p.mat', 'p')[np.newaxis, 15:215, ...])
                mask.append(load_mat(dataFD+'/Mask.mat', 'Mask')[np.newaxis, 15:215, ...])
            else:
                M_0.append(np.real(load_mat(dataFD+'/m0.mat', 'm0'))[np.newaxis, 30:230, ...])
                R_2.append(load_mat(dataFD+'/R2s.mat', 'R2s')[np.newaxis, 30:230, ...])
                phi_0.append(load_mat(dataFD+'/f0.mat', 'f0')[np.newaxis, 30:230, ...])
                phi_1.append(load_mat(dataFD+'/p.mat', 'p')[np.newaxis, 30:230, ...])
                mask.append(load_mat(dataFD+'/Mask.mat', 'Mask')[np.newaxis, 30:230, ...])

            # zscore of the labels
            M_0[idx] = (M_0[idx] - np.mean(M_0[idx][mask[idx]==1])) / np.std(M_0[idx][mask[idx]==1])
            R_2[idx] = (R_2[idx] - np.mean(R_2[idx][mask[idx]==1])) / np.std(R_2[idx][mask[idx]==1])
            phi_0[idx] = (phi_0[idx] - np.mean(phi_0[idx][mask[idx]==1])) / np.std(phi_0[idx][mask[idx]==1])
            phi_1[idx] = (phi_1[idx] - np.mean(phi_1[idx][mask[idx]==1])) / np.std(phi_1[idx][mask[idx]==1])

            # load input
            M_0_under.append(np.real(load_mat(dataFD_under+'/m0_{0}.mat'.format(self.num_echos), 'm0'))[np.newaxis, ...])
            R_2_under.append(load_mat(dataFD_under+'/R2s_{0}.mat'.format(self.num_echos), 'R2s')[np.newaxis, ...])
            phi_0_under.append(load_mat(dataFD_under+'/f0_{0}.mat'.format(self.num_echos), 'f0')[np.newaxis, ...])
            phi_1_under.append(load_mat(dataFD_under+'/p_{0}.mat'.format(self.num_echos), 'p')[np.newaxis, ...])

            # calculate means and stds of each subject input
            mean_M_0_under = np.mean(M_0_under[idx][mask[idx]==1])
            mean_R_2_under = np.mean(R_2_under[idx][mask[idx]==1])
            mean_phi_0_under = np.mean(phi_0_under[idx][mask[idx]==1])
            mean_phi_1_under = np.mean(phi_1_under[idx][mask[idx]==1])
            self.parameters_means.append(np.array([mean_M_0_under, mean_R_2_under, mean_phi_0_under, mean_phi_1_under]))

            std_M_0_under = np.std(M_0_under[idx][mask[idx]==1])
            std_R_2_under = np.std(R_2_under[idx][mask[idx]==1])
            std_phi_0_under = np.std(phi_0_under[idx][mask[idx]==1])
            std_phi_1_under = np.std(phi_1_under[idx][mask[idx]==1])
            self.parameters_stds.append(np.array([std_M_0_under, std_R_2_under, std_phi_0_under, std_phi_1_under]))

            # load reconstructed under-sampled iField 
            Recon = load_mat(dataFD_iField+'/iField_{}.mat'.format(self.ratio), 'Recons')
            Recon[::2, ...] = - Recon[::2, ...]
            iField.append(Recon[np.newaxis, ...])

            if self.plane == 'axial':
                self.num_slices[idx] = Recon.shape[2]
            elif self.plane == 'sagittal':
                self.num_slices[idx] = Recon.shape[1]
            elif self.plane == 'coronal':
                self.num_slices[idx] = Recon.shape[0]
            print('Num_slice on {}: {}'.format(subject_ID, self.num_slices[idx]))

        self.M_0 = np.concatenate(M_0, axis=0)
        self.M_0_under = np.concatenate(M_0_under, axis=0)
        self.R_2 = np.concatenate(R_2, axis=0)
        self.R_2_under = np.concatenate(R_2_under, axis=0)
        self.phi_0 = np.concatenate(phi_0, axis=0)
        self.phi_0_under = np.concatenate(phi_0_under, axis=0)
        self.phi_1 = np.concatenate(phi_1, axis=0)
        self.phi_1_under = np.concatenate(phi_1_under, axis=0)
        self.iField = np.concatenate(iField, axis=0)
        self.mask = np.concatenate(mask, axis=0)
        self.num_samples = np.sum(self.num_slices)

    def __len__(self):
        return int(self.num_samples)

    def __getitem__(self, idx):
        idx_slice = idx
        idx_subject = 0
        while (True):
            if (idx_slice - self.num_slices[idx_subject] < 0):
                break
            else:
                idx_slice -= int(self.num_slices[idx_subject])
                idx_subject += 1
        if self.plane == 'axial':
            M_0_slice = self.M_0[idx_subject, :, :, idx_slice][np.newaxis, ...]
            M_0_under_slice = self.M_0_under[idx_subject, :, :, idx_slice][np.newaxis, ...]

            R_2_slice = self.R_2[idx_subject, :, :, idx_slice][np.newaxis, ...]
            R_2_under_slice = self.R_2_under[idx_subject, :, :, idx_slice][np.newaxis, ...]

            phi_0_slice = self.phi_0[idx_subject, :, :, idx_slice][np.newaxis, ...]
            phi_0_under_slice = self.phi_0_under[idx_subject, :, :, idx_slice][np.newaxis, ...]

            phi_1_slice = self.phi_1[idx_subject, :, :, idx_slice][np.newaxis, ...]
            phi_1_under_slice = self.phi_1_under[idx_subject, :, :, idx_slice][np.newaxis, ...]

            mask_slice = self.mask[idx_subject, :, :, idx_slice][np.newaxis, ...]
            # assuming 1 coil and num_echos
            iField_slice = self.iField[idx_subject, :, :, idx_slice, np.newaxis, 0:self.num_echos]
        
        if self.plane == 'coronal':
            M_0_slice = self.M_0[idx_subject, idx_slice, :, :][np.newaxis, ...]
            M_0_under_slice = self.M_0_under[idx_subject, idx_slice, :, :][np.newaxis, ...]

            R_2_slice = self.R_2[idx_subject, idx_slice, :, :][np.newaxis, ...]
            R_2_under_slice = self.R_2_under[idx_subject, idx_slice, :, :][np.newaxis, ...]

            phi_0_slice = self.phi_0[idx_subject, idx_slice, :, :][np.newaxis, ...]
            phi_0_under_slice = self.phi_0_under[idx_subject, idx_slice, :, :][np.newaxis, ...]

            phi_1_slice = self.phi_1[idx_subject, idx_slice, :, :][np.newaxis, ...]
            phi_1_under_slice = self.phi_1_under[idx_subject, idx_slice, :, :][np.newaxis, ...]

            mask_slice = self.mask[idx_subject, idx_slice, :, :][np.newaxis, ...]
            # assuming 1 coil and num_echos
            iField_slice = self.iField[idx_subject, idx_slice, :, :, np.newaxis, 0:self.num_echos]

        targets = np.concatenate((M_0_slice, R_2_slice, phi_0_slice, phi_1_slice), axis=0)
        mask_slice = np.tile(mask_slice, [4, 1, 1])  # brain mask

        iField_slice_2 = np.zeros(iField_slice.shape+(2,))
        iField_slice_2[..., 0] = np.real(iField_slice)
        iField_slice_2[..., 1] = np.imag(iField_slice)  
        inputs = np.concatenate((M_0_under_slice, R_2_under_slice, 
                                    phi_0_under_slice, phi_1_under_slice), axis=0)
        targets = targets.astype(np.float32)
        mask_slice = mask_slice.astype(np.float32)
        iField_slice_2 = iField_slice_2.astype(np.float32)
        return targets, mask_slice, iField_slice_2, inputs, self.parameters_means[idx_subject], self.parameters_stds[idx_subject]

        

        


    
