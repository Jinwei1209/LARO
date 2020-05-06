import os
import numpy as np

from numpy import pi
from torch.utils import data
from utils.data import *
from utils.loss import *
from utils.operators import *



class MultiEchoSimu(data.Dataset):

    def __init__(self,
        rootDir = '/data/Jinwei/Multi_echo_kspace/dataset',
        subject_IDs = ['MS1'],
        num_echos = 3,
        flag_input = 0,  # 0 for four parameter under input, 1 for iField input
        flag_train = 1
    ):
        self.rootDir = rootDir
        self.subject_IDs = subject_IDs
        self.num_echos = num_echos
        self.num_coils = 1
        self.flag_train = flag_train
        self.load_subjects()
        self.flag_input = flag_input
        self.num_echos = num_echos

    def load_subjects(self):
        M_0, R_2, phi_0, phi_1, iField, phase, mask = [], [], [], [], [], [], []
        M_0_under, R_2_under, phi_0_under, phi_1_under = [], [], [], []
        self.num_subjects = len(self.subject_IDs)
        self.num_slices = np.zeros(self.num_subjects)
        for idx, subject_ID in enumerate(self.subject_IDs):
            print('Loading case: {0}'.format(idx))
            dataFD = self.rootDir + '/' + subject_ID
            M_0.append(np.real(load_mat(dataFD+'/m0.mat', 'm0'))[np.newaxis, ..., 0:68])
            M_0_under.append(np.real(load_mat(dataFD+'/m0_{0}.mat'.format(self.num_echos), 'm0'))[np.newaxis, ..., 0:68])
            # r_2 = load_mat(dataFD+'/R2s.mat', 'R2s')[np.newaxis, ...]
            # r_2[r_2 > 1e+1] = 0
            # r_2[r_2 < -1e+1] = 0
            # R_2.append(r_2)
            R_2.append(load_mat(dataFD+'/R2s.mat', 'R2s')[np.newaxis, ..., 0:68])
            R_2_under.append(load_mat(dataFD+'/R2s_{0}.mat'.format(self.num_echos), 'R2s')[np.newaxis, ..., 0:68])
            phi_0.append(load_mat(dataFD+'/f0.mat', 'f0')[np.newaxis, ..., 0:68])
            phi_0_under.append(load_mat(dataFD+'/f0_{0}.mat'.format(self.num_echos), 'f0')[np.newaxis, ..., 0:68])
            phi_1.append(load_mat(dataFD+'/p.mat', 'p')[np.newaxis, ..., 0:68])
            phi_1_under.append(load_mat(dataFD+'/p_{0}.mat'.format(self.num_echos), 'p')[np.newaxis, ..., 0:68])
            # phi_1.append(load_mat(dataFD+'/p1_unwrap.mat', 'p1')[np.newaxis, ...])
            iField.append(load_mat(dataFD+'/iField.mat', 'iField')[np.newaxis, :, :, 0:68, :])
            # phase.append(load_mat(dataFD+'/phase_unwrapped.mat', 'phase_unwrapped')[np.newaxis, ...])
            # # crop the brain mask
            # Mask = load_mat(dataFD+'/Mask.mat', 'Mask')[..., 0:68]
            # Mask = SMV(Mask, [512, 512, 68], [0.4688, 0.4688, 2], 2) > 0.999
            # mask.append(Mask[np.newaxis, ...])
            mask.append(load_mat(dataFD+'/Mask.mat', 'Mask')[np.newaxis, ..., 0:68])
            self.num_slices[idx] = 68

        self.M_0 = np.concatenate(M_0, axis=0)
        self.M_0_under = np.concatenate(M_0_under, axis=0)
        self.R_2 = np.concatenate(R_2, axis=0)
        self.R_2_under = np.concatenate(R_2_under, axis=0)
        self.phi_0 = np.concatenate(phi_0, axis=0)
        self.phi_0_under = np.concatenate(phi_0_under, axis=0)
        self.phi_1 = np.concatenate(phi_1, axis=0)
        self.phi_1_under = np.concatenate(phi_1_under, axis=0)
        self.iField = np.concatenate(iField, axis=0)
        # self.phase = np.concatenate(phase, axis=0)
        # # conj of iField because in fit_ppm_complex code there was M = conj(M)
        # self.iField, self.phase = np.conjugate(self.iField), -self.phase
        self.mask = np.concatenate(mask, axis=0)
        self.num_samples = np.sum(self.num_slices)
        # sampling pattern
        self.sampling_mask = load_mat(self.rootDir+'/sampling_pattern_30.mat', 'mask')
        # self.sampling_mask = np.ones(self.sampling_mask.shape)
        self.sampling_mask = np.fft.fftshift(self.sampling_mask)[..., np.newaxis, np.newaxis]
        self.sampling_mask = np.tile(self.sampling_mask, [1, 1, 1, self.num_echos])
        
        if self.flag_train:
            # compute mean
            self.mean_M_0_under = np.mean(self.M_0_under[self.mask==1])
            self.mean_R_2_under = np.mean(self.R_2_under[self.mask==1])
            self.mean_phi_0_under = np.mean(self.phi_0_under[self.mask==1])
            self.mean_phi_1_under = np.mean(self.phi_1_under[self.mask==1])

            # compute std
            self.std_M_0_under = np.std(self.M_0_under[self.mask==1])
            self.std_R_2_under = np.std(self.R_2_under[self.mask==1])
            self.std_phi_0_under = np.std(self.phi_0_under[self.mask==1])
            self.std_phi_1_under = np.std(self.phi_1_under[self.mask==1])

            self.parameters_means = [self.mean_M_0_under, self.mean_R_2_under, 
                                     self.mean_phi_0_under, self.mean_phi_1_under]

            self.parameters_stds = [self.std_M_0_under, self.std_R_2_under, 
                                    self.std_phi_0_under, self.std_phi_1_under]

    def __len__(self):
        return int(self.num_samples)

    def __getitem__(self, idx):
        idx_slice = idx
        idx_subject = 0
        while (True):
            if (idx_slice - self.num_slices[idx_subject] < 0):
                break
            else:
                idx_slice -= self.num_slices[idx_subject]
                idx_subject += 1

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
        targets = np.concatenate((M_0_slice, R_2_slice, phi_0_slice, phi_1_slice), axis=0)
        mask_slice = np.tile(mask_slice, [4, 1, 1])  # brain mask
        if self.flag_input == 1:
            # sensitivity maps are all ones, with dims = (coil, echo, row, column, 2)
            csm_slice_mat = np.ones(iField_slice.shape[0:2] + (self.num_coils, self.num_echos, ))
            csm_slice = np.ones((self.num_coils, self.num_echos, ) + iField_slice.shape[0:2])
            csm_slice = np.float32(csm_slice)
            csm_slice = np.float32(np.concatenate((csm_slice[..., np.newaxis], 
                                np.zeros(csm_slice.shape+(1,))), axis=-1))

            # coil-combined image, magnitude and temperal unwrapped phase
            kdata_slice, image_slice = self.generate_kdata(iField_slice, csm_slice_mat)
            mag_slice = abs(image_slice)
            unwrapped_phase = self.phase_unwrap(image_slice)  # temperal unwrapping, not good
            # unwrapped_phase = self.phase[idx_subject, :, :, idx_slice, 0:self.num_echos]  # spatial unwrapped (pre-processed, not good in the later echos)
            mag_slice = np.float32(np.transpose(mag_slice, (2, 0, 1)))
            unwrapped_phase = np.float32(np.transpose(unwrapped_phase, (2, 0, 1)))

            # sampling pattern, with dims = (coil, echo, row, column, 2)
            sampling_mask_slice = np.transpose(self.sampling_mask, (2, 3, 0, 1))
            sampling_mask_slice = np.float32(sampling_mask_slice)
            sampling_mask_slice = np.float32(np.concatenate((sampling_mask_slice[..., np.newaxis], 
                                            np.zeros(sampling_mask_slice.shape+(1,))), axis=-1))
            return targets, mask_slice, sampling_mask_slice, csm_slice, kdata_slice, mag_slice, unwrapped_phase
        else:
            iField_slice_2 = np.zeros(iField_slice.shape+(2,))
            iField_slice_2[..., 0] = np.real(iField_slice)
            iField_slice_2[..., 1] = np.imag(iField_slice)  
            inputs = np.concatenate((M_0_under_slice, R_2_under_slice, 
                                     phi_0_under_slice, phi_1_under_slice), axis=0)
            return targets, mask_slice, iField_slice_2, inputs

    def generate_kdata(self, iField_slice, csm_slice, noise_std=0.00):
        kdata_full = np.fft.fft2(iField_slice, axes=(0, 1), norm=None)
        noise = np.random.randn(kdata_full.shape[0], kdata_full.shape[1], kdata_full.shape[2], kdata_full.shape[3]) \
               + 1j*np.random.randn(kdata_full.shape[0], kdata_full.shape[1], kdata_full.shape[2], kdata_full.shape[3])
        noise = noise * noise_std * np.sqrt(iField_slice.shape[0]*iField_slice.shape[1]/2)
        kdata_under = (kdata_full + noise) * self.sampling_mask
        image_under = np.fft.ifft2(kdata_under, axes=(0, 1), norm=None)
        image_under = np.sum(image_under * np.conj(csm_slice), axis=2, keepdims=False)
        kdata_under = np.transpose(kdata_under, (2, 3, 0, 1))
        kdata_under = c2r_kdata(kdata_under)
        return kdata_under, image_under

    def phase_unwrap(self, image_slice):
        Y = np.angle(image_slice)
        c = Y[..., 1] - Y[..., 0]
        ind = np.argmin([abs(c-2*pi), abs(c), abs(c+2*pi)], axis=0)
        c[ind==0] = c[ind==0] - 2*pi
        c[ind==2] = c[ind==2] + 2*pi

        for n in np.arange(Y.shape[-1]-1):
            cd = Y[..., n+1] - Y[..., n] - c
            Y[cd<-pi, n+1:] += 2*pi
            Y[cd>pi, n+1:] -= 2*pi
        return Y

        

        


    
