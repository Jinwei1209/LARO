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
        num_echos = 6
    ):

        self.rootDir = rootDir
        self.subject_IDs = subject_IDs
        self.num_echos = num_echos
        self.num_coils = 1
        self.load_subjects()

    def load_subjects(self):
        M_0, R_2, phi_0, phi_1, iField, mask = [], [], [], [], [], []
        self.num_subjects = len(self.subject_IDs)
        self.num_slices = np.zeros(self.num_subjects)
        for idx, subject_ID in enumerate(self.subject_IDs):
            print('Loading case: {0}'.format(idx))
            dataFD = self.rootDir + '/' + subject_ID
            M_0.append(np.real(load_mat(dataFD+'/water.mat', 'water'))[np.newaxis, ...])
            r_2 = load_mat(dataFD+'/R2s.mat', 'R2s')[np.newaxis, ...]
            r_2[r_2 > 1e+1] = 0
            r_2[r_2 < -1e+1] = 0
            R_2.append(r_2)
            # R_2.append(load_mat(dataFD+'/R2s.mat', 'R2s')[np.newaxis, ...])
            phi_0.append(load_mat(dataFD+'/p0.mat', 'p0')[np.newaxis, ...])
            phi_1.append(load_mat(dataFD+'/p1.mat', 'p1')[np.newaxis, ...])
            iField.append(load_mat(dataFD+'/iField.mat', 'iField')[np.newaxis, ...])
            mask.append(load_mat(dataFD+'/Mask.mat', 'Mask')[np.newaxis, ...])

            self.num_slices[idx] = load_mat(dataFD+'/Mask.mat', 'Mask').shape[2]

        self.M_0 = np.concatenate(M_0, axis=0)
        self.R_2 = np.concatenate(R_2, axis=0)
        self.phi_0 = np.concatenate(phi_0, axis=0)
        self.phi_1 = np.concatenate(phi_1, axis=0)
        self.iField = np.concatenate(iField, axis=0)
        # conj of iField because in fit_ppm_complex code there was M = conj(M)
        self.iField = np.conjugate(self.iField)
        self.mask = np.concatenate(mask, axis=0)
        self.num_samples = np.sum(self.num_slices)
        # sampling pattern
        self.sampling_mask = load_mat(self.rootDir+'/sampling_pattern_30.mat', 'mask')
        self.sampling_mask = np.ones(self.sampling_mask.shape)
        self.sampling_mask = np.fft.fftshift(self.sampling_mask)[..., np.newaxis, np.newaxis]
        self.sampling_mask = np.tile(self.sampling_mask, [1, 1, 1, self.num_echos])

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
        R_2_slice = self.R_2[idx_subject, :, :, idx_slice][np.newaxis, ...]
        phi_0_slice = self.phi_0[idx_subject, :, :, idx_slice][np.newaxis, ...]
        phi_1_slice = self.phi_1[idx_subject, :, :, idx_slice][np.newaxis, ...]
        mask_slice = self.mask[idx_subject, :, :, idx_slice][np.newaxis, ...]

        # assuming 1 coil and num_echos
        iField_slice = self.iField[idx_subject, :, :, idx_slice, np.newaxis, 0:self.num_echos]

        # sensitivity maps are all ones, with dims = (coil, echo, row, column, 2)
        csm_slice_mat = np.ones(iField_slice.shape[0:2] + (self.num_coils, self.num_echos, ))
        csm_slice = np.ones((self.num_coils, self.num_echos, ) + iField_slice.shape[0:2])
        csm_slice = np.float32(csm_slice)
        csm_slice = np.float32(np.concatenate((csm_slice[..., np.newaxis], 
                               np.zeros(csm_slice.shape+(1,))), axis=-1))

        # coil-combined image, magnitude and temperal unwrapped phase
        kdata_slice, image_slice = self.generate_kdata(iField_slice, csm_slice_mat)
        mag_slice = abs(image_slice)
        unwrapped_phase = self.phase_unwrap(image_slice)
        mag_slice = np.float32(np.transpose(mag_slice, (2, 0, 1)))
        unwrapped_phase = np.float32(np.transpose(unwrapped_phase, (2, 0, 1)))

        # sampling pattern, with dims = (coil, echo, row, column, 2)
        sampling_mask_slice = np.transpose(self.sampling_mask, (2, 3, 0, 1))
        sampling_mask_slice = np.float32(sampling_mask_slice)
        sampling_mask_slice = np.float32(np.concatenate((sampling_mask_slice[..., np.newaxis], 
                                         np.zeros(sampling_mask_slice.shape+(1,))), axis=-1))

        target = np.concatenate((M_0_slice, R_2_slice, phi_0_slice, phi_1_slice), axis=0)
        mask_slice = np.tile(mask_slice, [4, 1, 1])

        return target, mask_slice, sampling_mask_slice, csm_slice, kdata_slice, mag_slice, unwrapped_phase

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

        

        


    
