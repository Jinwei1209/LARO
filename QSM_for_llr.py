import os
import time
import math
import argparse
import scipy.io as sio
import numpy as np

nslice = 200
nrol = 206
ncol = 80
necho = 10
for loupe in [-2]:  # [-1, 0, -2]
    for sub_id in [0]:  # [0, 1, 2]
        
        iField = np.zeros((80, 206, 200, 10, 2))
        rootName = '/data/Jinwei/QSM_raw_CBIC'
        # load kspace slice and sensMap
        subs = ['junghun', 'chao', 'alexey']
        sub = subs[sub_id]
        file = sio.loadmat('/data/Jinwei/QSM_raw_CBIC/data_cfl/{}/iField_llr_loupe={}_sub={}.mat'.format(sub, loupe, sub_id))
        Recons = file['iField_llr']
        Recons = np.transpose(Recons, [2, 1, 0, 3])
        
        iField[..., 0] = np.real(Recons)
        iField[..., 1] = np.imag(Recons)
        iField[:, :, 1::2, :, :] = - iField[:, :, 1::2, :, :]
        iField[..., 1] = - iField[..., 1]

        print('iField size is: ', iField.shape)
        if os.path.exists(rootName+'/results_QSM/iField.bin'):
            os.remove(rootName+'/results_QSM/iField.bin')
        iField.tofile(rootName+'/results_QSM/iField.bin')
        print('Successfully save iField.bin')

        # run MEDIN
        os.system('medi ' + rootName + '/results_QSM/iField.bin' 
                + ' --parameter ' + rootName + '/results_QSM/parameter.txt'
                + ' --temp ' + rootName +  '/results_QSM/'
                + ' --GPU ' + ' --device ' + '0' 
                + ' --CSF ' + ' -of QR')

        # read .bin files and save into .mat files
        QSM = np.fromfile(rootName+'/results_QSM/recon_QSM_10.bin', 'f4')
        QSM = np.transpose(QSM.reshape([80, 206, nslice]), [2, 1, 0])

        iMag = np.fromfile(rootName+'/results_QSM/iMag.bin', 'f4')
        iMag = np.transpose(iMag.reshape([80, 206, nslice]), [2, 1, 0])

        RDF = np.fromfile(rootName+'/results_QSM/RDF.bin', 'f4')
        RDF = np.transpose(RDF.reshape([80, 206, nslice]), [2, 1, 0])

        R2star = np.fromfile(rootName+'/results_QSM/R2star.bin', 'f4')
        R2star = np.transpose(R2star.reshape([80, 206, nslice]), [2, 1, 0])

        Mask = np.fromfile(rootName+'/results_QSM/Mask.bin', 'f4')
        Mask = np.transpose(Mask.reshape([80, 206, nslice]), [2, 1, 0]) > 0

        adict = {}
        adict['QSM'], adict['iMag'], adict['RDF'] = QSM, iMag, RDF
        adict['R2star'], adict['Mask'] = R2star, Mask
        sio.savemat(rootName+'/results_ablation/QSM_llr_loupe={}_sub={}.mat' \
            .format(loupe, sub_id), adict)
        print('successfully save data')