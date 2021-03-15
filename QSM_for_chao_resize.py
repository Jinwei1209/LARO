import os
import time
import math
import argparse
import scipy.io as sio
import numpy as np

nslice = 256
nrol = 206
ncol = 80
# nechos = ['04', '06', '08', '10']
nechos = ['05', '07', '09']
for sub_id in range(1, 31):
    # for idx, echo in enumerate([4, 6, 8, 10]):
    for idx, echo in enumerate([5, 7, 9]):
        necho = nechos[idx]
        iField = np.zeros((80, 206, 256, echo, 2))
        rootName = '/data/Jinwei/QSM_raw_CBIC'
        try:
            file = sio.loadmat(rootName+'/data_chao_resize/case{}.mat'.format(sub_id))
        except:
            continue
        Recons = file['iField'][..., :echo]
        Recons = np.transpose(Recons, [2, 1, 0, 3])
        
        iField[..., 0] = np.real(Recons)
        iField[..., 1] = np.imag(Recons)
        # iField[:, :, 1::2, :, :] = - iField[:, :, 1::2, :, :]
        iField[..., 1] = - iField[..., 1]
        iField = iField.astype(np.float32)
        print('iField size is: ', iField.shape)

        if os.path.exists(rootName+'/result_QSM_chao_resize/iField.bin'):
            os.remove(rootName+'/result_QSM_chao_resize/iField.bin')
        iField.tofile(rootName+'/result_QSM_chao_resize/iField.bin')
        print('Successfully save iField.bin')

        # run MEDIN
        os.system('medi ' + rootName + '/result_QSM_chao_resize/iField.bin' 
                + ' --parameter ' + rootName + '/result_QSM_chao_resize/parameter_{}.txt'.format(echo)
                + ' --temp ' + rootName +  '/result_QSM_chao_resize/'
                + ' --GPU ' + ' --device ' + '2'
                + ' -of QR') 
                # + ' --CSF ' + ' -of QR')


        # read .bin files and save into .mat files
        QSM = np.fromfile(rootName+'/result_QSM_chao_resize/recon_QSM_{}.bin'.format(necho), 'f4')
        QSM = np.transpose(QSM.reshape([80, 206, nslice]), [2, 1, 0])

        iMag = np.fromfile(rootName+'/result_QSM_chao_resize/iMag.bin', 'f4')
        iMag = np.transpose(iMag.reshape([80, 206, nslice]), [2, 1, 0])

        RDF = np.fromfile(rootName+'/result_QSM_chao_resize/RDF.bin', 'f4')
        RDF = np.transpose(RDF.reshape([80, 206, nslice]), [2, 1, 0])

        R2star = np.fromfile(rootName+'/result_QSM_chao_resize/R2star.bin', 'f4')
        R2star = np.transpose(R2star.reshape([80, 206, nslice]), [2, 1, 0])

        N_std = np.fromfile(rootName+'/result_QSM_chao_resize/N_std.bin', 'f4')
        N_std = np.transpose(N_std.reshape([80, 206, nslice]), [2, 1, 0])

        Mask = np.fromfile(rootName+'/result_QSM_chao_resize/Mask.bin', 'f4')
        Mask = np.transpose(Mask.reshape([80, 206, nslice]), [2, 1, 0]) > 0

        adict = {}
        adict['QSM'], adict['iMag'], adict['RDF'] = QSM, iMag, RDF
        adict['R2star'], adict['N_std'], adict['Mask'] = R2star, N_std, Mask
        sio.savemat(rootName+'/data_chao_echo_no_csf/QSM{}_{}.mat' \
            .format(sub_id, echo), adict)
        print('successfully save data')