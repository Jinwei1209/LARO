import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import argparse

from torch.utils import data
from loader.kdata_loader_GE import kdata_loader_GE
from scipy.stats import norm
from scipy.optimize import minimize
from bayesOpt.sample_loss import *
from bayesOpt.bayes_opt_policies import *
from bayesOpt.cross_validation import *

if __name__ == '__main__':
    # typein parameters
    parser = argparse.ArgumentParser(description='BO-LOUPE')
    parser.add_argument('--gpu_id', type=str, default='0, 1')
    parser.add_argument('--flag_policy', type=int, default=0)  # 0 for EI, 1 for qEI, 2 for qKG
    parser.add_argument('--cv', type=int, default=0)    # 0 for not doing cross-validation, 1 for doing cross-validation
    opt = {**vars(parser.parse_args())}
    # fixed parameters
    q = 1  # number of step lookahead
    contrast = 'T1'
    sampling_ratio = 0.1
    n_pre_samples = 8
    n_iters = 30

    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_id']

    print('Contrast is {0}, sampling ratio is {1}, use {2} GPU(s)!'.format(
        contrast, sampling_ratio, torch.cuda.device_count()))

    rootName = '/data/Jinwei/{0}_slice_recon_GE'.format(contrast)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loader = kdata_loader_GE(
            rootDir=rootName,
            contrast=contrast, 
            split='test',
            SNR=10,
            flag_BO=1,
            slice_spacing=25
        )
    data_loader = data.DataLoader(data_loader, batch_size=8, shuffle=False)

    bounds = np.array([[-1., -1.], [1., 1.]])  # for log uniform space
    # bounds = np.array([[0., 0.], [10., 10.]])  # for uniform space
    objective = lambda z: recon_loss(z, data_loader, device, sampling_ratio)

    # Read in data from a file.
    filename = 'presample_data.csv'
    # If data doesn't exist, generate it
    if not os.path.exists(filename):
        print('Randomly generate some samples')
        np.random.seed(1)
        x_list, y_list = [], []
        # random initialization
        for params in np.random.uniform(bounds[0], bounds[1], (n_pre_samples, bounds.shape[0])):
                x_list.append(params)
                y_list.append(objective(params))
        x = np.array(x_list)
        y = np.array(y_list)
        data = np.concatenate((x,y.reshape(len(y),1)), axis = 1)
        np.savetxt(filename,data)

    # Read in data from a file.  
    data = np.loadtxt(filename)
    x = data[:,0:2] # First two column of the data
    y = data[:,-1] # Last column of the data

    
    print('Value of best point found: {}'.format(y.max()))
    a_best, b_best = x[np.argmax(y)]
    best = [y.max()] # This will store the best value

    if opt['cv'] == 1:
        cross_validation(train_x = x, train_y = y)

    # if opt['flag_policy'] == 0:
    #     value_fig_name = 'Values_EI.png'
    #     mask_fig_name = 'mask_best_EI.png'
    #     policy_update(x, y, bounds, objective, n_iters, best, a_best, b_best,
    #                     EI_policy, q, value_fig_name, mask_fig_name, True)
    # elif opt['flag_policy'] == 1:
    #     value_fig_name = 'Values_qEI.png'
    #     mask_fig_name = 'mask_best_qEI.png'
    #     policy_update(x, y, bounds, objective, n_iters, best, a_best, b_best,
    #                     qEI_policy, q, value_fig_name, mask_fig_name, True)

    # else:
    #     value_fig_name = 'Values_qKG.png'
    #     mask_fig_name = 'mask_best_qKG.png'
    #     policy_update(x, y, bounds, objective, n_iters, best, a_best, b_best,
    #                     KG_policy, q, value_fig_name, mask_fig_name, True)

    value_fig_name = 'Values_EI.png'
    mask_fig_name = 'mask_best_EI.png'
    policy_update(x, y, bounds, objective, n_iters, best, a_best, b_best,
                    EI_policy, q, value_fig_name, mask_fig_name, True)
    value_fig_name = 'Values_qEI.png'
    mask_fig_name = 'mask_best_qEI.png'
    policy_update(x, y, bounds, objective, n_iters, best, a_best, b_best,
                    qEI_policy, q, value_fig_name, mask_fig_name, True)
    value_fig_name = 'Values_qKG.png'
    mask_fig_name = 'mask_best_qKG.png'
    policy_update(x, y, bounds, objective, n_iters, best, a_best, b_best,
                    KG_policy, q, value_fig_name, mask_fig_name, True)







    
    