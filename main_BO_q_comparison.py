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
    q = np.arange(1,6,2)  # number of step lookahead
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
    objective = lambda z: recon_loss(z, data_loader, sampling_ratio)

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
    # if save_dir doesn't exist, generate it
    if not os.path.exists('bo_results'):
        os.makedirs('bo_results')

    # Read in data from a file.  
    data = np.loadtxt(filename)
    data[:,-1] = np.array([objective(data[i, 0:2]) for i in range(len(data))])

    if opt['cv'] == 1:
        cross_validation(train_x = data[:,0:2], train_y = data[:,-1])

    best_qEI = np.zeros([len(q), n_iters+1])
    best_qKG = np.zeros([len(q), n_iters+1])
    q_ind = 0
    for step in q:
        value_fig_name = ''
        mask_fig_name = ''
        best_qEI[q_ind, :], x_qEI, y_qEI = policy_update(data, bounds, objective, n_iters,
                        qEI_policy, step, value_fig_name, mask_fig_name, sampling_ratio)
        # params_best_qEI = x_qEI[np.argmax(y_qEI)]
        # recon_loss(params_best_qEI, data_loader, sampling_ratio, K=20, save_name='Recons_qEI')
        best_qKG[q_ind, :], x_qKG, y_qKG = policy_update(data, bounds, objective, n_iters,
                        KG_policy, step, value_fig_name, mask_fig_name, sampling_ratio)  
        # params_best_qKG = x_qKG[np.argmax(y_qKG)]
        # recon_loss(params_best_qKG, data_loader, sampling_ratio, K=20, save_name='Recons_qKG')
        q_ind += 1


    fig, ax = plt.subplots()
    for i in range(len(q)): 
        ax.plot(best_qEI[i, :],'k*--', label = '{}-batch EI'.format(q[i]))
    plt.xlabel('number of iteration')
    plt.ylabel('Best value found')
    plt.legend()
    plt.savefig('./bo_results/qEI_comparison_.png')
    plt.close

    fig, ax = plt.subplots()
    for i in range(len(q)): 
        ax.plot(best_qKG[i, :],'k*--', label = '{}-batch KG'.format(q[i]))
    plt.xlabel('number of iteration')
    plt.ylabel('Best value found')
    plt.legend()
    plt.savefig('./bo_results/qKG_comparison_.png')
    plt.close()








    
    