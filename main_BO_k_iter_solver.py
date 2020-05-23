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
    parser.add_argument('--q', type=int, default=1) 
    opt = {**vars(parser.parse_args())}
    # fixed parameters
    q = opt['q']  # number of step lookahead
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
    objective_k5 = lambda z: recon_loss(z, data_loader, sampling_ratio, K=5)
    objective_k10 = lambda z: recon_loss(z, data_loader, sampling_ratio, K=10)

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

    # # Read in data from file
    data_k5 = np.loadtxt(filename)
    data_k5[:,-1] = np.array([objective_k5(data_k5[i, 0:2]) for i in range(len(data_k5))])

    data_k10 = np.loadtxt(filename)
    data_k10[:,-1] = np.array([objective_k10(data_k10[i, 0:2]) for i in range(len(data_k10))])

    # if opt['cv'] == 1:
    #     cross_validation(train_x = data[:,0:2], train_y = data[:,-1])


    # EI
    if q == 1:
        value_fig_name = './bo_results/Values_EI.png'
        mask_k5_fig_name = './bo_results/mask_k5_best_EI.png'
        best_EI_k5, x_EI, y_EI = policy_update(data_k5, bounds, objective_k5, n_iters,
                        EI_policy, q, value_fig_name, mask_k5_fig_name, sampling_ratio, True)
        params_best_EI = x_EI[np.argmax(y_EI)]
        recon_loss(params_best_EI, data_loader, sampling_ratio, K=20, save_name='Recons_EI_k5')

        mask_k10_fig_name = './bo_results/mask_k10_best_EI.png'
        best_EI_k10, x_EI, y_EI = policy_update(data_k10, bounds, objective_k10, n_iters,
                        EI_policy, q, value_fig_name, mask_k10_fig_name, sampling_ratio, True)
        params_best_EI = x_EI[np.argmax(y_EI)]
        recon_loss(params_best_EI, data_loader, sampling_ratio, K=20, save_name='Recons_EI_k10')

    # qEI
    value_fig_name = './bo_results/Values_qEI.png'
    mask_k5_fig_name = './bo_results/mask_k5_best_qEI.png'
    best_qEI_k5, x_qEI, y_qEI = policy_update(data_k5, bounds, objective_k5, n_iters,
                    qEI_policy, q, value_fig_name, mask_k5_fig_name, sampling_ratio, True)
    params_best_qEI = x_qEI[np.argmax(y_qEI)]
    recon_loss(params_best_qEI, data_loader, sampling_ratio, K=20, save_name='Recons_qEI_k5')

    mask_k10_fig_name = './bo_results/mask_k10_best_qEI.png'
    best_qEI_k10, x_qEI, y_qEI = policy_update(data_k10, bounds, objective_k10, n_iters,
                    qEI_policy, q, value_fig_name, mask_k10_fig_name, sampling_ratio, True)
    params_best_qEI = x_qEI[np.argmax(y_qEI)]
    recon_loss(params_best_qEI, data_loader, sampling_ratio, K=20, save_name='Recons_qEI_k10')

    # qKG
    value_fig_name = './bo_results/Values_qKG.png'
    mask_k5_fig_name = './bo_results/mask_k5_best_qKG.png'
    best_qKG_k5, x_qKG, y_qKG = policy_update(data_k5, bounds, objective_k5, n_iters,
                    KG_policy, q, value_fig_name, mask_k5_fig_name, sampling_ratio, True)  
    params_best_qKG = x_qKG[np.argmax(y_qKG)]
    recon_loss(params_best_qKG, data_loader, sampling_ratio, K=20, save_name='Recons_qKG_k5')

    mask_k10_fig_name = './bo_results/mask_k10_best_qKG.png'
    best_qKG_k10, x_qKG, y_qKG = policy_update(data_k10, bounds, objective_k10, n_iters,
                    KG_policy, q, value_fig_name, mask_k10_fig_name, sampling_ratio, True)  
    params_best_qKG = x_qKG[np.argmax(y_qKG)]
    recon_loss(params_best_qKG, data_loader, sampling_ratio, K=20, save_name='Recons_qKG_k10')


    if q == 1:
        fig, ax = plt.subplots()
        ax.plot(best_EI_k5, 'b+--', label = 'EI-k5')
        ax.plot(best_qEI_k5,'k*--', label = 'qEI-k5')
        ax.plot(best_qKG_k5, 'ro--', label = 'qKG-k5')
        ax.plot(best_EI_k10, 'b+-', label = 'EI-k10')
        ax.plot(best_qEI_k10,'k*-', label = 'qEI-k10')
        ax.plot(best_qKG_k10, 'ro-', label = 'qKG-k10')
        plt.xlabel('number of iteration')
        plt.ylabel('Best value found')
        plt.legend()
        plt.savefig('./bo_results/policy_comparison_best_results.png')

        plt.close()

    else:
        fig, ax = plt.subplots()
        ax.plot(best_qEI_k5,'k*--', label = 'qEI-k5')
        ax.plot(best_qKG_k5, 'ro--', label = 'qKG-k5')
        ax.plot(best_qEI_k10,'k*-', label = 'qEI-k10')
        ax.plot(best_qKG_k10, 'ro-', label = 'qKG-k10')
        plt.xlabel('number of iteration')
        plt.ylabel('Best value found')
        plt.legend()
        plt.savefig('./bo_results/policy_comparison_best_results.png')
        plt.close()