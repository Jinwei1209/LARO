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



def policy_update(x, y, bounds, objective, n_iters, best, a_best, b_best,
                policy_func, value_fig_name, mask_fig_name):

    for i in range(n_iters):
        new_point, new_value = policy_func(train_x = x, train_y = y, bounds = bounds, objective = objective)
        # Add the new data
        x = np.concatenate((x, new_point.numpy()))
        y = np.concatenate((y, new_value.numpy()))
        best.append(y.max())

        if best[-1] != best[-2]:
            a_best, b_best = new_point[np.argmax(new_values)].numpy()[0]
            print('Update best parameters')

        print('Iteration {:2d}, value={:0.3f}, best value={:0.3f}'.format(i, new_value, best[-1]))
        print()

    plt.figure()
    plt.plot(best,'o-')
    plt.xlabel('Iteration')
    plt.ylabel('Best value found')
    plt.savefig(value_fig_name)
    plt.close()

    p_pattern = gen_pattern(10**a_best, 10**b_best, r_spacing=3)
    # p_pattern = gen_pattern(a_best, b_best, r_spacing=3)
    u = np.random.uniform(0, np.mean(p_pattern)/sampling_ratio, size=(256, 192))
    masks = p_pattern > u
    masks[128-13:128+12, 96-13:96+12] = 1
    plt.figure()
    plt.imshow(masks)
    plt.savefig(mask_fig_name)
    plt.close()



if __name__ == '__main__':

    # typein parameters
    parser = argparse.ArgumentParser(description='BO-LOUPE')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--flag_policy', type=int, default=0)  # 0 for EI, 1 for KG
    opt = {**vars(parser.parse_args())}
    # fixed parameters
    q = 1  # number of lookahead
    contrast = 'T1'
    sampling_ratio = 0.1
    n_pre_samples = 3
    n_iters = 30

    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_id']

    print('Contrast is {0}, Sampling ratio is {1}'.format(contrast, sampling_ratio))

    rootName = '/data/Jinwei/{0}_slice_recon_GE'.format(contrast)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_loader = kdata_loader_GE(
            rootDir=rootName,
            contrast=contrast, 
            split='test',
            SNR = 10
        )
    data_loader = data.DataLoader(data_loader, batch_size=4, shuffle=False)

    bounds = np.array([[-1., -1.], [1., 1.]])  # for log uniform space
    # bounds = np.array([[0., 0.], [10., 10.]])  # for uniform space

    print('device: {}'.format(device))

    objective = lambda z: recon_loss(z, data_loader, device, sampling_ratio)

    x_list, y_list = [], []
    # random initialization
    for params in np.random.uniform(bounds[0], bounds[1], (n_pre_samples, bounds.shape[0])):
            x_list.append(params)
            y_list.append(objective(params))
    
    x = np.array(x_list)
    y = np.array(y_list)
    print('Value of best point found: {}'.format(y.max()))

    a_best, b_best = x[np.argmax(y)]
    best = [y.max()] # This will store the best value

    if opt['flag_policy'] == 0:
        value_fig_name = 'Values_EI.png'
        mask_fig_name = 'mask_best_EI.png'
        policy_update(x, y, bounds, objective, n_iters, best, a_best, b_best,
                        EI_policy, value_fig_name, mask_fig_name)
    else:
        value_fig_name = 'Values_KG.png'
        mask_fig_name = 'mask_best_KG.png'
        policy_update(x, y, bounds, objective, n_iters, best, a_best, b_best,
                        KG_policy, value_fig_name, mask_fig_name)

    # for i in range(n_iters):
    #     new_point, new_value = policy(train_x = x, train_y = y, bounds = bounds, objective = objective)
    #     # Add the new data
    #     x = np.concatenate((x, new_point.numpy()))
    #     y = np.concatenate((y, new_value.numpy()))
    #     best.append(y.max())

    #     if best[-1] != best[-2]:
    #         a_best, b_best = new_point[np.argmax(new_values)].numpy()[0]
    #         print('Update best parameters')

    #     print('Iteration {:2d}, value={:0.3f}, best value={:0.3f}'.format(i, new_value, best[-1]))
    #     print()

    # plt.figure()
    # plt.plot(best,'o-')
    # plt.xlabel('Iteration')
    # plt.ylabel('Best value found')
    # plt.savefig(value_fig_name)
    # plt.close()

    # p_pattern = gen_pattern(10**a_best, 10**b_best, r_spacing=3)
    # # p_pattern = gen_pattern(a_best, b_best, r_spacing=3)
    # u = np.random.uniform(0, np.mean(p_pattern)/sampling_ratio, size=(256, 192))
    # masks = p_pattern > u
    # masks[128-13:128+12, 96-13:96+12] = 1
    # plt.figure()
    # plt.imshow(masks)
    # plt.savefig(mask_fig_name)
    # plt.close()







    
    