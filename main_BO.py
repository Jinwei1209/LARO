import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import argparse

from torch.utils import data
from loader.kdata_loader_GE import kdata_loader_GE
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition import qKnowledgeGradient
from botorch.optim import optimize_acqf
from botorch.utils.sampling import manual_seed
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from scipy.stats import norm
from scipy.optimize import minimize
from bayesOpt.sample_loss import *

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
            SNR = 100
        )
    data_loader = data.DataLoader(data_loader, batch_size=4, shuffle=False)

    bounds = np.array([[-1., -1.], [1., 1.]])  # for log uniform space
    # bounds = np.array([[0., 0.], [10., 10.]])  # for uniform space

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
        # EI policy
        for i in range(n_iters):
            # Fit the model
            noises = torch.zeros(len(y))
            model = SingleTaskGP(torch.tensor(x), torch.tensor(y).unsqueeze(-1), likelihood = FixedNoiseGaussianLikelihood(noise=noises))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)

            # Optimize EI
            BoTorch_EI = ExpectedImprovement(model=model, best_f=y.max())
            new_point, new_point_EI = optimize_acqf(
                acq_function=BoTorch_EI,
                bounds=torch.tensor(bounds),
                q=1,
                num_restarts=50,
                raw_samples=100,
                options={},
            )

            # Evaluate the objective
            new_value = objective(new_point.numpy()[0])
            y_list.append(new_value)

            # Add the new data
            x = np.concatenate((x, new_point.numpy()))
            y = np.array(y_list)
            best.append(y.max())

            if y[-1] == best[-1]:
                a_best, b_best = new_point.numpy()[0]
                print('Update best parameters')

            print('Iteration {:2d}, value={:0.3f}, best value={:0.3f}'.format(i, new_value, best[-1]))
            print()

        plt.figure()
        plt.plot(best,'o-')
        plt.xlabel('Iteration')
        plt.ylabel('Best value found')
        plt.savefig('Values_EI.png')
        plt.close()

        p_pattern = gen_pattern(10**a_best, 10**b_best, r_spacing=3)
        # p_pattern = gen_pattern(a_best, b_best, r_spacing=3)
        u = np.random.uniform(0, np.mean(p_pattern)/sampling_ratio, size=(256, 192))
        masks = p_pattern > u
        masks[128-13:128+12, 96-13:96+12] = 1
        plt.figure()
        plt.imshow(masks)
        plt.savefig('mask_best_EI.png')
        plt.close()

    else:
        # KG policy
        for i in range(n_iters):
            # Fit the model
            model = SingleTaskGP(torch.tensor(x), torch.tensor(y).unsqueeze(-1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)

            # Optimize gKG
            qKG = qKnowledgeGradient(model=model, num_fantasies=128)
            with manual_seed(1234):
                candidates, acq_value = optimize_acqf(
                    acq_function=qKG, 
                    bounds=torch.tensor(bounds),
                    q=q,
                    num_restarts=10,
                    raw_samples=512,
                    options={},
                )

            # Evaluate the objective
            for j in range(q):
                new_value = objective(candidates[j:j+1].numpy()[0])
                y_list.append(new_value)

            # Add the new data
            x = np.concatenate((x, candidates.numpy()))
            y = np.array(y_list)
            best.append(y.max())

            for j in range(q):
                if y[-1-j] == best[-1]:
                    a_best, b_best = candidates[j:j+1].numpy()[0]
                    print('Update best parameters')

            print('Iteration {:2d}, value={:0.3f}, best value={:0.3f}'.format(i, np.max(y[-q:]), best[-1]))
            print()

        plt.figure()
        plt.plot(best,'o-')
        plt.xlabel('Iteration')
        plt.ylabel('Best value found')
        plt.savefig('values_KG.png')
        plt.close()

        p_pattern = gen_pattern(10**a_best, 10**b_best, r_spacing=3)
        # p_pattern = gen_pattern(a_best, b_best, r_spacing=3)
        u = np.random.uniform(0, np.mean(p_pattern)/sampling_ratio, size=(256, 192))
        masks = p_pattern > u
        masks[128-13:128+12, 96-13:96+12] = 1
        plt.figure()
        plt.imshow(masks)
        plt.savefig('mask_best_KG.png')
        plt.close()






    
    