''' 
bayesian optimization using expected improvement policy 
to find best implementation choice 
'''
import numpy as np
import torch
import os

from torch.utils import data
from loader.kdata_loader_GE import kdata_loader_GE
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition import qKnowledgeGradient
from botorch.acquisition import PosteriorMean
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from botorch.acquisition import qExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.sampling import manual_seed
from scipy.stats import norm
from scipy.optimize import minimize
from bayesOpt.sample_loss import *


def EI_policy(train_x, train_y, bounds, objective):

		# Fit the model by GP
		# Take train_x and train_y as training data and producte a fitted "model" object
		noises = torch.zeros(len(train_y))
		model  = SingleTaskGP(torch.tensor(train_x), torch.tensor(train_y).unsqueeze(-1),
							likelihood = FixedNoiseGaussianLikelihood(noise=noises))
		# optimize the model hyperparam by maximizing the log marginal likelihood
		mll    = ExactMarginalLogLikelihood(model.likelihood, model)
		fit_gpytorch_model(mll)

		# Optimize EI
		BoTorch_EI = ExpectedImprovement(model=model, best_f=train_y.max())
		new_point, new_point_EI = optimize_acqf(
		    acq_function=BoTorch_EI,
		    bounds=torch.tensor(bounds),
		    q=1,
		    num_restarts=50,
		    raw_samples=100,
		    options={},
		)

		# Evaluate the objective
		new_value = np.array([objective(new_point.numpy()[0])])

		return new_point, new_value


def qEI_policy(train_x, train_y, bounds, objective, q=1):

		# Take train_x and train_y and fit the model by GP
		model  = SingleTaskGP(torch.tensor(train_x), torch.tensor(train_y).unsqueeze(-1))
		mll    = ExactMarginalLogLikelihood(model.likelihood, model)
		fit_gpytorch_model(mll)
		# construct MC EI 
		sampler = SobolQMCNormalSampler(num_samples=500, seed=0, resample=False)
		MC_EI = qExpectedImprovement(model, best_f=train_y.max(), sampler=sampler)
		# optimize qEI for q-batch
		torch.manual_seed(seed=0) # to keep the restart conditions the same
		candidates, acq_value = optimize_acqf(
		    acq_function=MC_EI,
		    bounds=bounds,
		    q=q,
		    num_restarts=20,
		    raw_samples=100,
		    options={},
		)
		# Evaluate the objective
		new_values = np.array([objective(candidates[i].numpy()) for i in range(len(candidates))])

		return candidates, new_values



def KG_policy(train_x, train_y, bounds, objective, q=1):

		# Fit the model by GP
		# Take train_x and train_y as training data and producte a fitted "model" object
		# noises = torch.zeros(len(train_y))
		# model  = SingleTaskGP(torch.tensor(train_x), torch.tensor(train_y).unsqueeze(-1),
		# 					likelihood = FixedNoiseGaussianLikelihood(noise=noises))
		# optimize the model hyperparam by maximizing the log marginal likelihood
		model  = SingleTaskGP(torch.tensor(train_x), torch.tensor(train_y).unsqueeze(-1))
		mll    = ExactMarginalLogLikelihood(model.likelihood, model)
		fit_gpytorch_model(mll)

		# construct the actual qKG based on poesterior mean
		argmax_pmean, max_pmean = optimize_acqf(
			acq_function=PosteriorMean(model), 
			bounds=torch.tensor(bounds),
			q=1,
			num_restarts=50,
			raw_samples=2048,
		)

		qKG_offset = qKnowledgeGradient(model, num_fantasies=128)

		qKG = qKnowledgeGradient(
			model,
			num_fantasies=128,
			sampler=qKG_offset.sampler,
			current_value=max_pmean,
		)

		# optimize the qKG
		with manual_seed(1234):
			candidates, acq_value = optimize_acqf(
				acq_function=qKG, 
				bounds=torch.tensor(bounds),
				q=q,
				num_restarts=10,
				raw_samples=512,
			)

		# Evaluate the objective
		new_values = np.array([objective(candidates[i].numpy()) for i in range(len(candidates))])

		return candidates, new_values


def policy_update(x, y, bounds, objective, n_iters, best, a_best, b_best,
                policy_func, value_fig_name, mask_fig_name, Plot = False):
    
    for i in range(n_iters):
        new_point, new_value = policy_func(train_x = x, train_y = y, bounds = bounds, objective = objective)
        # Add the new data
        x = np.concatenate((x, new_point.numpy()))
        y = np.concatenate((y, new_value))
        best.append(y.max())

        if best[-1] != best[-2]:
            a_best, b_best = new_point[np.argmax(new_value)]
            print('Update best parameters')

        # print('Iteration {:2d}, value={:0.3f}, best value={:0.3f}'.format(i, new_value, best[-1]))
        print('Iteration {:2d}, best value={:0.3f}'.format(i, best[-1]))
        print()

    if Plot:
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

