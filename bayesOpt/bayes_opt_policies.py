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
		new_value = objective(new_point.numpy()[0])

		return new_point, new_value


def KG_policy(train_x, train_y, bounds, objective):

		# Fit the model by GP
		# Take train_x and train_y as training data and producte a fitted "model" object
		noises = torch.zeros(len(train_y))
		model  = SingleTaskGP(torch.tensor(train_x), torch.tensor(train_y).unsqueeze(-1),
							likelihood = FixedNoiseGaussianLikelihood(noise=noises))
		# optimize the model hyperparam by maximizing the log marginal likelihood
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

		qKG = qKnowledgeGradient(
			model,
			num_fantasies=128,
			sampler=qKG.sampler,
			current_value=max_pmean,
		)

		# optimize the qKG
		with manual_seed(1234):
			candidates, acq_value = optimize_acqf(
				acq_function=qKG, 
				bounds=torch.tensor(bounds),
				q=2,
				num_restarts=10,
				raw_samples=512,
			)

		# Evaluate the objective
		new_values = [objective(candidates[i].numpy()) for i in range(q)]

		return candidates, new_values

