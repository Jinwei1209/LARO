import matplotlib.pyplot as plt
import numpy as np
import torch

from torch.utils import data
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood


def remove(T,i):
    # Remove the ith component from the 1-dimensional tensor T
    assert(i<len(T))
    return np.concatenate([T[:i], T[i+1:]])


def cross_validation(train_x, train_y):
    loo_mean = []
    loo_sdev = []

    for i in range(len(train_x)):
        
        # Remove the ith datapoint from the training set
        loo_train_x = remove(train_x,i)
        loo_train_y = remove(train_y,i)
        noises = torch.zeros(len(loo_train_y))
        # fit the GP model
        model = SingleTaskGP(torch.tensor(loo_train_x), torch.tensor(loo_train_y).unsqueeze(-1), 
                            likelihood = FixedNoiseGaussianLikelihood(noise=noises))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll) 
        # # Get posterior mean and variance
        posterior = model.posterior(torch.tensor(train_x)[i].unsqueeze(0))
        m = posterior.mean.cpu().detach().numpy()[0][0]
        v = posterior.variance.cpu().detach().numpy()[0][0]
        
        loo_mean.append(m)
        loo_sdev.append(1.5*np.sqrt(v))
    
    fig, ax = plt.subplots()

    ax.errorbar(train_y,loo_mean,loo_sdev,fmt='o')
    ax.plot([min(train_y),max(train_y)],[min(train_y),max(train_y)],'k--')

    plt.xlabel('observed y')
    plt.ylabel('predicted y')
    plt.savefig('cross_validation.png')