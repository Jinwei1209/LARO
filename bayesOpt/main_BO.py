import matplotlib.pyplot as plt
import numpy as np

from torch.utils import data
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from matplotlib import cm
from bayesOpt.bayes_opt import *
from bayesOpt.plotters import *
from bayesOpt.sample_loss import *
from loader.kdata_loader_GE import kdata_loader_GE

if __name__ == '__main__':

    contrast = 'T1'
    sampling_ratio = 0.1
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    print('Contrast is {0}, Sampling ratio is {1}'.format(contrast, sampling_ratio))

    rootName = '/data/Jinwei/{0}_slice_recon_GE'.format(contrast)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_loader = kdata_loader_GE(
            rootDir=rootName,
            contrast=contrast, 
            split='test',
            noiseLevel = 1
        )
    data_loader = data.DataLoader(data_loader, batch_size=4, shuffle=False)

    # bounds = np.array([[-1, 1], [-1, 1]])  # for log uniform space
    bounds = np.array([[0, 10], [0, 10]])  # for uniform space


    sample_loss = lambda z: recon_loss(z, data_loader, device, sampling_ratio)

    xp, yp = bayesian_optimisation(n_iters=30, 
                                sample_loss=sample_loss,
                                # x0 = np.array([[0.5, 2.5], [2.5, 1.5], [2.5, 2.5]]), 
                                bounds=bounds,
                                n_pre_samples=3,
                                random_search=10000)

    # p_pattern = gen_pattern(a=9.99, b=9.94, r_spacing=3)
    # plt.figure()
    # plt.imshow(p_pattern)
    # plt.colorbar()
    # plt.savefig('p_pattern.png')

    # print(np.mean(p_pattern))

    # u = np.random.uniform(size=(256, 192))
    # plt.imshow(p_pattern>u)
    # np.mean(p_pattern>u)
    # plt.savefig('binary_pattern.png')



    
    