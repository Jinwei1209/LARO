"""functions for the tasks: Optimal kspace sampling pattern design
"""

import numpy as np
import math
import torch
import time

from models.dc_st_pmask import *
from utils.test import *

# # slow
# def gen_pattern(a, b, r_spacing=3):
#     M_c = int(256/2)  # center of kspace
#     N_c = int(192/2)  # center of kspace
#     n = math.floor(M_c/r_spacing)  # number of regions
#     p_pattern = np.zeros((256, 192))
#     # generate density of the pattern
#     k = 0
#     for r_1 in range(0, M_c, r_spacing):
#         r_2 = r_1 + r_spacing
#         for i in range(256):
#             for j in range(192):
#                 if p_pattern[i, j] != 0:
#                     continue
#                 else:
#                     radius = np.sqrt((i - M_c)**2 + (j - N_c)**2)
#                     if radius <  r_2 and radius >= r_1:
#                         p_pattern[i, j] = np.exp(-(b*k/n)**a)
#         k += 1
#     # keep the center square for calibration
#     p_pattern[M_c-13:M_c+12, N_c-13:N_c+12] = 1
#     return p_pattern

# # fast
def gen_pattern(a, b, num_row=256, num_col=192, r_spacing=3):
    M_c = int(num_row/2)  # center of kspace
    N_c = int(num_col/2)  # center of kspace
    n = math.floor(M_c/r_spacing)  # number of regions
    p_pattern = np.zeros((num_row, num_col)).flatten()
    indices = np.array(np.meshgrid(range(num_row), range(num_col))).T.reshape(-1,2)  # row first
    distances = np.sqrt((indices[:, 0] - M_c)**2 + (indices[:, 1] - N_c)**2)
    distances_orders = distances // r_spacing  # get the order of the distances
    for k in range(M_c//r_spacing + 1):
        p_pattern[distances_orders == k] = np.exp(-(b*k/n)**a)
    p_pattern = p_pattern.reshape(num_row, num_col)
    p_pattern[M_c-13:M_c+12, N_c-13:N_c+12] = 1
    return p_pattern

def recon_loss(params, data_loader, sampling_ratio=0.1, K=5, save_name=None):
    print('a = {0}, b = {1}'.format(params[0], params[1]))
    p_pattern = gen_pattern(a=10**params[0], b=10**params[1], r_spacing=3)
    # p_pattern = gen_pattern(a=params[0], b=params[1], r_spacing=3)
    model = DC_ST_Pmask(input_channels=2, filter_channels=32, lambda_dll2=1e-4, 
                        lambda_tv=1e-4, rho_penalty=1e-2, flag_ND=3, flag_solver=-3, 
                        flag_TV=1, K=K, rescale=True, samplingRatio=sampling_ratio, flag_fix=1, pmask_BO=p_pattern)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    metrices_test = Metrices()
    model.cuda()
    t0 = time.time()
    recons = []
    with torch.no_grad(): 
        for idx, (inputs, targets, csms, brain_masks) in enumerate(data_loader):
            # if idx % 10 == 0:
                # print('Reconstructing slice #{0}'.format(idx))
            inputs = inputs.cuda()
            targets = targets.cuda()
            csms = csms.cuda()
            # calculating metrices
            Xs = model(inputs, csms)
            recons.append(Xs[-1].cpu().detach())
            metrices_test.get_metrices(Xs[-1], targets)
    ave_psnr = np.mean(np.asarray(metrices_test.PSNRs))
    print('Total Time to evaluate objective function once is: %.2f s' % (time.time()-t0))
    if save_name is not None:
        recons = np.concatenate(recons, axis=0)
        recons = np.transpose(recons, [2,3,0,1])
        recons = np.squeeze(np.sqrt(recons[..., 0]**2 + recons[..., 1]**2))
        adict = {}
        adict['recons'] = recons
        sio.savemat('./results/{0}.mat'.format(save_name), adict)
    return 10**(ave_psnr-39)


    

