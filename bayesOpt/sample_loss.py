"""functions for the tasks: Optimal kspace sampling pattern design
"""

import numpy as np
import math

from models.dc_st_pmask import *
from utils.test import *

def gen_pattern(a, b, r_spacing=3):
    M_c = int(256/2)  # center of kspace
    N_c = int(192/2)  # center of kspace
    n = math.floor(M_c/r_spacing)  # number of regions
    p_pattern = np.zeros((256, 192))
    # generate density of the pattern
    k = 0
    for r_1 in range(0, M_c, r_spacing):
        r_2 = r_1 + r_spacing
        for i in range(256):
            for j in range(192):
                if p_pattern[i, j] != 0:
                    continue
                else:
                    radius = np.sqrt((i - M_c)**2 + (j - N_c)**2)
                    if radius <  r_2 and radius >= r_1:
                        p_pattern[i, j] = np.exp(-(b*k/n)**a)
        k += 1
    # keep the center square for calibration
    p_pattern[M_c-13:M_c+12, N_c-13:N_c+12] = 1
    return p_pattern

def recon_loss(params, data_loader, device, sampling_ratio=0.1):
    print('a = {0}, b = {1}'.format(params[0], params[1]))
    # p_pattern = gen_pattern(a=10**params[0], b=10**params[1], r_spacing=3)
    p_pattern = gen_pattern(a=params[0], b=params[1], r_spacing=3)
    model = DC_ST_Pmask(input_channels=2, filter_channels=32, lambda_dll2=1e-4, 
                        lambda_tv=1e-4, rho_penalty=1e-2, flag_ND=3, flag_solver=-3, 
                        flag_TV=1, K=20, rescale=True, samplingRatio=sampling_ratio, flag_fix=1, pmask_BO=p_pattern)
    metrices_test = Metrices()
    model.to(device)
    for idx, (inputs, targets, csms, brain_masks) in enumerate(data_loader):
        # if idx % 10 == 0:
            # print('Reconstructing slice #{0}'.format(idx))
        inputs = inputs.to(device)
        targets = targets.to(device)
        csms = csms.to(device)
        # calculating metrices
        Xs = model(inputs, csms)
        metrices_test.get_metrices(Xs[-1], targets)
    ave_psnr = np.mean(np.asarray(metrices_test.PSNRs))
    return 10**(ave_psnr-39)


    

