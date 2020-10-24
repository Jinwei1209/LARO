import torch
import numpy as np

from models.dc_blocks import mlpy_in_cg, conj_in_cg, dvd_in_cg

def fit_R2_LM(s0, max_iter=10, tol=1e-2):
    '''
        non-fitting to get R2s and water
    '''
    nsamples = s0.size()[0]
    nechos = s0.size()[2]
    nrows = s0.size()[3]
    ncols = s0.size()[4]
    # s0[s0==0] = 1e-8
    tr = torch.arange(0, nechos)[None, None, :, None, None]  # 1st dimen: real = 1
    tr = tr.repeat(nsamples, 1, 1, nrows, ncols).to('cuda')  # echo time
    tc = torch.cat((tr, torch.zeros(tr.size()).to('cuda')), dim=1) # complex t

    y = torch.zeros(nsamples, 4, nrows, ncols).to('cuda')  # 1st dimension: 0:2 for R2s, 2:4 for water
    y[:, 2, ...] = torch.sqrt(s0[:, 0, 0, ...]**2 + s0[:, 1, 0, ...]**2)

    P_mag = torch.exp(y[:, 0:1, None, ...].repeat(1, 1, nechos, 1, 1) * tr) 
    P_phase = y[:, 1:2, None, ...].repeat(1, 1, nechos, 1, 1) * tr
    P = torch.cat((P_mag * torch.cos(P_phase),  P_mag * torch.sin(P_phase)), dim=1) # signal decay

    i = 0
    angles = torch.atan(s0[:, 1:2, ...] / s0[:, 0:1, ...])
    expi_angles = torch.cat((torch.cos(angles), torch.sin(angles)), dim=1)
    while (i<max_iter):
        sn = mlpy_in_cg(mlpy_in_cg(P, y[:, 2:4, None, ...].repeat(1, 1, nechos, 1, 1)), expi_angles)
        sr = s0 - sn

        Bcol01 = mlpy_in_cg(tc, sn)
        Bcol02 = mlpy_in_cg(P, expi_angles)
        dy = invB(Bcol01, Bcol02, sr)

        y += dy
        i += 1
        P_mag = torch.exp(y[:, 0:1, None, ...].repeat(1, 1, nechos, 1, 1) * tr) 
        P_phase = y[:, 1:2, None, ...].repeat(1, 1, nechos, 1, 1) * tr
        P = torch.cat((P_mag * torch.cos(P_phase), P_mag * torch.sin(P_phase)), dim=1) # signal decay


        update = dy[:, 0, ...]
        update[torch.isnan(update)] = 0
        update[torch.isinf(update)] = 0

    y[torch.isnan(y)] = 0
    y[torch.isinf(y)] = 0
    return y


def invB(col1, col2, y):
    # assemble A^H*A
    b11 = torch.sum(mlpy_in_cg(conj_in_cg(col1), col1), dim=2)
    b12 = torch.sum(mlpy_in_cg(conj_in_cg(col1), col2), dim=2)
    b22 = torch.sum(mlpy_in_cg(conj_in_cg(col2), col2), dim=2)

    # inversion of A^H*A
    d = (mlpy_in_cg(b11, b22) - mlpy_in_cg(b12, conj_in_cg(b12)))
    ib11 = dvd_in_cg(b22, d)
    ib12 =  - dvd_in_cg(b12, d)
    ib22 = dvd_in_cg(b11, d)

    # y project onto A^H
    py1 = torch.sum(mlpy_in_cg(conj_in_cg(col1), y), dim=2)
    py2 = torch.sum(mlpy_in_cg(conj_in_cg(col2), y), dim=2)

    # calculate dy
    dy1 = mlpy_in_cg(ib11, py1) + mlpy_in_cg(ib12, py2)
    dy2 = mlpy_in_cg(conj_in_cg(ib12), py1) + mlpy_in_cg(ib22, py2)

    return torch.cat((dy1, dy2), dim=1)
