import torch
import numpy as np

from models.dc_blocks import mlpy_in_cg, conj_in_cg, dvd_in_cg


def fit_R2_LM(M, max_iter=10, tol=1e-2):
    '''
        nonlinear fitting to get R2s and water,
        M: (batch, height, width, 2, nechos)
    '''
    matrix_size = M.size()
    nechos = matrix_size[-1]
    s0 = M.view(-1, 2, nechos).permute(2, 1, 0)  # (nechos, 2, nvoxels)
    numvox = s0.size()[-1]

    # s0[s0==0] = 1e-8
    tr = torch.arange(0, nechos)[:, None, None]  # 1st dim: real = 1
    tr = tr.repeat(1, 1, numvox).to('cuda')  # echo time
    tc = torch.cat((tr, torch.zeros(tr.size()).to('cuda')), dim=1) # complex t, (nechos, 2, nvoxels)

    y = torch.zeros(4, numvox).to('cuda')  # 0st dimension: 0:2 for R2s, 2:4 for water
    y[2, :] = torch.sqrt(s0[0, 0, :]**2 + s0[0, 1, :]**2)

    y_times_t = mlpy_in_cg(y[None, 0:2, :].repeat(nechos, 1, 1), tc)
    P_mag = torch.exp(y_times_t[:, 0:1, :])
    P_phase = y_times_t[:, 1:2, :]
    P = torch.cat((P_mag * torch.cos(P_phase),  P_mag * torch.sin(P_phase)), dim=1) # signal decay

    i = 0
    angles = torch.atan2(s0[:, 1:2, :], s0[:, 0:1, :])
    expi_angles = torch.cat((torch.cos(angles), torch.sin(angles)), dim=1)
    while (i<max_iter):
        sn = mlpy_in_cg(mlpy_in_cg(P, y[None, 2:4, :].repeat(nechos, 1, 1)), expi_angles)
        sr = s0 - sn

        Bcol01 = mlpy_in_cg(tc, sn)
        Bcol02 = mlpy_in_cg(P, expi_angles)
        dy = invB(Bcol01, Bcol02, sr)

        # print(y.size())
        # print(dy.size())
        y += dy
        i += 1

        y_times_t = mlpy_in_cg(y[None, 0:2, :].repeat(nechos, 1, 1), tc)
        P_mag = torch.exp(y_times_t[:, 0:1, :])
        P_phase = y_times_t[:, 1:2, :]
        P = torch.cat((P_mag * torch.cos(P_phase),  P_mag * torch.sin(P_phase)), dim=1) # signal decay
    
    R2s = - y[0, :].view(matrix_size[0], matrix_size[1], matrix_size[2])
    water = y[2, :].view(matrix_size[0], matrix_size[1], matrix_size[2])
    R2s[torch.isnan(R2s)] = 0
    R2s[torch.isinf(R2s)] = 0
    water[torch.isnan(R2s)] = 0
    water[torch.isinf(R2s)] = 0
    return [R2s, water]


def invB(col1, col2, y):
    # assemble A^H*A
    b11 = torch.sum(mlpy_in_cg(conj_in_cg(col1), col1), dim=0, keepdim=True)  # (1, 2, nvoxels)
    b12 = torch.sum(mlpy_in_cg(conj_in_cg(col1), col2), dim=0, keepdim=True)
    b22 = torch.sum(mlpy_in_cg(conj_in_cg(col2), col2), dim=0, keepdim=True)

    # inversion of A^H*A
    d = (mlpy_in_cg(b11, b22) - mlpy_in_cg(b12, conj_in_cg(b12)))
    ib11 = dvd_in_cg(b22, d)
    ib12 =  - dvd_in_cg(b12, d)
    ib22 = dvd_in_cg(b11, d)

    # y project onto A^H
    py1 = torch.sum(mlpy_in_cg(conj_in_cg(col1), y), dim=0, keepdim=True)
    py2 = torch.sum(mlpy_in_cg(conj_in_cg(col2), y), dim=0, keepdim=True)

    # calculate dy
    dy1 = mlpy_in_cg(ib11, py1) + mlpy_in_cg(ib12, py2)
    dy2 = mlpy_in_cg(conj_in_cg(ib12), py1) + mlpy_in_cg(ib22, py2)

    return torch.cat((dy1[0, ...], dy2[0, ...]), dim=0)


def arlo(te, y, flag_water=0):
    '''
        arlo for R2s estimation, y: (batch, height, width, nechos)
    '''
    nte = len(te)
    if nte < 2:
        return []
    
    sz = y.size()
    edx = len(sz)
    if sz[edx-1] != nte:
        raise Exception("Number of echoes does not match mGRE images")

    yy = torch.zeros(sz[:-1]).to('cuda')
    yx = torch.zeros(sz[:-1]).to('cuda')
    beta_yx = torch.zeros(sz[:-1]).to('cuda')
    beta_xx = torch.zeros(sz[:-1]).to('cuda')

    for j in range(nte-2):
        alpha = (te[j+2]-te[j])*(te[j+2]-te[j])/2/(te[j+1]-te[j])
        tmp = (2*te[j+2]*te[j+2] - te[j]*te[j+2] - te[j]*te[j] + 3*te[j]*te[j+1] -3*te[j+1]*te[j+2])/6
        beta = tmp/(te[j+2]-te[j+1])
        gamma = tmp/(te[j+1]-te[j])
        y1 =  y[..., j] * (te[j+2]-te[j]-alpha+gamma) + y[..., j+1] * (alpha-beta-gamma) + y[..., j+2]*beta
        x1 = y[..., j] - y[..., j+2]
        yy = yy + y1 * y1
        yx = yx + y1 * x1
        beta_yx = beta_yx + beta * y1 * x1
        beta_xx = beta_xx + beta * x1 * x1

    r2 =  (yx + beta_xx) / (beta_yx + yy)
    r2[torch.isnan(r2)] = 0
    r2[torch.isinf(r2)] = 0

    if flag_water:
        A = torch.exp(-r2[..., None] * torch.tensor(np.array(te)[None, None, None, :]).to('cuda'))
        water = torch.sum(A * y, dim=-1) / torch.sum(A * A, dim=-1)
        return [r2, water]
    else:
        return r2


def fit_complex(M, max_iter=30):
    '''
        fit_ppm_complex, M: (batch, height, width, 2, nechos)
    '''
    pi = 3.14159265358979323846
    # M[:, :, :, 1, :] =  - M[:, :, :, 1, :]
    s0 = M.size()
    nechos = s0[-1]

    M = M.view(-1, 2, nechos)
    s = M.size()

    Y = torch.atan2(M[:, 1, :3], M[:, 0, :3])  # signed angle
    c = Y[:, 1:2] - Y[:, 0:1]
    ind = torch.argmin(torch.cat((torch.abs(c-2*pi), torch.abs(c), torch.abs(c+2*pi)), dim=1), dim=1)
    c = c[:, 0]
    c[ind==0] = c[ind==0] - 2*pi
    c[ind==2] = c[ind==2] + 2*pi

    for n in range(2):
        cd = Y[:, n+1] - Y[:, n] - c
        Y[cd<-pi, n+1:] = Y[cd<-pi, n+1:] + 2*pi
        Y[cd>pi, n+1:] = Y[cd>pi, n+1:] - 2*pi

    A = torch.tensor([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]).to('cuda')
    AA_inv = torch.tensor([[5/6, -1/2], [-1/2, 1/2]]).to('cuda')
    ip = torch.matmul(AA_inv, A.permute(1, 0))
    ip = torch.matmul(ip, Y.permute(1, 0))
    p0 = ip[0, :]
    p1 = ip[1, :]

    dp1 = p1
    tol = torch.mean(p1**2) * 1e-4
    idx_iter = 0

    # weigthed least square, calculation of WA'*WA
    v1 = torch.ones((1, nechos)).to('cuda')
    v2 = torch.range(0, nechos-1)[None, :].to('cuda')
    tmp = torch.ones((s[0], 1)).to('cuda')
    abs_M = torch.sqrt(M[:, 0, :]**2 + M[:, 1, :]**2)

    a11 = torch.sum(abs_M**2 * (torch.matmul(tmp, v1**2)), dim=1)
    a12 = torch.sum(abs_M**2 * (torch.matmul(tmp, v1*v2)), dim=1)
    a22 = torch.sum(abs_M**2 * (torch.matmul(tmp, v2**2)), dim=1)

    # inversion
    d = a11 * a22 - a12**2
    ai11 = a22/d
    ai12 = -a12/d
    ai22 = a11/d

    tmp1 = torch.matmul(tmp, v1)[:, None, :]
    tmp1 = torch.cat((tmp1, torch.zeros(tmp1.size()).to('cuda')), dim=1)  # (nvoxel, 2, necho)
    tmp2 = torch.matmul(tmp, v2)[:, None, :]
    tmp2 = torch.cat((tmp2, torch.zeros(tmp2.size()).to('cuda')), dim=1)  # (nvoxel, 2, necho)

    abs_M = abs_M[:, None, :]

    while torch.mean(dp1**2) > tol and idx_iter < max_iter:
        idx_iter += 1
        W_phase = torch.matmul(p0[:, None], v1) + torch.matmul(p1[:, None], v2)
        W_phase = W_phase[:, None, :]
        # abs_M = abs_M[:, None, :]
        W = torch.cat((abs_M*torch.cos(W_phase), abs_M*torch.sin(W_phase)), dim=1)  # (nvoxel, 2, necho)
        conj_1iW = torch.cat((-abs_M*torch.sin(W_phase), -abs_M*torch.cos(W_phase)), dim=1)  # (nvoxel, 2, necho)

        # projection
        # tmp1 = torch.matmul(tmp, v1)[:, None, :]
        # tmp1 = torch.cat((tmp1, torch.zeros(tmp1.size()).to('cuda')), dim=1)  # (nvoxel, 2, necho)
        # tmp2 = torch.matmul(tmp, v2)[:, None, :]
        # tmp2 = torch.cat((tmp2, torch.zeros(tmp2.size()).to('cuda')), dim=1)  # (nvoxel, 2, necho)
        pr1 = torch.sum(mlpy_in_cg(mlpy_in_cg(conj_1iW, tmp1), M-W), dim=2)  # (nvoxel, 2)
        pr2 = torch.sum(mlpy_in_cg(mlpy_in_cg(conj_1iW, tmp2), M-W), dim=2)  # (nvoxel, 2)

        dp0 = ai11 * pr1[:, 0] + ai12 * pr2[:, 0]  # (nvoxel,)
        dp1 = ai12 * pr1[:, 0] + ai22 * pr2[:, 0]  # (nvoxel,)
        dp0[torch.isnan(dp0)] = 0
        dp1[torch.isnan(dp1)] = 0

        # update
        p0 = p0 + dp0
        p1 = p1 + dp1

    p1[p1>pi] = torch.fmod(p1[p1>pi] + pi, 2*pi) - pi
    p1[p1<-pi] = torch.fmod(p1[p1<-pi] + pi, 2*pi) - pi
    p1 = p1.view(s0[0], s0[1], s0[2])
    p0 = p0.view(s0[0], s0[1], s0[2])
    return [p1, p0]
    

