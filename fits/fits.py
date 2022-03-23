from numpy.core.numeric import Inf
import torch
import numpy as np

from models.dc_blocks import mlpy_in_cg, conj_in_cg, dvd_in_cg
from skimage.restoration import unwrap_phase


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


def fit_complex_all(iField, TE):
    '''
    four parameter fitting (myfit_all in matlab):
        iField: complex multi-echo data (batch, height, width, numte)
        TE: echo times
    '''
    max_iter = 30
    tol = 1e-6
    reg_p = 1
    lambda_l2 = 1e-6
    modR2s = 1
    modm0 = 1

    alpha = 1
    delta_TE = TE[1] - TE[0]
    
    te = TE
    TE = torch.tensor(TE).to('cuda')
    numte = len(TE)

    # make the first echo phase all zeros
    iField = iField * torch_exp1j(-iField[..., 0:1].angle().repeat(1, 1, 1, numte))

    matrix_size = iField.size()[:-1]
    S = iField.view(-1, numte)
    numvox = S.size()[0]
    t = TE[None, :].repeat(numvox, 1)

    y = torch.zeros(4, numvox).to('cuda')

    # initialization of m0 and r2s
    [R2s, water] = arlo(te, iField.abs(), flag_water=1)
    y[0, :] = torch.flatten(water)
    y[1, :] = torch.flatten(R2s)

    nechos = iField.size()[-1]
    Y = S[:, :3].angle()

    # phase unwrapping of the first three echos
    Y = Y.view(matrix_size[0], matrix_size[1], matrix_size[2], 3).cpu().numpy()
    for i in range(3):
        Y[0, :, :, i] = unwrap_phase(Y[0, :, :, i])
    Y = torch.tensor(Y).to('cuda')
    Y = Y.view(-1, 3)
    c = Y[:, 1] - Y[:, 0]

    # initial of f0 and p (using the first two echos)
    A = torch.tensor([[1.0, te[0]], [1.0, te[1]]]).to('cuda')
    ip = torch.matmul(torch.inverse(A), Y[:, :2].permute(1, 0))
    y[2, :] = ip[0, :]
    y[3, :] = ip[1, :]
    y_init = y
    
    cf =  [ [11, 22, 33, 44], [11, 23, 34, 42], [11, 24, 32, 43],
            [11, 24, 33, 42], [11, 23, 32, 44], [11, 22, 34, 43],
            [12, 21, 33, 44], [13, 21, 34, 42], [14, 21, 32, 43],
            [14, 21, 33, 42], [13, 21, 32, 44], [12, 21, 34, 43],
            [12, 23, 31, 44], [13, 24, 31, 42], [14, 22, 31, 43],
            [14, 23, 31, 42], [13, 22, 31, 44], [12, 24, 31, 43],
            [12, 23, 34, 41], [13, 24, 32, 41], [14, 22, 33, 41],
            [14, 23, 32, 41], [13, 22, 34, 41], [12, 24, 33, 41] ]

    cf_sign = torch.ones(np.shape(cf)[0]).to('cuda')
    cf_sign[3:9] = - cf_sign[3:9]
    cf_sign[15:21] = - cf_sign[15:21]

    iteration = 0
    update = np.Inf
    while ((iteration < max_iter) and (update > tol)):
        W = y[0:1, :].permute(1, 0).repeat(1, numte) * torch.exp(-t*y[1:2, :].permute(1, 0).repeat(1, numte)) \
            * torch_exp1j(y[2:3, :].permute(1, 0).repeat(1, numte) + y[3:4, :].permute(1, 0).repeat(1, numte)*t)
        W_m0 = torch.exp(-t*y[1:2, :].permute(1, 0).repeat(1, numte)) \
            * torch_exp1j(y[2:3, :].permute(1, 0).repeat(1, numte) + y[3:4, :].permute(1, 0).repeat(1, numte)*t)

        a11 = torch.sum(torch.real(torch.conj(W_m0) * W_m0), dim=1)
        a12 = torch.sum(torch.real(-torch.conj(W_m0) * W * t), dim=1)
        a13 = torch.sum(torch.real(torch.conj(W_m0) * W * 1j), dim=1)
        a14 = torch.sum(torch.real(torch.conj(W_m0) * W * t * 1j), dim=1)
        a21 = torch.sum(torch.real(-torch.conj(W) * W_m0 * t), dim=1)
        a22 = torch.sum(torch.real(torch.conj(W) * W * t * t), dim=1)
        a23 = torch.sum(torch.real(-torch.conj(W) * W * 1j * t), dim=1)
        a24 = torch.sum(torch.real(-torch.conj(W) * W * t * 1j * t), dim=1)
        a31 = torch.sum(torch.real(torch.conj(W) * (-1j) * W_m0), dim=1)
        a32 = torch.sum(torch.real(-torch.conj(W) * (-1j) * W * t), dim=1)
        a33 = torch.sum(torch.real(torch.conj(W) * (-1j) * W * 1j), dim=1)
        a34 = torch.sum(torch.real(torch.conj(W) * (-1j) * W * t * 1j), dim=1)
        a41 = torch.sum(torch.real(torch.conj(W) * (-1j) * W_m0 * t), dim=1)
        a42 = torch.sum(torch.real(-torch.conj(W) * (-1j) * W * t * t), dim=1)
        a43 = torch.sum(torch.real(torch.conj(W) * (-1j) * W * 1j * t), dim=1)
        a44 = torch.sum(torch.real(torch.conj(W) * (-1j) * W * t * 1j * t), dim=1)

        b1 = torch.sum(torch.real(torch.conj(W_m0) * (S-W)), dim=1)
        b2 = torch.sum(torch.real(-torch.conj(W) * t * (S-W)), dim=1)
        b3 = torch.sum(torch.real((-1j) * torch.conj(W) * (S-W)), dim=1)
        b4 = torch.sum(torch.real((-1j) * torch.conj(W) * t * (S-W)), dim=1)

        # l2 regularization
        a11 += lambda_l2
        a22 += lambda_l2
        a33 += lambda_l2
        a44 += lambda_l2

        determ = torch.zeros(a11.size()).to('cuda')
        for idx in range(np.shape(cf)[0]):
            cdeterm = eval('a{}'.format(cf[idx][0]))
            for idx2 in range(1, np.shape(cf)[1]):
                cdeterm = cdeterm * eval('a{}'.format(cf[idx][idx2]))
            determ = determ + cf_sign[idx] * cdeterm

        dy1 = (a22*a33*a44 + a23*a34*a42 + a24*a32*a43 - a24*a33*a42 - a23*a32*a44 - a22*a34*a43) / determ * b1 \
            + (-a12*a33*a44 - a13*a34*a42 - a14*a32*a43 + a14*a33*a42 + a13*a32*a44 + a12*a34*a43) / determ * b2 \
            + (a12*a23*a44 + a13*a24*a42 + a14*a22*a43 - a14*a23*a42 - a13*a22*a44 - a12*a24*a43) / determ * b3 \
            + (-a12*a23*a34 - a13*a24*a32 - a14*a22*a33 + a14*a23*a32 + a13*a22*a34 + a12*a24*a33) / determ * b4

        dy2 = (-a21*a33*a44 - a23*a34*a41 - a24*a31*a43 + a24*a33*a41 + a23*a31*a44 + a21*a34*a43) / determ * b1 \
            + (a11*a33*a44 + a13*a34*a41 + a14*a31*a43 - a14*a33*a41 - a13*a31*a44 - a11*a34*a43) / determ * b2 \
            + (-a11*a23*a44 - a13*a24*a41 - a14*a21*a43 + a14*a24*a41 + a13*a21*a44 + a11*a24*a43) / determ * b3 \
            + (a11*a23*a34 + a13*a24*a31 + a14*a21*a33 - a14*a23*a31 - a13*a21*a34 - a11*a24*a33) / determ * b4    
        
        dy3 = (a21*a32*a44+a22*a34*a41+a24*a31*a42-a24*a32*a41-a22*a31*a44-a21*a34*a42)/determ*b1 \
            + (-a11*a32*a44-a12*a34*a41-a14*a31*a42+a14*a32*a41+a12*a31*a44+a11*a34*a42)/determ*b2 \
            + (a11*a22*a44+a12*a24*a41+a14*a21*a42-a14*a22*a41-a12*a21*a44-a11*a24*a42)/determ*b3 \
            + (-a11*a22*a34-a12*a24*a31-a14*a21*a32+a14*a22*a31+a12*a21*a34+a11*a24*a32)/determ*b4

        dy4 = (-a21*a32*a43-a22*a33*a41-a23*a31*a42+a23*a32*a41+a22*a31*a43+a21*a33*a42)/determ*b1 \
            + (a11*a32*a43+a12*a33*a41+a13*a31*a42-a13*a32*a41-a12*a31*a43-a11*a33*a42)/determ*b2 \
            + (-a11*a22*a43-a12*a23*a41-a13*a21*a42+a13*a22*a41+a12*a21*a43+a11*a23*a42)/determ*b3 \
            + (a11*a22*a33+a12*a23*a31+a13*a21*a32-a13*a22*a31-a12*a21*a33-a11*a23*a32)/determ*b4    

        dy = torch.cat([dy1[None, :], dy2[None, :], dy3[None, :], dy4[None, :]], dim=0)    
        dy[torch.isnan(dy)] = 0
        dy[torch.isinf(dy)] = 0
        y += dy

        if modm0 == 1:
            ty = y[0, :]
            ty[ty < 0] = 0
            y[0, :] = ty
        
        if modR2s == 1:
            ty = y[1, :]
            ty[ty < 0] = 0
            y[1, :] = ty

        iteration += 1
        update = torch.abs(torch.sum(dy)/numvox)
        # print('Iter = {} | update = {}'.format(iteration, update))

    m0 = y[0, :].view(matrix_size)
    r2s = y[1, :].view(matrix_size)
    f0 = y[2, :].view(matrix_size)
    p = y[3, :].view(matrix_size)

    return [m0, r2s, f0, p]


def torch_exp1j(y):
    return torch.cos(y) + 1j * torch.sin(y)


def fit_T1_M0():
    return 0
    
