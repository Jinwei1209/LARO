"""
This file contains data consistency blocks in the model
"""
import torch
import torch.nn as nn
import numpy as np
from utils.data import *
from utils.loss import *
from utils.operators import *

def mlpy_in_cg(a, b):
    """
    multiply two 'complex' tensors (with the second dim = 2, representing real and imaginary parts)
    """
    # device = a.get_device()
    # out = torch.empty(a.shape).to(device)
    # out[:,0,...] = a[:,0,...]*b[:,0,...] - a[:,1,...]*b[:,1,...]
    # out[:,1,...] = a[:,0,...]*b[:,1,...] + a[:,1,...]*b[:,0,...]

    out1 = a[:,0:1,...]*b[:,0:1,...] - a[:,1:2,...]*b[:,1:2,...]
    out2 = a[:,0:1,...]*b[:,1:2,...] + a[:,1:2,...]*b[:,0:1,...]
    out = torch.cat([out1, out2], dim=1)
    return out

def conj_in_cg(a):
    """
    conjugate of a complex number (with the second dim = 2, representing real and imaginary parts)
    """
    # device = a.get_device()
    # out = torch.empty(a.shape).to(device)
    # out[:,0,...] = a[:,0,...]
    # out[:,1,...] = -a[:,1,...]

    out = torch.cat([a[:,0:1,...], -a[:,1:2,...]], dim=1)
    return out

def dvd_in_cg(a, b):
    """
        division between a and b (with the second dim = 2, representing real and imaginary parts)
    """
    denom0 = mlpy_in_cg(conj_in_cg(b), b)[:,0:1,...]
    denom = torch.cat([denom0, denom0], dim=1)
    out = mlpy_in_cg(conj_in_cg(b), a) / denom
    return out

# complex valued CG layer
class DC_layer():
    def __init__(self, A, rhs, flag_precond=0, precond=0, use_dll2=1):
        self.AtA = lambda z: A.AtA(z, use_dll2=use_dll2)
        self.rhs = rhs
        self.device = rhs.get_device()
        if flag_precond == 0:
            self.flag_precond = flag_precond
        else:
            self.flag_precond = flag_precond
            epsilon = torch.ones(precond.shape).to(self.device)*1e-4
            self.M_inv = mlpy_in_cg(precond, precond)  # precond: C^-1, M_inv = C^-TC^-1

    def CG_body(self, i, rTr, x, r, p):
        Ap = self.AtA(p)
        alpha = rTr / torch.sum(mlpy_in_cg(conj_in_cg(p), Ap))
        alpha = torch.ones(x.shape).to(self.device)*alpha
        alpha[:,1,...] = 0

        x = x + mlpy_in_cg(p, alpha)
        r = r - mlpy_in_cg(Ap, alpha)
        rTrNew = torch.sum(mlpy_in_cg(conj_in_cg(r), r))

        beta = rTrNew /  rTr
        beta = torch.ones(x.shape).to(self.device)*beta
        beta[:,1,...] = 0
        p = r + mlpy_in_cg(p, beta)
        return i+1, rTrNew, x, r, p

    def precond_CG_body(self, i, rTy, x, r, y, p):
        Ap = self.AtA(p)
        alpha = cplx_dvd(rTy, torch.sum(mlpy_in_cg(conj_in_cg(p), Ap), dim=(0, 2, 3)))
        alpha = alpha.repeat(x.shape[0], x.shape[2], x.shape[3], 1).permute(0, 3, 1, 2)

        x = x + mlpy_in_cg(p, alpha)
        r = r + mlpy_in_cg(Ap, alpha)
        y = mlpy_in_cg(self.M_inv, r)

        rTyNew = torch.sum(mlpy_in_cg(conj_in_cg(r), y), dim=(0, 2, 3))
        beta = cplx_dvd(rTyNew, rTy)
        beta = beta.repeat(x.shape[0], x.shape[2], x.shape[3], 1).permute(0, 3, 1, 2)
        p = -y + mlpy_in_cg(beta, p)
        return i+1, rTyNew, x, r, y, p

    def while_cond(self, i, rTr, max_iter=10):
        return (i<max_iter) and (rTr>1e-10)

    def CG_iter(self, max_iter=10):
        x = torch.zeros(self.rhs.shape).to(self.device)

        if self.flag_precond == 0:
            i, r, p = 0, self.rhs, self.rhs
            rTr = torch.sum(mlpy_in_cg(conj_in_cg(r), r))
            while self.while_cond(i, rTr, max_iter):
                i, rTr, x, r, p = self.CG_body(i, rTr, x, r, p)
                # print('i = {0}, rTr = {1}'.format(i, rTr))
        elif self.flag_precond == 1:
            i, r = 0, -self.rhs
            y = mlpy_in_cg(self.M_inv, r)
            p = -y
            rTy = torch.sum(mlpy_in_cg(conj_in_cg(r), y), dim=(0, 2, 3))  #(2,) tensor
            rTr = torch.sum(mlpy_in_cg(conj_in_cg(r), r))
            while self.while_cond(i, rTr, max_iter):
                i, rTy, x, r, y, p = self.precond_CG_body(i, rTy, x, r, y, p)
                rTr = torch.sum(mlpy_in_cg(conj_in_cg(r), r))
                # print('i = {0}, rTr = {1}'.format(i, rTr))
        return x

# real valued CG layer for undersampled field estimation problem
class DC_layer_real():
    def __init__(self, A, rhs, flag, use_dll2=1, lambda_dll2=1):
        self.AtA = lambda z: A.AtA(z, flag=flag, use_dll2=use_dll2, lambda_dll2=lambda_dll2)
        self.rhs = rhs
        self.device = rhs.get_device()

    def CG_body(self, i, rTr, x, r, p):
        Ap = self.AtA(p)
        alpha = rTr / torch.sum(p*Ap)
        x = x + p*alpha
        r = r - Ap*alpha
        rTrNew = torch.sum(r**2)
        beta = rTrNew /  rTr
        p = r + p*beta
        return i+1, rTrNew, x, r, p

    def while_cond(self, i, rTr, max_iter=10):
        return (i<max_iter) and (rTr>1e-10)

    def CG_iter(self, max_iter=10):
        x = torch.zeros(self.rhs.shape).to(self.device)
        i, r, p = 0, self.rhs, self.rhs
        rTr = torch.sum(r**2)
        while self.while_cond(i, rTr, max_iter):
            i, rTr, x, r, p = self.CG_body(i, rTr, x, r, p)
            # print('i = {0}, rTr = {1}'.format(i, rTr))
        return x

# CG layer for (necho, nrow, ncol) data
class DC_layer_multiEcho():
    def __init__(self, A, rhs, echo_cat=1, necho=10,
                 flag_precond=0, precond=0, use_dll2=1):
        self.AtA = lambda z: A.AtA(z, use_dll2=use_dll2)
        self.echo_cat = echo_cat
        self.necho = necho
        self.flag_precond = flag_precond
        if self.echo_cat:
            self.rhs = torch_channel_deconcate(rhs) # (batch, 2, echo, row, col)
        else:
            self.rhs = rhs
        if self.flag_precond:
            # precond: C^-1, M_inv = C^-TC^-1, size: (batch, 2, echo, row, col)
            self.M_inv = mlpy_in_cg(conj_in_cg(precond), precond)
        self.device = rhs.get_device()

    def CG_body(self, i, rTr, x, r, p):
        if self.echo_cat:
            Ap = self.AtA(torch_channel_concate(p, self.necho)) # (batch, 2*echo, row, col)
            Ap = torch_channel_deconcate(Ap) # (batch, 2, echo, row, col)
        else:
            Ap = self.AtA(p)
        alpha = rTr / torch.sum(mlpy_in_cg(conj_in_cg(p), Ap))

        x = x + p * alpha
        r = r - Ap * alpha
        rTrNew = torch.sum(mlpy_in_cg(conj_in_cg(r), r))

        beta = rTrNew /  rTr
        p = r + p * beta
        return i+1, rTrNew, x, r, p

    def precond_CG_body(self, i, rTy, x, r, y, p):
        if self.echo_cat:
            Ap = self.AtA(torch_channel_concate(p, self.necho)) # (batch, 2*echo, row, col)
            Ap = torch_channel_deconcate(Ap) # (batch, 2, echo, row, col)
        else:
            Ap = self.AtA(p)
        alpha = cplx_dvd(rTy, torch.sum(mlpy_in_cg(conj_in_cg(p), Ap), dim=(0, 2, 3, 4)))
        alpha = alpha[None, :, None, None, None]

        x = x + mlpy_in_cg(p, alpha)
        r = r + mlpy_in_cg(Ap, alpha)
        y = mlpy_in_cg(self.M_inv, r)

        rTyNew = torch.sum(mlpy_in_cg(conj_in_cg(r), y), dim=(0, 2, 3, 4))
        beta = cplx_dvd(rTyNew, rTy)
        beta = beta[None, :, None, None, None]
        p = -y + mlpy_in_cg(beta, p)
        return i+1, rTyNew, x, r, y, p

    def while_cond(self, i, rTr, max_iter=10):
        return (i<max_iter) and (rTr>1e-10)

    def CG_iter(self, max_iter=10):
        x = torch.zeros(self.rhs.shape).to(self.device)

        if self.flag_precond == 0:
            i, r, p = 0, self.rhs, self.rhs
            rTr = torch.sum(mlpy_in_cg(conj_in_cg(r), r))
            while self.while_cond(i, rTr, max_iter):
                i, rTr, x, r, p = self.CG_body(i, rTr, x, r, p)
            return x
        elif self.flag_precond == 1:
            i, r = 0, -self.rhs
            y = mlpy_in_cg(self.M_inv, r)
            p = -y
            rTy = torch.sum(mlpy_in_cg(conj_in_cg(r), y), dim=(0, 2, 3, 4))  #(2,) tensor
            rTr = torch.sum(mlpy_in_cg(conj_in_cg(r), r))
            while self.while_cond(i, rTr, max_iter):
                i, rTy, x, r, y, p = self.precond_CG_body(i, rTy, x, r, y, p)
                rTr = torch.sum(mlpy_in_cg(conj_in_cg(r), r))
                # print('i = {0}, rTr = {1}'.format(i, rTr))
            return x
        


