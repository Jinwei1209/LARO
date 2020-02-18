"""
This file contains data consistency blocks in model
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
    device = a.get_device()
    out = torch.empty(a.shape).to(device)
    out[:,0,...] = a[:,0,...]*b[:,0,...] - a[:,1,...]*b[:,1,...]
    out[:,1,...] = a[:,0,...]*b[:,1,...] + a[:,1,...]*b[:,0,...]
    return out


def conj_in_cg(a):
    """
    conjugate of a complex number (with the second dim = 2, representing real and imaginary parts)
    """
    device = a.get_device()
    out = torch.empty(a.shape).to(device)
    out[:,0,...] = a[:,0,...]
    out[:,1,...] = -a[:,1,...]
    return out


class DC_layer():


    def __init__(self, A, rhs, use_dll2=1):
        self.AtA = lambda z: A.AtA(z, use_dll2=use_dll2)
        self.rhs = rhs
        self.device = rhs.get_device()


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


    def while_cond(self, i, rTr, max_iter=10):
        return (i<max_iter) and (rTr>1e-10)


    def CG_iter(self, max_iter=10):
      
        x = torch.zeros(self.rhs.shape).to(self.device)
        i, r, p = 0, self.rhs, self.rhs
        rTr = torch.sum(mlpy_in_cg(conj_in_cg(r), r))

        while self.while_cond(i, rTr, max_iter):
            i, rTr, x, r, p = self.CG_body(i, rTr, x, r, p)
            # print('i = {0}, rTr = {1}'.format(i, rTr))

        return x
        


