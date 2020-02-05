import torch
import torch.nn as nn
import numpy as np
from utils.data import *


def lossL1():
    return nn.L1Loss()


def loss_classificaiton():
    return nn.BCELoss()


def lossL2():
    return nn.MSELoss()

    

