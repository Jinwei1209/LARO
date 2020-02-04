import torch
import torch.nn as nn
import torch.nn.functional as F


def ResBlock(
    input_dim, 
    filter_dim, 
    kernel_size=3,
    stride=1,
    padding=1, 
    use_norm=1,  # 0 for no, 1 for batchnorm, 2 for instant norm
    N=5,
    unc_map=False
):  
    layers = []

    layers.append(nn.Conv2d(input_dim, filter_dim, kernel_size, stride, padding))
    if use_norm == 1:
        layers.append(nn.BatchNorm2d(filter_dim))
    elif use_norm == 2:
        layers.append(nn.GroupNorm(filter_dim, filter_dim))
    layers.append(nn.ReLU(inplace=True))

    for i in range(N-1):
        layers.append(nn.Conv2d(filter_dim, filter_dim, kernel_size, stride, padding))
        if use_norm == 1:
            layers.append(nn.BatchNorm2d(filter_dim))
        elif use_norm == 2:
            layers.append(nn.GroupNorm(filter_dim, filter_dim))
        layers.append(nn.ReLU(inplace=True))
    if unc_map:
        layers.append(nn.Conv2d(filter_dim, input_dim+2, 1))
    else:
        layers.append(nn.Conv2d(filter_dim, input_dim, 1))

    return layers