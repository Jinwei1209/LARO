import torch
import torch.nn as nn
import torch.nn.functional as F
from models.initialization import *


class Prob_Mask(nn.Module):

    def __init__(
        self,
        ncoil=1,
        nrow=256,
        ncol=184,
        slope=0.25
    ):
        super(Prob_Mask, self).__init__()
        self.slope = slope
        self.weight_parameters = nn.Parameter(torch.zeros(ncoil, nrow, ncol), requires_grad=True)

    def forward(self, x):

        device = x.get_device()
        self.weight_parameters = self.weight_parameters.to(device)
        self.prob_mask = 1/(1+torch.exp(-self.slope*self.weight_parameters))
        return self.prob_mask

