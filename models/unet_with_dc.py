import torch.nn.functional as F
import torch.nn as nn
from models.unet import *
from models.dc_blocks import *
from torch.autograd import Variable
from utils.loss import *


class Unet_with_DC(nn.Module):


    def __init__(
        self,
        input_channels,
        output_channels,
        num_filters,
        lambda_dll2, # initializing lambda_dll2
        K=1
    ):
        super(Unet_with_DC, self).__init__()
        self.Unet_block = Unet(input_channels, output_channels, num_filters)
        self.K = K
        self.lambda_dll2 = nn.Parameter(torch.ones(1)*lambda_dll2, requires_grad=True)
        # self.lambda_dll2 = torch.tensor(lambda_dll2)


    def forward(self, x, csms, masks):

        device = x.get_device()
        x_start = x
        self.lambda_dll2 = self.lambda_dll2.to(device)
        A = Back_forward(csms, masks, self.lambda_dll2)
        for i in range(self.K):
            x_block = self.Unet_block(x)
            rhs = x_start + self.lambda_dll2*x_block
            dc_layer = DC_layer(A, rhs)
            x = dc_layer.CG_iter()

        return x