import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable, grad
from models.cplx import *
from models.BCRNN import *


class ComplexCRNNcell(nn.Module):
    """
    Complex Convolutional RNN cell that evolves over time
    Parameters
    -----------------
    input: 5d tensor, shape (batch_size, channel, 1, width, height)  (channel are real&imag)
    hidden: hidden states in temporal dimension, 5d tensor, shape (batch_size, channel, hidden_size, width, height)
    Returns
    -----------------
    output: 5d tensor, shape (batch_size, channel, hidden_size, width, height)
    """
    def __init__(self, input_size, hidden_size, kernel_size, flag_convFT=0, flag_bn=1, flag_hidden=1):
        super(ComplexCRNNcell, self).__init__()
        self.kernel_size = kernel_size
        self.flag_bn = flag_bn
        self.flag_hidden = flag_hidden
        if flag_convFT:
            self.i2h = Conv2dFT(input_size, hidden_size, kernel_size)
            self.h2h = Conv2dFT(hidden_size, hidden_size, kernel_size)
        else:
            self.i2h = ComplexConv2d(input_size, hidden_size, kernel_size)
            self.h2h = ComplexConv2d(hidden_size, hidden_size, kernel_size)
        if self.flag_bn == 2:
            self.bn_i2h = nn.GroupNorm(hidden_size*2, hidden_size*2)
            self.bn_h2h = nn.GroupNorm(hidden_size*2, hidden_size*2)
        elif self.flag_bn == 3:
            self.bn_i2h = ComplexInstanceNorm2d(hidden_size)
            self.bn_h2h = ComplexInstanceNorm2d(hidden_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input, hidden):
        in_to_hid = self.i2h(input)
        (N, _, C, W, H) = in_to_hid.size()
        if self.flag_hidden:
            hid_to_hid = self.h2h(hidden)
        if self.flag_bn == 2:
            in_to_hid = self.bn_i2h(in_to_hid.view(N, 2*C, W, H)).view(N, 2, C, W, H)
            if self.flag_hidden:
                hid_to_hid = self.bn_h2h(hid_to_hid.view(N, 2*C, W, H)).view(N, 2, C, W, H)
        elif self.flag_bn == 3:
            in_to_hid = self.bn_i2h(in_to_hid)
            if self.flag_hidden:
                hid_to_hid = self.bn_h2h(hid_to_hid)

        if self.flag_hidden:
            hidden = self.relu(in_to_hid + hid_to_hid)
        else:
            hidden = self.relu(in_to_hid)

        return hidden


class ComplexBCRNNlayer(nn.Module):
    """
    Bidirectional Complex Convolutional RNN layer
    Parameters
    --------------------
    incomings: input: 5d tensor, [input_image] with shape (num_seqs, batch_size, channel, width, height)
                      channel are real&imag
               test: True if in test mode, False if in train mode
    Returns
    --------------------
    output: 5d tensor, shape (n_seq, channel, hidden_size, width, height)
    """
    def __init__(self, input_size, hidden_size, kernel_size, flag_convFT=0, flag_bn=1, flag_hidden=1):
        super(ComplexBCRNNlayer, self).__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.flag_convFT = flag_convFT
        self.flag_bn = flag_bn
        self.flag_hidden = flag_hidden
        self.CRNN_model = ComplexCRNNcell(self.input_size, self.hidden_size, self.kernel_size, self.flag_convFT, self.flag_bn, self.flag_hidden)
        
    def forward(self, input, test=False):
        nt, nb, nc, nx, ny = input.shape
        size_h = [nb, nc, self.hidden_size, nx, ny]
        if test:
            with torch.no_grad():
                hid_init = Variable(torch.zeros(size_h)).cuda()
        else:
            hid_init = Variable(torch.zeros(size_h)).cuda()

        output_f = []
        output_b = []
        # forward
        hidden = hid_init
        for i in range(nt):
            hidden = self.CRNN_model(input[i, :, :, None, :, :], hidden)
            output_f.append(hidden)
        output_f = torch.cat(output_f)
        # backward
        hidden = hid_init
        for i in range(nt):
            hidden = self.CRNN_model(input[nt - i - 1, :, :, None, :, :], hidden)
            output_b.append(hidden)
        output_b = torch.cat(output_b[::-1])

        output = output_f + output_b

        return output