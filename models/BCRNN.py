import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable, grad


class CRNNcell(nn.Module):
    """
    Convolutional RNN cell that evolves over time
    Parameters
    -----------------
    input: 4d tensor, shape (batch_size, channel, width, height)
    hidden: hidden states in temporal dimension, 4d tensor, shape (batch_size, hidden_size, width, height)
    Returns
    -----------------
    output: 4d tensor, shape (batch_size, hidden_size, width, height)
    """
    def __init__(self, input_size, hidden_size, kernel_size):
        super(CRNNcell, self).__init__()
        self.kernel_size = kernel_size
        self.i2h = nn.Conv2d(input_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        self.h2h = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        self.bn_i2h = nn.GroupNorm(hidden_size, hidden_size)
        self.bn_h2h = nn.GroupNorm(hidden_size, hidden_size)
        # add iteration hidden connection
        # self.ih2ih = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        self.relu = nn.ReLU(inplace=True)

    # def forward(self, input, hidden_iteration, hidden):
    def forward(self, input, hidden):
    # def forward(self, input):
        in_to_hid = self.i2h(input)
        hid_to_hid = self.h2h(hidden)
        in_to_hid = self.bn_i2h(in_to_hid)
        hid_to_hid = self.bn_h2h(hid_to_hid)
        # ih_to_ih = self.ih2ih(hidden_iteration)

        # hidden = self.relu(in_to_hid + hid_to_hid + ih_to_ih)
        hidden = self.relu(in_to_hid + hid_to_hid)
        # hidden = self.relu(in_to_hid)

        return hidden


class BCRNNlayer(nn.Module):
    """
    Bidirectional Convolutional RNN layer
    Parameters
    --------------------
    incomings: input: 5d tensor, [input_image] with shape (num_seqs, batch_size, channel, width, height)
               input_iteration: 5d tensor, [hidden states from previous iteration] with shape (n_seq, n_batch, hidden_size, width, height)
               test: True if in test mode, False if in train mode
    Returns
    --------------------
    output: 5d tensor, shape (n_seq, n_batch, hidden_size, width, height)
    """
    def __init__(self, input_size, hidden_size, kernel_size):
        super(BCRNNlayer, self).__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.input_size = input_size
        # self.CRNN_model1 = CRNNcell(self.input_size, self.hidden_size, self.kernel_size)
        # self.CRNN_model2 = CRNNcell(self.input_size, self.hidden_size, self.kernel_size)
        self.CRNN_model = CRNNcell(self.input_size, self.hidden_size, self.kernel_size)
    # def forward(self, input, input_iteration, test=False):
    def forward(self, input, test=False):
        nt, nb, nc, nx, ny = input.shape
        size_h = [nb, self.hidden_size, nx, ny]
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
            # hidden = self.CRNN_model(input[i], input_iteration[i], hidden)
            hidden = self.CRNN_model(input[i], hidden)
            # hidden = self.CRNN_model(input[i])
            output_f.append(hidden)
        output_f = torch.cat(output_f)

        # backward
        hidden = hid_init
        for i in range(nt):
            # hidden = self.CRNN_model(input[nt - i - 1], input_iteration[nt - i -1], hidden)
            hidden = self.CRNN_model(input[nt - i - 1], hidden)
            # hidden = self.CRNN_model(input[nt - i - 1])
            output_b.append(hidden)
        output_b = torch.cat(output_b[::-1])

        output = output_f + output_b
        # output = output_f

        if nb == 1:
            output = output.view(nt, 1, self.hidden_size, nx, ny)

        return output
