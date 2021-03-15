import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable, grad


class CLSTMcell(nn.Module):
    """
    Convolutional LSTM cell that evolves over time
    Parameters
    -----------------
    input: 4d tensor, shape (batch_size, channel, width, height)
    hidden: hidden states in temporal dimension, 4d tensor, shape (batch_size, hidden_size, width, height)
    cell_state: cell states in temporal dimension, 4d tensor, shape (batch_size, cell_size, width, height)
    Returns
    -----------------
    output: 4d tensor, shape (batch_size, hidden_size, width, height)
    """
    def __init__(self, input_size, hidden_size, kernel_size):
        super(CLSTMcell, self).__init__()
        self.kernel_size = kernel_size
        self.conv0 = nn.Sequential(
            nn.Conv2d(input_size, hidden_size, kernel_size, padding=self.kernel_size // 2),
            nn.ReLU()
        )
        self.conv_i = nn.Sequential(
            nn.Conv2d(hidden_size + hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2),
            nn.Sigmoid()
        )
        self.conv_f = nn.Sequential(
            nn.Conv2d(hidden_size + hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(hidden_size + hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2),
            nn.Tanh()
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(hidden_size + hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2),
            nn.Sigmoid()
        )

    def forward(self, input, hidden, cell):
        x = self.conv0(input)
        x = torch.cat((x, hidden), 1)
        i = self.conv_i(x)
        f = self.conv_f(x)
        g = self.conv_g(x)
        o = self.conv_o(x)
        cell = f * cell + i * g
        hidden = o * torch.tanh(cell)

        return [hidden, cell]


class BCLSTMlayer(nn.Module):
    """
    Bidirectional Convolutional LSTM layer
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
        super(BCLSTMlayer, self).__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.CLSTM_model = CLSTMcell(self.input_size, self.hidden_size, self.kernel_size)

    def forward(self, input, test=False):
        nt, nb, nc, nx, ny = input.shape
        size_h = [nb, self.hidden_size, nx, ny]
        if test:
            with torch.no_grad():
                hid_init = Variable(torch.zeros(size_h)).cuda()
        else:
            hid_init = Variable(torch.zeros(size_h)).cuda()
            cell_init = Variable(torch.zeros(size_h)).cuda()

        output_f = []
        output_b = []
        # forward
        hidden = hid_init
        cell = cell_init
        for i in range(nt):
            [hidden, cell] = self.CLSTM_model(input[i], hidden, cell)
            output_f.append(hidden)
        output_f = torch.cat(output_f)

        # backward
        hidden = hid_init
        cell = cell_init
        for i in range(nt):
            [hidden, cell] = self.CLSTM_model(input[nt - i - 1], hidden, cell)
            output_b.append(hidden)
        output_b = torch.cat(output_b[::-1])

        output = output_f + output_b

        if nb == 1:
            output = output.view(nt, 1, self.hidden_size, nx, ny)

        return output
