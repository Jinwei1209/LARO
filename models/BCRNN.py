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
    def __init__(self, input_size, hidden_size, kernel_size, flag_convFT=0, flag_bn=1, flag_hidden=1):
        super(CRNNcell, self).__init__()
        self.kernel_size = kernel_size
        self.flag_bn = flag_bn
        self.flag_hidden = flag_hidden
        if flag_convFT:
            self.i2h = Conv2dFT(input_size, hidden_size, kernel_size)
            self.h2h = Conv2dFT(hidden_size, hidden_size, kernel_size)
        else:
            self.i2h = nn.Conv2d(input_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
            self.h2h = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        if self.flag_bn:
            self.bn_i2h = nn.GroupNorm(hidden_size, hidden_size)
            self.bn_h2h = nn.GroupNorm(hidden_size, hidden_size)
        # add iteration hidden connection
        # self.ih2ih = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        self.relu = nn.ReLU(inplace=True)

    # def forward(self, input, hidden_iteration, hidden):
    def forward(self, input, hidden):
    # def forward(self, input):
        in_to_hid = self.i2h(input)
        if self.flag_hidden:
            hid_to_hid = self.h2h(hidden)
        if self.flag_bn:
            in_to_hid = self.bn_i2h(in_to_hid)
            if self.flag_hidden:
                hid_to_hid = self.bn_h2h(hid_to_hid)
        # ih_to_ih = self.ih2ih(hidden_iteration)

        # hidden = self.relu(in_to_hid + hid_to_hid + ih_to_ih)
        if self.flag_hidden:
            hidden = self.relu(in_to_hid + hid_to_hid)
        else:
            hidden = self.relu(in_to_hid)

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
    def __init__(self, input_size, hidden_size, kernel_size, flag_convFT=0, flag_bn=1, flag_hidden=1):
        super(BCRNNlayer, self).__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.flag_convFT = flag_convFT
        self.flag_bn = flag_bn
        self.flag_hidden = flag_hidden
        # self.CRNN_model1 = CRNNcell(self.input_size, self.hidden_size, self.kernel_size)
        # self.CRNN_model2 = CRNNcell(self.input_size, self.hidden_size, self.kernel_size)
        self.CRNN_model = CRNNcell(self.input_size, self.hidden_size, self.kernel_size, self.flag_convFT, self.flag_bn, self.flag_hidden)
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


class MultiLevelBCRNNlayer(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, flag_convFT=0):
        super(MultiLevelBCRNNlayer, self).__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.flag_convFT = flag_convFT
        self.BCRNNlayer1 = BCRNNlayer(self.input_size, self.hidden_size, self.kernel_size, self.flag_convFT)
        self.BCRNNlayer2 = BCRNNlayer(self.input_size, self.hidden_size*2, self.kernel_size, self.flag_convFT)
        self.BCRNNlayer3 = BCRNNlayer(self.input_size, self.hidden_size*4, self.kernel_size, self.flag_convFT)
        self.BCRNNlayer4 = BCRNNlayer(self.input_size, self.hidden_size*8, self.kernel_size, self.flag_convFT)
        self.downsampling = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)
    def forward(self, input, test=False):
        outputs = []
        nt, nb, nc, nx, ny = input.shape
        output1 = self.BCRNNlayer1(input, test)
        output1 = output1.view(-1, self.hidden_size, nx, ny)
        outputs.append(output1)
        
        input = self.downsampling(input)
        _, _, _, nx, ny = input.shape
        output2 = self.BCRNNlayer2(input, test)
        output2 = output2.view(-1, self.hidden_size*2, nx, ny)
        outputs.append(output2)

        input = self.downsampling(input)
        _, _, _, nx, ny = input.shape
        output3 = self.BCRNNlayer3(input, test)
        output3 = output3.view(-1, self.hidden_size*4, nx, ny)
        outputs.append(output3)

        input = self.downsampling(input)
        _, _, _, nx, ny = input.shape
        output4 = self.BCRNNlayer4(input, test)
        output4 = output4.view(-1, self.hidden_size*8, nx, ny)
        outputs.append(output4)
        return outputs


class Conv2dFT(nn.Module):
    """
    Convolutional layer with half image domain and half k-space domain feature generation
    -----------------
    input: 4d tensor, shape (batch_size, channel, width, height)
    -----------------
    output: 4d tensor, shape (batch_size, hidden_size, width, height)
    """
    def __init__(self, input_size, hidden_size, kernel_size):
        super(Conv2dFT, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.i2h_image_domain = nn.Conv2d(input_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        self.i2h_fourier_domain = nn.Conv2d(input_size, hidden_size, kernel_size, padding=self.kernel_size // 2, padding_mode='reflect')
        self.bn_i2h = nn.GroupNorm(hidden_size, hidden_size)

    def fftshift(self, image):
        '''
            channel last fftshift
        '''
        (_, _, nrows, ncols, _) = image.size()
        image = torch.cat((image[:, :, nrows//2:nrows, ...], image[:, :, 0:nrows//2, ...]), dim=2)
        image = torch.cat((image[:, :, :, ncols//2:ncols, ...], image[:, :, :, 0:ncols//2, ...]), dim=3)
        return image

    def forward(self, input):
        in_to_hid_image = self.i2h_image_domain(input)  # (batch, hidden/2, width, height) on image domain
        
        # (batch, input_size, width, height) to (batch, input_size/2, width, height, 2)
        input = torch.cat([input[:, :self.input_size//2, :, :, None], input[:, self.input_size//2:, :, :, None]], dim=-1)
        input_kspace = self.fftshift(torch.fft(input, 2))
        input_kspace = torch.cat([input_kspace[..., 0], input_kspace[..., 1]], dim=1)  # (batch, input_size, width, height)

        # Fourier domain convolution
        in_to_hid_fourier = self.i2h_fourier_domain(input_kspace)  # (batch, hidden/2, width, height)
        
        # (batch, hidden/2, width, height) to (batch, hidden/4, width, height, 2)
        in_to_hid_fourier = torch.cat([in_to_hid_fourier[:, :self.hidden_size//2, :, :, None], in_to_hid_fourier[:, self.hidden_size//2:, :, :, None]], dim=-1)
        in_to_hid_fourier = torch.ifft(self.fftshift(in_to_hid_fourier), 2)
        in_to_hid_fourier = torch.cat([in_to_hid_fourier[..., 0], in_to_hid_fourier[..., 1]], dim=1)  # (batch, hidden/2, width, height)
        
        # concatenate image and Fourier domain features
        # hidden = torch.cat([in_to_hid_image, in_to_hid_fourier], dim=1)  # (batch, hidden, width, height)
        hidden = in_to_hid_image + in_to_hid_fourier
        
        return hidden

