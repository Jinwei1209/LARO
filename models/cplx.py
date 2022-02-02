import torch
import torch.nn as nn
from models.initialization import init_weights

class ComplexConv2d(nn.Module):
    """
    Convolutional layer with complex and complex convolution:
    (X+iY)*(a+ib) = (X*a-Y*b) + i(Y*a+X*b)
    -----------------
    input: 5d tensor, shape (batch_size, 2, input_size, width, height) with "2" representing "real&imag"
    -----------------
    output: 5d tensor, shape (batch_size, 2, hidden_size, width, height)
    """
    def __init__(self, input_size, hidden_size, kernel_size):
        super(ComplexConv2d, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.i2h_real = nn.Conv2d(input_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        self.i2h_imag = nn.Conv2d(input_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        self.i2h_real.apply(init_weights)
        self.i2h_imag.apply(init_weights)

    def forward(self, input):
        real = self.i2h_real(input[:, 0, ...]) - self.i2h_imag(input[:, 1, ...])
        imag = self.i2h_imag(input[:, 0, ...]) + self.i2h_real(input[:, 1, ...])
        return torch.cat([real[:, None, ...], imag[:, None, ...]], 1)


class ComplexConv2dTrans(nn.Module):
    """
    Transpose convolutional layer for upsampling with complex and complex convolution:
    (X+iY)*(a+ib) = (X*a-Y*b) + i(Y*a+X*b)
    -----------------
    input: 5d tensor, shape (batch_size, 2, input_size, width, height)
    -----------------
    output: 5d tensor, shape (batch_size, 2, hidden_size, width*2, height*2)
    """
    def __init__(self, input_size, hidden_size, kernel_size):
        super(ComplexConv2dTrans, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.i2h_real = nn.ConvTranspose2d(input_size, hidden_size, kernel_size=kernel_size, stride=2)
        self.i2h_imag = nn.ConvTranspose2d(input_size, hidden_size, kernel_size=kernel_size, stride=2)
        self.i2h_real.apply(init_weights)
        self.i2h_imag.apply(init_weights)
    
    def forward(self, input):
        real = self.i2h_real(input[:, 0, ...]) - self.i2h_imag(input[:, 1, ...])
        imag = self.i2h_imag(input[:, 0, ...]) + self.i2h_real(input[:, 1, ...])
        return torch.cat([real[:, None, ...], imag[:, None, ...]], 1)


class ComplexMaxPool2d(nn.Module):
    """
    Complex 2d max pooling with max abs(X+iY)
    input: 5d tensor, shape (batch_size, 2, input_size, width, height)
    -----------------
    output: 5d tensor, shape (batch_size, 2, input_size, width_out, height_out)
    """
    def __init__(self, kernel_size, stride, padding):
        super(ComplexMaxPool2d, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, return_indices=True)
        
    def forward(self, input):    
        _, indices = self.pool(input[:, 0, ...]**2 + input[:, 1, ...]**2)
        flat = input.flatten(3, -1)
        ix = indices.flatten(2 -1)
        output = torch.gather(flat, -1, ix)
        return output.reshape(input.shape[:3], indices.shape[2:])