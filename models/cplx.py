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
        ix = indices.flatten(2, -1)
        output_real = torch.gather(flat[:, 0, ...], dim=-1, index=ix)
        output_imag = torch.gather(flat[:, 1, ...], dim=-1, index=ix)
        output = torch.cat([output_real[:, None, ...], output_imag[:, None, ...]], 1)
        return output.reshape(*input.shape[:3], *indices.shape[2:])


class _ComplexInstanceNorm(nn.Module):
    """
    Instance normalizaiton base class
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(_ComplexInstanceNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features, 3))
            self.bias = nn.Parameter(torch.Tensor(num_features, 2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_parameter('running_mean', None)
        self.register_parameter('running_covar', None)
        self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.constant_(self.weight[:,:2], 1.4142135623730951)
            nn.init.zeros_(self.weight[:,2])
            nn.init.zeros_(self.bias)


class ComplexInstanceNorm2d(_ComplexInstanceNorm):
    """
    2D intance normalization class.
    input: 5d tensor, shape (batch_size, 2, input_size, width, height)
    output: 5d tensor, shape (batch_size, 2, input_size, width, height)
    """
    def forward(self, input):
        # calculate mean of real and imaginary part
        mean_r = input[:, 0:1, ...].mean([3, 4])  # (batch, 1, input_size)
        mean_i = input[:, 1:2, ...].mean([3, 4])  # (batch, 1, input_size)
        mean = torch.cat([mean_r, mean_i], 1)  # (batch, 2, input_size)

        input = input - mean[..., None, None]

        # Elements of the covariance matrix (biased for train)
        n = input.numel() / (input.size(2)*input.size(1)*input.size(0))
        Crr = 1./n*input[:, 0:1, ...].pow(2).sum(dim=[3, 4])+self.eps  # (batch, 1, input_size)
        Cii = 1./n*input[:, 1:2, ...].pow(2).sum(dim=[3, 4])+self.eps  # (batch, 1, input_size)
        Cri = (input[:, 0:1, ...].mul(input[:, 1:2, ...])).mean(dim=[3, 4])  # (batch, 1, input_size)

        # calculate the inverse of the covariance matrix
        det = Crr*Cii-Cri.pow(2)
        Crr_new = Cii / det
        Cii_new = Crr / det
        Cri_new = -Cri / det

        # calculate the square root of the covariance matrix
        s = torch.sqrt(Crr_new*Cii_new-Cri_new.pow(2))
        t = torch.sqrt(Cii_new+Crr_new + 2 * s)
        inverse_t = 1.0 / t
        Rrr = (Crr_new + s) * inverse_t  # (batch, 1, input_size)
        Rii = (Cii_new + s) * inverse_t  # (batch, 1, input_size)
        Rri = Cri_new * inverse_t  # (batch, 1, input_size)

        input_real = Rrr[..., None,None]*input[:, 0:1, ...] + Rri[..., None,None]*input[:, 1:2, ...]
        input_imag = Rii[..., None,None]*input[:, 1:2, ...] + Rri[..., None,None]*input[:, 0:1, ...]
        input = torch.cat([input_real, input_imag], 1)  # (batch, 2, input_size, width, height)

        if self.affine:
            input_real = self.weight[None,:, 0,None,None]*input[:, 0, ...]+self.weight[None,:, 2,None,None]*input[:, 1, ...]+\
                    self.bias[None,:, 0,None,None]
            input_imag = self.weight[None,:, 2,None,None]*input[:, 0, ...]+self.weight[None,:, 1,None,None]*input[:, 1, ...]+\
                    self.bias[None,:, 1,None,None]
            input = torch.cat([input_real[:, None, ...], input_imag[:, None, ...]], 1)  # (batch, 2, input_size, width, height)

        return input 