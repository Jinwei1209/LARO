import torch.nn.functional as F
import torch.nn as nn
from models.unet_blocks import *


class Unet(nn.Module):

    def __init__(
        self,
        input_channels,
        output_channels,
        num_filters,
        use_bn=1,
        use_deconv=1,
        skip_connect=False,
        slim=False,
        convFT=False
    ):

        super(Unet, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_filters = num_filters
        self.downsampling_path = nn.ModuleList()
        self.upsampling_path = nn.ModuleList()
        self.skip_connect = skip_connect

        for i in range(len(self.num_filters)):

            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = self.num_filters[i]

            if i == 0:
                pool = False
            else:
                pool = True

            self.downsampling_path.append(DownConvBlock(input_dim, output_dim, use_bn=use_bn, pool=pool, slim=slim, convFT=convFT))

        for i in range(len(self.num_filters)-2, -1, -1):

            input_dim = self.num_filters[i+1]
            output_dim = self.num_filters[i]

            self.upsampling_path.append(UpConvBlock(input_dim, output_dim, use_bn=use_bn, use_deconv=use_deconv, slim=slim, convFT=convFT))
        
        self.last_layer = nn.Conv2d(output_dim, self.output_channels, kernel_size=3, padding=1)


    def forward(self, x):

        blocks = []
        x_start = x
        for idx, down in enumerate(self.downsampling_path):
            x = down(x)
            
            if idx != len(self.downsampling_path)-1:
                blocks.append(x)

        for idx, up in enumerate(self.upsampling_path):
            x = up(x, blocks[-idx-1])

        del blocks

        x = self.last_layer(x)

        if self.skip_connect:
            x = x + x_start

        return x


class UnetWithInputFeatures(nn.Module):

    def __init__(
        self,
        input_channels,
        output_channels,
        num_filters,
        use_bn=1,
        use_deconv=1,
        skip_connect=False,
        slim=False,
        convFT=False
    ):

        super(UnetWithInputFeatures, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_filters = num_filters
        self.downsampling_path = nn.ModuleList()
        self.upsampling_path = nn.ModuleList()
        self.skip_connect = skip_connect

        for i in range(len(self.num_filters)):
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = self.num_filters[i]
            if i == 0:
                pool = False
                self.downsampling_path.append(DownConvBlock(input_dim, output_dim, use_bn=use_bn, pool=pool, slim=slim, convFT=convFT))
            else:
                pool = True
                self.downsampling_path.append(DownConvBlock2(input_dim, output_dim, use_bn=use_bn, pool=pool, slim=slim, convFT=convFT))
        
        for i in range(len(self.num_filters)-2, -1, -1):
            input_dim = self.num_filters[i+1]
            output_dim = self.num_filters[i]
            self.upsampling_path.append(UpConvBlock(input_dim, output_dim, use_bn=use_bn, use_deconv=use_deconv, slim=slim, convFT=convFT))
        
        self.last_layer = nn.Conv2d(output_dim, self.output_channels, kernel_size=3, padding=1)


    def forward(self, features):
        blocks = []
        x = features[0]
        x_start = x
        for idx, down in enumerate(self.downsampling_path):
            if idx == 0:
                x = down(x)
            else:
                x = down(x, features[idx])
            
            if idx != len(self.downsampling_path)-1:
                blocks.append(x)

        for idx, up in enumerate(self.upsampling_path):
            x = up(x, blocks[-idx-1])

        del blocks

        x = self.last_layer(x)

        if self.skip_connect:
            x = x + x_start

        return x

    
def multi_unet(
    input_channels,
    output_channels,
    num_filters,
    use_bn=1,
    use_deconv=1,
    skip_connect=False,
    K=1
):
    unet_list = nn.ModuleList()
    for i in range(K):
        unet_list.append(Unet(
            input_channels,
            output_channels,
            num_filters,
            use_bn,
            use_deconv,
            skip_connect
        ))
    return unet_list
