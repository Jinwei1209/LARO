import torch
import torch.nn.functional as F
import torch.nn as nn
from models.initialization import init_weights


def conv_block(
    input_dim,
    output_dim,
    kernel_size=3,
    stride=1,
    padding=1,
    activation=nn.ReLU(inplace=True),
    use_bn=True
):

    layers = [nn.Conv2d(
        input_dim,
        output_dim,
        kernel_size,
        stride,
        padding
    )]

    if use_bn:
        layers.append(nn.BatchNorm2d(output_dim))
    
    if activation:
        layers.append(activation)

    return nn.Sequential(*layers)


class Basic_D(nn.Module):

    def __init__(
        self,
        input_channels,
        output_channels,
        num_filters
    ):

        super(Basic_D, self).__init__()
        self.max_layers = len(num_filters)
        layers = nn.Sequential()

        layers.add_module(
            'initial_{0}-{1}'.format(input_channels+output_channels, num_filters[0]),
            conv_block(input_channels+output_channels, num_filters[0], 4, 2, 1, use_bn=False)
        )
        output_dim = num_filters[0]

        for idx in range(1, self.max_layers-1):
            input_dim = output_dim
            output_dim = num_filters[idx]
            layers.add_module(
                'pyramid_{0}-{1}'.format(input_dim, output_dim),
                conv_block(input_dim, output_dim, 4, 2, 1, )
            )

        input_dim = output_dim
        output_dim = num_filters[idx+1]
        layers.add_module(
            'last_{0}-{1}'.format(input_dim, output_dim),
            conv_block(input_dim, output_dim, 4, 1, 1, )
        )

        input_dim, output_dim = output_dim, 1
        layers.add_module(
            'output_{0}-{1}'.format(input_dim, output_dim),
            conv_block(input_dim, output_dim, 4, 1, 1, nn.Sigmoid(), False)
        )

        self.layers = layers
        self.layers.apply(init_weights)

    def forward(self, a, b):

        inputs = torch.cat((a, b), dim=1)
        outputs = self.layers(inputs)

        return outputs



