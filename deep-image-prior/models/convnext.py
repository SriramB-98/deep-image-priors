import torch
import torch.nn as nn
from numpy.random import normal
from numpy.linalg import svd
from math import sqrt
import torch.nn.init
from .common import *

from transformers import ConvNextConfig, ConvNextModel

class ConvNext(ConvNextModel):
     def __init__(self, num_input_channels, num_output_channels, num_blocks, num_channels, need_residual=True, act_fun='LeakyReLU', need_sigmoid=True, norm_layer=nn.BatchNorm2d, pad='reflection'):
        config = ConvNextConfig( num_channels, patch_size, num_stages, hidden_sizes = Nonedepths = Nonehidden_act = 'gelu'initializer_range = 0.02layer_norm_eps = 1e-12is_encoder_decoder = Falselayer_scale_init_value = 1e-06drop_path_rate = 0.0image_size = 224**kwargs )
        super().__init__(config)

        if need_residual:
            s = ResidualSequential
        else:
            s = nn.Sequential

        stride = 1
        # First layers
        layers = [
            # nn.ReplicationPad2d(num_blocks * 2 * stride + 3),
            conv(num_input_channels, num_channels, 3, stride=1, bias=True, pad=pad),
            act(act_fun)
        ]
        # Residual blocks
        # layers_residual = []
        for i in range(num_blocks):
            layers += [s(*get_block(num_channels, norm_layer, act_fun))]
       
        layers += [
            nn.Conv2d(num_channels, num_channels, 3, 1, 1),
            norm_layer(num_channels, affine=True)
        ]

        # if need_residual:
        #     layers += [ResidualSequential(*layers_residual)]
        # else:
        #     layers += [Sequential(*layers_residual)]

        # if factor >= 2: 
        #     # Do upsampling if needed
        #     layers += [
        #         nn.Conv2d(num_channels, num_channels *
        #                   factor ** 2, 3, 1),
        #         nn.PixelShuffle(factor),
        #         act(act_fun)
        #     ]
        layers += [
            conv(num_channels, num_output_channels, 3, 1, bias=True, pad=pad),
            nn.Sigmoid()
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)

    def eval(self):
        self.model.eval()
