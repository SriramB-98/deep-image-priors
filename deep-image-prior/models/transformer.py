import torch
import torch.nn as nn
from numpy.random import normal
from numpy.linalg import svd
from math import sqrt
import torch.nn.init
from .common import *
from torchvision.models.vision_transformer import VisionTransformer
from torchvision.transforms import Resize

class Transformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.numpatches = self.image_size//self.patch_size
        self.c = 3#input_depth
        self.upsample = torch.nn.Linear(self.hidden_dim, self.c*self.patch_size*self.patch_size)
        self.resize = Resize((self.image_size, self.image_size)) 
        
    def forward(self, x: torch.Tensor, output='imgs'):
        # Reshape and permute the input tensor
        recover = Resize(x.shape[-2:]) 
        x = self.resize(x)
        x = self._process_input(x)
#         print(x.shape)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
#         print(x.shape)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)
        
        if output == 'logits':
            x = x[:, 0]
            x = self.heads(x)
        elif output == 'imgs':
            x = self.upsample(x[:, 1:])
            x = x.reshape(n, self.numpatches, self.numpatches, self.c, self.patch_size, self.patch_size)
            x = x.permute(0, 3, 1, 4, 2, 5)
            x = x.reshape(n, 3, self.image_size, self.image_size)
#             inp[:,:,-self.image_size:, -self.image_size:] = torch.sigmoid(x)
#             return inp
            
        return recover(torch.sigmoid(x))

    def eval(self):
        self.model.eval()
        
class TransformerWithCNN(VisionTransformer):
    def __init__(self, cnn_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.numpatches = self.image_size//self.patch_size
        self.c = 3#input_depth
        self.upsample = torch.nn.Linear(self.hidden_dim, self.c*self.patch_size*self.patch_size)
        mag, self.sgn = abs(cnn_num), np.sign(cnn_num)
        self.conv = torch.nn.Sequential(
                        (*([torch.nn.Conv2d(3, 3, 3, padding='same', padding_mode='reflect'),
                        torch.nn.ReLU()]*mag)[:-1])
#                         torch.nn.Conv2d(3, 3, 3, padding='same', padding_mode='reflect'),
#                         torch.nn.ReLU(),
#                         torch.nn.Conv2d(3, 3, 3, padding='same', padding_mode='reflect'),
                    )
        
    def forward(self, x: torch.Tensor, output='imgs'):
        if self.sgn == -1:
            x = self.conv(x)
        # Reshape and permute the input tensor
        x = self._process_input(x)
#         print(x.shape)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
#         print(x.shape)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)
        
        if output == 'logits':
            x = x[:, 0]
            x = self.heads(x)
        elif output == 'imgs':
            x = self.upsample(x[:, 1:])
            x = x.reshape(n, self.numpatches, self.numpatches, self.c, self.patch_size, self.patch_size)
            x = x.permute(0, 3, 1, 4, 2, 5)
            x = x.reshape(n, 3, self.image_size, self.image_size)
            if self.sgn == 1:
                x = self.conv(x)
        return torch.sigmoid(x)

    def eval(self):
        self.model.eval()

