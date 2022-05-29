import torch
import torch.nn as nn
from numpy.random import normal
from numpy.linalg import svd
from math import sqrt
import torch.nn.init
from .common import *
from transformers import SwinConfig, SwinForMaskedImageModeling, SwinModel

class SwinTransformer(SwinForMaskedImageModeling):        
    def __init__(self, image_size, patch_size, num_layers, num_heads, hidden_dim, mlp_ratio):
        config = SwinConfig(
                    image_size=image_size,
                    patch_size=patch_size,
                    num_layers=num_layers,
                    num_heads=[num_heads]*num_layers,
                    embed_dim=hidden_dim,
                    mlp_ratio=mlp_ratio
                    )
        super().__init__(config)
    
    def forward(self, *args, **kwargs):
        x = super().forward(*args, **kwargs)
#         return x
        return torch.sigmoid(x.logits)

    def eval(self):
        self.model.eval()
