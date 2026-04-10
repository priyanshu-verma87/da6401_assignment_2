"""VGG11 encoder
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

def _conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    """
    Conv2d(3x3, pad=1) -> BatchNorm2d -> ReLU.
    bias=False because BN has its own learnable bias (beta).
    padding=1 preserves spatial dimensions after convolution.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class VGG11Encoder(nn.Module):
    """VGG11-style encoder with optional intermediate feature returns.

    Architecture:
        Block 1 : Conv(64)               → MaxPool  (B, 64, 112, 112)
        Block 2 : Conv(128)              → MaxPool  (B, 128, 56, 56)
        Block 3 : Conv(256) → Conv(256)  → MaxPool  (B, 256, 28, 28)
        Block 4 : Conv(512) → Conv(512)  → MaxPool  (B, 512, 14, 14)
        Block 5 : Conv(512) → Conv(512)  → MaxPool  (B, 512, 7, 7)

    Architectural Reasoning:
    BatchNorm placement:
        BN2d placed after every Conv2d and before ReLU.
        Reason: BN normalises the pre-activation distribution, keeping inputs to ReLU in a healthy range. This stabilises training, accelerates convergence, and allows higher stable learning rates by reducing internal covariate shift.

    Args:
        in_channels : number of input image channels 
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        # Convolutional blocks 
        self.block1 = nn.Sequential(_conv_bn_relu(in_channels, 64))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block2 = nn.Sequential(_conv_bn_relu(64, 128))
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block3 = nn.Sequential(
            _conv_bn_relu(128, 256),
            _conv_bn_relu(256, 256),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block4 = nn.Sequential(
            _conv_bn_relu(256, 512),
            _conv_bn_relu(512, 512),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block5 = nn.Sequential(
            _conv_bn_relu(512, 512),
            _conv_bn_relu(512, 512),
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: input image tensor [B, 3, H, W].
            return_features: if True, also return skip maps for U-Net decoder.

        Returns:
            - if return_features=False: bottleneck feature tensor.
            - if return_features=True: (bottleneck, feature_dict).
        """
        # Block 1 
        s1 = self.block1(x)          # (B,  64, 224, 224)
        x  = self.pool1(s1)          # (B,  64, 112, 112)

        # Block 2 
        s2 = self.block2(x)          # (B, 128, 112, 112)
        x  = self.pool2(s2)          # (B, 128,  56,  56)

        # Block 3 
        s3 = self.block3(x)          # (B, 256,  56,  56)
        x  = self.pool3(s3)          # (B, 256,  28,  28)

        # Block 4 
        s4 = self.block4(x)          # (B, 512,  28,  28)
        x  = self.pool4(s4)          # (B, 512,  14,  14)

        # Block 5 
        s5 = self.block5(x)          # (B, 512,  14,  14)
        x  = self.pool5(s5)          # (B, 512,   7,   7): bottleneck

        # return 
        if not return_features:
            return x

        feature_dict = {
            'block1': s1,
            'block2': s2,
            'block3': s3,
            'block4': s4,
            'block5': s5,
        }
        return x, feature_dict
    
    def get_blocks(self) -> list:
        """
        Returns conv blocks in order.
        Used in Tasks 2/3/4 to selectively freeze/unfreeze backbone layers.
        """
        return [self.block1, self.block2, self.block3, self.block4, self.block5]


# Alias so autograder can do: from models.vgg11 import VGG11
VGG11 = VGG11Encoder