import torch
import torch.nn as nn

from .BaseBlock import BaseBlock


class ResidualBlock(BaseBlock):
    def __init__(self, in_channels, out_channels, expansion=2, downsampling=1, bias=False):
        super().__init__(in_channels, out_channels, expansion, downsampling, bias)
        self.inp_blocks = nn.Sequential(
            self.get_bn_relu_conv(self.in_channels, self.out_channels, kernel_size=1, bias=bias),
            self.get_bn_relu_conv(self.out_channels, self.out_channels, kernel_size=3, \
                downsampling=self.downsampling, padding=1, bias=bias),
            self.get_bn_relu_conv(self.out_channels, self.expanded_channels, kernel_size=1, bias=bias)
        )

    def get_bn_relu_conv(self, in_channels, out_channels, kernel_size, downsampling=1, padding=0, bias=False):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size, downsampling, padding, bias=bias)
        )

    def forward(self, x):
        residual = x
        if self.is_shortcut:
            residual = self.shortcut(x)
        x = self.inp_blocks(x)
        x = x + residual
        return x
