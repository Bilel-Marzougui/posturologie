import torch 
import torch.nn as nn

from .ResidualBlock import ResidualBlock


class HourGlass(nn.Module):
    def __init__(self, num_blocks, in_channels):
        super(HourGlass, self).__init__()
        self.in_channels = in_channels
        self.num_blocks = num_blocks
        self.upper_branch = ResidualBlock(in_channels, int(in_channels / 2))
        self.pool = nn.MaxPool2d(2)
        self.lower_branch_encoder = ResidualBlock(in_channels, int(in_channels / 2))

        if self.num_blocks > 1:
            self.inner = HourGlass(self.num_blocks - 1, in_channels)
        else:
            self.inner = ResidualBlock(in_channels, int(in_channels / 2))

        self.lower_blocks_decoder = ResidualBlock(in_channels, int(in_channels / 2))
        self.upsample_block = nn.Upsample(scale_factor=2.0)

    def forward(self, x):
        # To understand the architecture further, look at this figure here ...
        _upper_branch = self.upper_branch(x)
        _pool = self.pool(x)
        # Lower Branch (encoder side)
        _lower_branch_enc = self.lower_branch_encoder(_pool)
        # Recursion to smaller hourglass modules
        _inner = self.inner(_lower_branch_enc)
        # Lower Branch (decoder side)
        _lower_branch_dec = self.lower_blocks_decoder(_inner)
        # Upsampling
        _upsampled = self.upsample_block(_lower_branch_dec)
        return _upsampled + _upper_branch
