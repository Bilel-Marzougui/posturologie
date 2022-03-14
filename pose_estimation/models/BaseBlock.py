import torch
import torch.nn as nn


class BaseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=2, downsampling=1, bias=False):
        super(BaseBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        self.downsampling = downsampling
        self.bias = bias
        self.expanded_channels = self.out_channels * self.expansion
        self.is_shortcut = (self.in_channels != self.expanded_channels)

        if self.is_shortcut:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(self.in_channels),
                nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1, stride=self.downsampling, bias=False)
            )
