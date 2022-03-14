import torch 
import torch.nn as nn

from .ResidualBlock import ResidualBlock
from .HourGlass import HourGlass


class StackedHourGlass(nn.Module):
    def __init__(self, num_stacks, inp_channels, num_classes, num_blocks=4):
        super(StackedHourGlass, self).__init__()
        self.num_stacks = num_stacks
        self.inp_channels = inp_channels
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        # Preprocessing Conv layers on the inputs before the hourglass modules
        self.pre_hg = nn.Sequential(
            # 3 x 256 x 256
            self.get_conv_bn_relu(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=True),
            # 64 x 128 x 128
            ResidualBlock(64, 64),
            # 128 x 128 x 128
            nn.MaxPool2d(2),
            # 128 x 64 x 64
            ResidualBlock(128, 128),
            # 256 x 64 x 64
            ResidualBlock(256, int(inp_channels / 2))
            # 256 x 64 x 64
        )
        # List of all the hourglass modules
        self.hour_glass = nn.ModuleList([
            HourGlass(num_blocks=self.num_blocks, in_channels=self.inp_channels)
            for i in range(self.num_stacks)
        ])
        
        # List of all the feature (Residual + 1x1 Conv) layers that are right after the hourglass modules
        self.features = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(self.inp_channels, int(self.inp_channels / 2)),
                self.get_conv_bn_relu(self.inp_channels, self.inp_channels, 1)
            ) for i in range(self.num_stacks)
        ])
        
        # List of all the 1x1 Conv layers that generate the intermediate outputs (predictions) of the network
        self.preds = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.inp_channels, self.num_classes, 1)
            ) for i in range(self.num_stacks)
        ])
        
        # Conv layer that remaps the predictions to the intermediate features 
        self.combine_preds = nn.ModuleList([
            self.get_conv_bn_relu(self.num_classes, self.inp_channels, 1)
            for i in range(self.num_stacks - 1)
        ])
        
        # Apply another conv layer to the intermediate features produced earlier
        self.combine_features = nn.ModuleList([
            self.get_conv_bn_relu(self.inp_channels, self.inp_channels, 1)
            for i in range(self.num_stacks - 1)
        ])


    def get_conv_bn_relu(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        outputs = []
        _out = self.pre_hg(x)
        for i in range(self.num_stacks):
            _hg = self.hour_glass[i](_out)
            _features = self.features[i](_hg)
            # Compute predictions
            _preds = self.preds[i](_features)
            outputs.append(_preds)
            # Merge the multiple branches for all the hourglass modules except the last one
            if i < self.num_stacks - 1:
                _combine_preds = self.combine_preds[i](_preds)
                _combine_features = self.combine_features[i](_features)
                _out = _out + _combine_features + _combine_preds
        # outputs = torch.stack(outputs, dim=1)
        return outputs

