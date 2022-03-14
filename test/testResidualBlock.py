import sys
sys.path.append("../")
import torch
import pose_estimation

from torchsummary import summary
from pose_estimation.models.HourGlass import HourGlass
from pose_estimation.models.BaseBlock import BaseBlock
from pose_estimation.models.ResidualBlock import ResidualBlock
from pose_estimation.models.StackedHourGlass import StackedHourGlass


# print("-"*100)
# print("ResidualBlock layer in stacked-hour-glass network")
# res_block = ResidualBlock(64, 32, expansion=4)
# print(res_block)
# # summary(res_block, input_size=(64, 32, 32))

# print("-"*100)
# print("Hourglass")
# hglass = HourGlass(num_blocks=4, in_channels=256)
# print(hglass)
# summary(hglass, input_size=(256, 64, 64))

print("-"*100)
print("Hourglass")
shg = StackedHourGlass(num_stacks=6, inp_channels=256, num_blocks=4, num_classes=16)
# print(shg)
# summary(shg, input_size=(3, 256, 256))

inp = torch.zeros((1, 3, 256, 256))
out = shg(inp)
print(len(out))
print(out[0].size())
print(out.size())