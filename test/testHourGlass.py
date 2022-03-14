import sys
sys.path.append("../")

import pose_estimation
from pose_estimation.models.BaseBlock import BaseBlock
from pose_estimation.models.HourGlass import HourGlass
from pose_estimation.models.ResidualBlock import ResidualBlock

print("-"*100)
print("HourGlass with 128 channels as input")
hg = HourGlass(4, 128)
print(hg)
