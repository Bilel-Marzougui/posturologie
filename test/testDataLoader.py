import cv2
import sys
sys.path.append("../")
import torch
import torchvision
import pose_estimation

from torchvision import transforms, utils
from pose_estimation.dataset.MPII import MPIIDataset
from pose_estimation.utils.visulatizations.disp_utils import to_np, disp_heatmap
from pose_estimation.utils.transforms.CropAndScale import CropAndScale
from pose_estimation.utils.transforms.AffineTransform import AffineTransform
from pose_estimation.utils.transforms.RandomHorizontalFlip import RandomHorizontalFlip

input_res = 256
output_res = 64
dataset_dir = "/Users/shashanks./college/rrc/dataset/mpii/images"
annotation_path = "/Users/shashanks./college/rrc/dataset/mpii/annot/train.json"

train_transform = transforms.Compose([
    RandomHorizontalFlip(0.5),
    CropAndScale(input_res),
    AffineTransform(input_res, output_res, rotation_prob=0.6)
])

train_dataset = MPIIDataset(annotation_path, dataset_dir, input_res, output_res, transform=train_transform)

for i, sample in enumerate(train_dataset):
    inp, out = sample["image"], sample["heat_maps"]
    print("input.size(): {}, output.size(): {}".format(inp.size(), out.size()))
    print("input - max: {}, min: {}".format(inp.max(), inp.min()))
    print("inp.type(): {}, out.type(): {}".format(inp.type(), out.type()))
    inp_np, out_np = to_np(inp), to_np(out)
    inp_np = inp_np.transpose((1, 2, 0))
    disp_heatmap(cv2.resize(inp_np, (64, 64)), out_np)
    break