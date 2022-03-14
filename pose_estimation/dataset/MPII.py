import os
import cv2
import math
import json
import torch
import torchvision
import numpy as np

from torch.utils.data import Dataset
from .GenerateHeatMaps import GenerateHeatMaps

class MPIIDataset(Dataset):
    def __init__(self, annotation_path, dataset_dir, input_res, output_res, is_train=True, transform=None):
        self.annotation_path = annotation_path
        self.dataset_dir = dataset_dir
        self.is_train = is_train
        self.transform = transform
        self.input_res = input_res
        self.output_res = output_res
        with open(self.annotation_path, 'r') as f:
            self.annotation_json = json.load(f)

    def __len__(self):
        return len(self.annotation_json)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_dir, self.annotation_json[idx]['image'])
        center = self.annotation_json[idx]['center']
        scale = self.annotation_json[idx]['scale']
        joints = np.array(self.annotation_json[idx]['joints'])
        visible_joints = np.array(self.annotation_json[idx]['joints_vis'])
        img = cv2.imread(img_path)[:,:,::-1] #OpenCV uses BGR channels

        # Adjust center/scale slightly to avoid cropping limbs
        if center[0] != -1:
            center[1] = center[1] + 15 * scale
            scale = scale * 1.25

        inp = {
            "image": img,
            "keypoints": joints,
            "visible_keypoints": visible_joints,
            "scale": scale,
            "center": center
        }
        if self.transform:
            inp = self.transform(inp)
        heat_map_obj = GenerateHeatMaps(self.output_res, inp["keypoints"], inp["visible_keypoints"])
        heat_maps = heat_map_obj.get_heatmaps()
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = inp["image"].transpose((2, 0, 1)) / 255.0
        image = image.astype(np.float64)
        heat_maps = heat_maps.astype(np.float64)
        return {
            'image': torch.from_numpy(image),
            'heat_maps': torch.from_numpy(heat_maps)
        }