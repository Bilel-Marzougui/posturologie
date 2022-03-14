import cv2
import math
import torch
import torchvision
import numpy as np

from copy import deepcopy
from ...config.mpii_config import flip_pairs
# from config.mpii_config import flip_pairs


class RandomHorizontalFlip(object):
    """Flips the image and corresponding keypoints randomly with a probability given by prob.
    """
    def __init__(self, prob=0.5):
        assert prob >= 0 and prob <= 1, "Invalid probability"
        self.prob = prob

    def __call__(self, sample):
        img = deepcopy(sample.get("image"))
        keypoints = deepcopy(sample.get("keypoints"))
        visible_keypoints = deepcopy(sample.get("visible_keypoints"))
        scale = deepcopy(sample.get("scale"))
        center = deepcopy(sample.get("center"))
        img_shape = np.array(img.shape[:2])

        # Get the centerpoints, we flip rows and columns as we are dealing with matrices
        img_center = img_shape[::-1] / 2
        if np.random.rand() < self.prob:
            img =  img[:, ::-1, :] # Flip the image
            keypoints[:, 0] += 2 * (img_center[0] - keypoints[:, 0])
            center[0] += 2 * (img_center[0] - center[0])
            # On flipping, the indices of certain keypoints must be exchanged (e.g. left and right ankle)
            for pair in flip_pairs:
                idx_1, idx_2 = pair
                keypoints[idx_1, :], keypoints[idx_2, :] = keypoints[idx_2, :], keypoints[idx_1, :].copy()
                visible_keypoints[idx_1], visible_keypoints[idx_2] = visible_keypoints[idx_2], \
                    visible_keypoints[idx_1].copy()
            updated_sample = {
                "image": img,
                "keypoints": keypoints,
                "visible_keypoints": visible_keypoints, # Will remain same
                "scale": scale,
                "center": center
            }
            return updated_sample
        return deepcopy(sample)