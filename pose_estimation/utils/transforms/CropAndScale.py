import cv2
import math
import torch
import torchvision
import numpy as np

from copy import deepcopy


class CropAndScale(object):
    """Crops an image based on the defined bounding box
    If bounding box coordinates lies outside the image, then clips it and replaces
    the clipped portion with zeros.

    Finally resizes the image to a square sized image given by the parameter size

    Parameters
    ----------
    scale: float
        Final size of the image
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = deepcopy(sample.get("image"))
        keypoints = deepcopy(sample.get("keypoints"))
        visible_keypoints = deepcopy(sample.get("visible_keypoints"))
        scale = deepcopy(sample.get("scale"))
        center = deepcopy(sample.get("center"))

        # Need to multiply scale by 200 in case of MPII dataset
        # See the Hourglass paper for more details
        scale = int(scale * 200.0)
        w, h = scale, scale
        orig_h, orig_w, _ = img.shape

        # Compute the upper left and bottom right bounding boxes of the person of interest
        upper_left = int(center[0] - scale / 2.0), int(center[1] - scale / 2.0)
        bottom_right = upper_left[0] + w, upper_left[1] + h

        # New image of size H x W x C, H & W are cropped sizes (Here H = W)
        new_shape = [bottom_right[1] - upper_left[1], bottom_right[0] - upper_left[0], img.shape[2]] 
        new_img = np.zeros(new_shape)

        # Range to fill new array
        # Look at this diagram () for more details:
        new_x = max(0, -upper_left[0]), min(bottom_right[0], orig_w) - upper_left[0]
        new_y = max(0, -upper_left[1]), min(bottom_right[1], orig_h) - upper_left[1]

        # Range to sample from original image
        # Look at this diagram () for more details:
        old_x = max(0, upper_left[0]), min(orig_w, bottom_right[0])
        old_y = max(0, upper_left[1]), min(orig_h, bottom_right[1])

        # Update image array
        new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]
        new_img = cv2.resize(new_img, (self.size, self.size))
        new_img = new_img.astype(np.uint8)

        # Update keypoints location
        keypoints = keypoints - [upper_left[0], upper_left[1]]
        keypoints = keypoints * self.size / scale
        # Update center point
        new_center = np.array([new_img.shape[0] / 2, new_img.shape[1] / 2])
        # Update scale
        new_scale = max(new_img.shape[0], new_img.shape[1])

        updated_sample = {
            "image": new_img,
            "keypoints": keypoints,
            "visible_keypoints": visible_keypoints, # Will remain same
            "scale": new_scale,
            "center": new_center
        }
        return updated_sample