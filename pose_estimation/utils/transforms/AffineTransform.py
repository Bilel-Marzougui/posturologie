import cv2
import math
import torch
import torchvision
import numpy as np

from copy import deepcopy


class AffineTransform(object):
    def __init__(self, input_res=256, output_res=64, rotation_prob=0.6):
        self.input_res = input_res
        self.output_res = output_res
        self.rotation_prob = rotation_prob

    def get_translation_matrix(self, pt):
        "Translate the points to the given point pt"
        T = np.float32([
            [1, 0, pt[0]],
            [0, 1, pt[1]],
            [0, 0, 1]
        ])
        return T

    def get_rotation_matrix(self, rot_angle):
        "Rotate the points with rot_angle around the center"
        rot_rad = - rot_angle * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        T = np.float32([
            [cs, -sn, 0],
            [sn,  cs, 0],
            [0 ,   0, 1]
        ])
        return T

    def get_scale_matrix(self, scale_x, scale_y):
        "Scale the points along x & y directions with scale_x & scale_y"
        T = np.float32([
            [scale_x, 0      , 0],
            [0      , scale_y, 0],
            [0      , 0      , 1]
        ])
        return T

    def get_affine_matrix(self, center, scale, res, rot_angle):
        # First translate all the image points to the center of the image
        # We want to scale the image from the center portion
        T1 = self.get_translation_matrix(-center)

        # Scale the image along x & y with values scale_x & scale_y
        # Numpy arrays and image axes are flipped
        scale_x, scale_y = res[1] / scale, res[0] / scale
        T2 = self.get_scale_matrix(scale_x, scale_y)

        # Rotate the image around the center with angle rot_angle
        T3 = self.get_rotation_matrix(rot_angle)

        # Translate the image points to the new origin point
        # Numpy arrays and image axes are flipped
        T4 = self.get_translation_matrix([res[1] / 2, res[0] / 2])
        T_final = np.dot(T4, np.dot(T3, np.dot(T2, T1)))
        return T_final

    def get_random_range(self, xmin, xmax):
        return np.random.random() * (xmax - xmin) + xmin

    def get_keypoints(self, keypoints, T_keypoints):
        new_keypoints = np.c_[keypoints, np.ones(len(keypoints))]
        return np.dot(new_keypoints, T_keypoints.T).reshape(keypoints.shape)

    def update_visible_keypoints(self, res, keypoints, visible_keypoints):
        for i, point in enumerate(keypoints):
            # Axes of image and keypoints are interchanged
            x, y = np.round(point[0]), np.round(point[1])
            if x < 0 or x >= res[1] or y < 0 or y >= res[0]:
                visible_keypoints[i] = 0
        return visible_keypoints

    def __call__(self, sample):
        img = deepcopy(sample.get("image"))
        keypoints = deepcopy(sample.get("keypoints"))
        visible_keypoints = deepcopy(sample.get("visible_keypoints"))
        scale = deepcopy(sample.get("scale"))
        center = deepcopy(sample.get("center"))

        # Scale and rotate by a random value in given range
        scale = scale * self.get_random_range(xmin=0.75, xmax=1.25)
        rot_angle = 0
        if np.random.uniform() >= self.rotation_prob:
            rot_angle = self.get_random_range(xmin=-30, xmax=30)

        # Get affine transforms for image and keypoints respectively
        T_img = self.get_affine_matrix(center, scale, (self.input_res, self.input_res), rot_angle)[:2]
        T_keypoints = self.get_affine_matrix(center, scale, (self.output_res, self.output_res), rot_angle)[:2]

        new_img = cv2.warpAffine(img, T_img, (self.input_res, self.input_res))
        new_keypoints = self.get_keypoints(keypoints, T_keypoints)
        new_visible_keypoints = self.update_visible_keypoints((self.output_res, self.output_res), new_keypoints, visible_keypoints)

        updated_sample = {
            "image": new_img,
            "keypoints": new_keypoints,
            "visible_keypoints": new_visible_keypoints,
            "scale": scale,
            "center": center
        }
        return updated_sample