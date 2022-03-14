import numpy as np
import scipy.stats as st

class GenerateHeatMaps(object):
    def __init__(self, output_res, keypoints, visible_keypoints, sigma=1):
        self.sigma = sigma
        self.keypoints = keypoints
        self.visible_keypoints = visible_keypoints
        self.num_keypoints = len(self.keypoints)
        self.size = output_res
        self.kernlen = 6 * self.sigma + 4 # arbitrarily chosen
        self.heat_map = np.zeros((self.num_keypoints, self.size, self.size))

    def gkern(self, kernlen=10, nsig=1):
        """Returns a 2D Gaussian kernel.
        Implementation taken from this stackoverflow answer: https://stackoverflow.com/a/29731818
        """
        x = np.linspace(-nsig, nsig, kernlen+1)
        kern1d = np.diff(st.norm.cdf(x))
        kern2d = np.outer(kern1d, kern1d)
        return kern2d / kern2d[kernlen // 2, kernlen // 2]

    def get_heatmaps(self):
        for keypoint_num in range(self.keypoints.shape[0]):
            if self.visible_keypoints[keypoint_num]:
                # Get the keypoint location and generate the gaussian kernel
                x, y = int(np.round(self.keypoints[keypoint_num, 0])), int(np.round(self.keypoints[keypoint_num, 1]))
                _, orig_h, orig_w = self.heat_map.shape
                gauss_kern = self.gkern(self.kernlen, self.sigma)

                # Define bounds of the gaussian
                upper_left = x - 3 * self.sigma - 1, y - 3 * self.sigma - 1
                bottom_right = x + 3 * self.sigma + 2, y + 3 * self.sigma + 2

                # Range to fill new array
                # Look at this diagram (./images/bbox_image_intersection.png) for better understanding:
                new_x = max(0, -upper_left[0]), min(bottom_right[0], orig_w) - upper_left[0]
                new_y = max(0, -upper_left[1]), min(bottom_right[1], orig_h) - upper_left[1]

                # Range to sample from original heatmap
                # Look at this diagram (./images/bbox_image_intersection.png) for better understanding:
                old_x = max(0, upper_left[0]), min(orig_w, bottom_right[0])
                old_y = max(0, upper_left[1]), min(orig_h, bottom_right[1])

                self.heat_map[keypoint_num, old_y[0]:old_y[1], old_x[0]:old_x[1]] = gauss_kern[new_y[0]:new_y[1], new_x[0]:new_x[1]]
        return self.heat_map