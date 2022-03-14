import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ...config.mpii_config import keypoints_idx_to_names

def disp_keypoints_image(img, keypoints):
    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(1, 1, 1)
    keypoints_num = list(range(len(keypoints)))
    ax1.imshow(img)
    ax1.scatter(keypoints[:, 0], keypoints[:, 1])
    for i, txt in enumerate(keypoints_num):
        ax1.annotate(txt, (keypoints[i, 0], keypoints[i, 1]), c='w')
    plt.show()

def disp_bbox_image(img, center, scale):
    x, y = int(center[0] - scale / 2.0), int(center[1] - scale / 2.0)
    w, h = int(scale), int(scale)
    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.imshow(img)
    # Create a Rectangle patch
    rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
    # Add the patch to the Axes
    ax1.add_patch(rect)
    plt.show()

def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
    return mycmap

def disp_heatmap(img, heat_maps):
    y, x = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    #Use base cmap to create transparent
    mycmap = transparent_cmap(plt.cm.Reds)
    #Plot image and overlay colormap
    # fig, ax = plt.subplots(4, 4, figsize=(12, 12), sharey='row')
    fig, ax = plt.subplots(4, 4, figsize=(12, 12))
    heatmap_num = 0
    for row in range(4):
        for col in range(4):
            ax[row, col].set_title(keypoints_idx_to_names[heatmap_num])
            ax[row, col].imshow(img)
            cb = ax[row, col].contourf(x, y, heat_maps[heatmap_num], 15, cmap=mycmap)
            heatmap_num += 1
    fig.tight_layout()
    plt.show()

def to_np(inp):
    return inp.detach().numpy()