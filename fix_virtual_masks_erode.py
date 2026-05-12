import cv2
import numpy as np
import glob
import os
from pathlib import Path

mask_dir = 'data/inpaint360/car/inpaint_2d_unseen_mask_virtual'

masks = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
# Erode by 56 pixels to undo 56 pixels of the 81-pixel dilation, leaving a 25-pixel dilation.
kernel = np.ones((56, 56), np.uint8)

for mask_path in masks:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is not None:
        final_mask = cv2.erode(mask, kernel, iterations=1)
        cv2.imwrite(mask_path, final_mask)

print("Virtual masks have been eroded back to a reasonable dilation size (approx +25px).")
