import cv2
import numpy as np
import glob
import os
from pathlib import Path

mask_dir = 'data/inpaint360/car/inpaint_2d_unseen_mask_virtual'
render_dir = 'output/inpaint360/car/virtual/ours_object_removal/iteration_2000/renders'

masks = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
# Dilate by 40 pixels (kernel 81x81) to aggressively cover the hole
kernel = np.ones((81, 81), np.uint8)

for mask_path in masks:
    filename = Path(mask_path).name
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is not None:
        final_mask = cv2.dilate(mask, kernel, iterations=1)
        cv2.imwrite(mask_path, final_mask)

print("Virtual masks have been massively dilated to cover the full pruned holes!")
