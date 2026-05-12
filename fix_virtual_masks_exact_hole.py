import cv2
import numpy as np
import glob
import os
from pathlib import Path

mask_dir = 'data/inpaint360/car/inpaint_2d_unseen_mask_virtual'
render_dir = 'output/inpaint360/car/virtual/ours_object_removal/iteration_2000/renders'

masks = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
renders = sorted(glob.glob(os.path.join(render_dir, '*.png')))

kernel = np.ones((11, 11), np.uint8)

for mask_path, render_path in zip(masks, renders):
    render = cv2.imread(render_path)
    gray_render = cv2.cvtColor(render, cv2.COLOR_BGR2GRAY)
    is_hole = gray_render < 5
    
    # Create mask from hole
    hole_mask = np.zeros_like(gray_render)
    hole_mask[is_hole] = 255
    
    # Dilate slightly to ensure it covers the edge
    final_mask = cv2.dilate(hole_mask, kernel, iterations=1)
    cv2.imwrite(mask_path, final_mask)

print("Virtual masks have been set to exactly the pruned hole + slight dilation.")
