import cv2
import numpy as np

# Load a representative render and its corresponding mask
render_path = "output/inpaint360/car/virtual/ours_object_removal/iteration_2000/renders/00000.png"
mask_path = "data/inpaint360/car/inpaint_2d_unseen_mask_virtual/00000.png"

render = cv2.imread(render_path)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# Create an overlay image
overlay = render.copy()

# Highlight the black holes (where render is very dark) in green
gray_render = cv2.cvtColor(render, cv2.COLOR_BGR2GRAY)
is_hole = gray_render < 5
overlay[is_hole] = [0, 255, 0]  # Green

# Highlight the mask in red
is_mask = mask > 128
overlay[is_mask] = overlay[is_mask] * 0.5 + np.array([0, 0, 255]) * 0.5  # Translucent red

cv2.imwrite("/home/nmarur21/.gemini/antigravity/brain/15249c44-0c17-4872-becc-f4a4d216e2de/browser/mask_overlap.png", overlay)
print("Overlap saved to mask_overlap.png")

