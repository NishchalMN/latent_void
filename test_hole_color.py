import cv2
import numpy as np
import glob

# Find the first image
img_path = glob.glob('data/inpaint360/car/images_unseen_virtual/*')[0]
mask_path = glob.glob('data/inpaint360/car/inpaint_2d_unseen_mask_virtual/*')[0]

print(f"Testing {img_path} and {mask_path}")

img = cv2.imread(img_path)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

black_pixels = np.all(img <= 5, axis=-1)
outside_mask = mask < 128
black_outside = np.logical_and(black_pixels, outside_mask)

print(f"Total pixels: {img.shape[0] * img.shape[1]}")
print(f"Mask pixels: {np.sum(mask >= 128)}")
print(f"Black pixels: {np.sum(black_pixels)}")
print(f"Black pixels outside mask: {np.sum(black_outside)}")
