import cv2
import numpy as np

img = cv2.imread('data/inpaint360/car/images_inpaint_unseen_virtual/00000.JPG')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
black_pixels = np.sum(gray < 5)
print(f"Number of almost black pixels (< 5): {black_pixels}")

mask = cv2.imread('data/inpaint360/car/inpaint_2d_unseen_mask_virtual/00000.png', cv2.IMREAD_GRAYSCALE)
mask_pixels = np.sum(mask > 128)
print(f"Number of mask pixels: {mask_pixels}")

