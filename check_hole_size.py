import cv2
import numpy as np

render = cv2.imread('output/inpaint360/car/virtual/ours_object_removal/iteration_2000/renders/00000.png')
gray_render = cv2.cvtColor(render, cv2.COLOR_BGR2GRAY)
is_hole = gray_render < 5
print(f"Number of hole pixels: {np.sum(is_hole)}")
