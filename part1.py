import numpy as np
import cv2
import time
from utils import conv

EXPORT_IMAGES = False

image = cv2.imread("./data/pup.jpg", cv2.IMREAD_GRAYSCALE) / 255.0

# 5x5 averaging filter kernel (low pass)
avg_kernel = np.ones((5,5)) / 25.0

# 5x5 high pass filter
high_pass_kernel = [[-1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1],
                    [-1, -1, 24, -1, -1],
                    [-1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1]]
high_pass_kernel = np.array(high_pass_kernel) / 25.0

# Sobel edge detection filters
sx_kernel = [[-1,0,1],
             [-2,0,2],
             [-1,0,1]]

sy_kernel = [[1,2,1],
             [0,0,0],
             [-1,-2,-1]]

# High pass filter
conv_start = time.time()
high = conv(image, high_pass_kernel)
conv_end = time.time()
print("Highpass execution time (s): " + str(conv_end - conv_start))

# Low pass filter
low = conv(image, avg_kernel)

# Sobel edge filter
sx = np.abs(conv(image, sx_kernel))
sy = np.abs(conv(image, sy_kernel))
sobel = sx + sy


if EXPORT_IMAGES:
    cv2.imwrite("./figures/part1/high_pup.jpg", high * 255)
    cv2.imwrite("./figures/part1/low_pup.jpg", low * 255)
    cv2.imwrite("./figures/part1/edgex_pup.jpg", sx * 255)
    cv2.imwrite("./figures/part1/edgey_pup.jpg", sy * 255)
    cv2.imwrite("./figures/part1/edge_pup.jpg", sobel * 255)
