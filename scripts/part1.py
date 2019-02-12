import numpy as np
import cv2
import time
from utils import conv, threshold

EXPORT_IMAGES = True

image = cv2.imread("../data/pup.jpg", cv2.IMREAD_GRAYSCALE) / 255.0

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
low2 = conv(low, avg_kernel)
low3 = conv(low2, avg_kernel)


# Sobel edge filter
sx = np.abs(conv(image, sx_kernel))
sx = threshold(sx, 0.3)
sy = np.abs(conv(image, sy_kernel))
sy = threshold(sy, 0.3)
sobel = sx + sy
sobel[sobel > 255] = 255

if EXPORT_IMAGES:
    cv2.imwrite("../figures/part1/high_pup.jpg", high * 255)
    cv2.imwrite("../figures/part1/low_pup.jpg", low * 255)
    cv2.imwrite("../figures/part1/low2_pup.jpg", low2 * 255)
    cv2.imwrite("../figures/part1/low3_pup.jpg", low3 * 255)
    cv2.imwrite("../figures/part1/edgex_pup.jpg", sx * 255)
    cv2.imwrite("../figures/part1/edgey_pup.jpg", sy * 255)
    cv2.imwrite("../figures/part1/edge_pup.jpg", sobel * 255)
