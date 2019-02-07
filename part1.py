import numpy as np
import cv2
import time
from filters import conv

image = cv2.imread("./data/pup.jpg") / 255.0

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
cv2.imwrite("./figures/high_pup.jpg", high * 255)

# Low pass filter
low = conv(image, avg_kernel)
cv2.imwrite("./figures/low_pup.jpg", low * 255)

# Sobel edge filter
sx = np.abs(conv(image, sx_kernel))
cv2.imwrite("./figures/edgex_pup.jpg", sx * 255)
sy = np.abs(conv(image, sy_kernel))
cv2.imwrite("./figures/edgey_pup.jpg", sy * 255)
sobel = sx + sy
cv2.imwrite("./figures/edge_pup.jpg", sobel * 255)
