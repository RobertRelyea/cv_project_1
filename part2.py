import numpy as np
import cv2
from skimage.util import random_noise
import matplotlib
from filters import conv, median, snr

EXPORT_IMAGES = False

image = cv2.imread("./data/pup.jpg", cv2.IMREAD_GRAYSCALE) / 255.0

# Generate salt and pepper noise
seasoned_image = random_noise(image, mode='s&p', seed=0)
seasoned_snr = snr(image, seasoned_image)
print("Salt and pepper SNR: " + str(seasoned_snr) + " dB")

# Generate Gaussian noise
gaussed_image = random_noise(image, mode='gaussian', seed=0)
gaussed_snr = snr(image, gaussed_image)
print("Gaussian SNR: " + str(gaussed_snr) + " dB")

# Apply a median filter over image
# 5x5 averaging filter kernel (low pass)
avg_kernel = np.ones((5,5)) / 25.0
averaged_simage = conv(seasoned_image, avg_kernel)
averaged_gimage = conv(gaussed_image, avg_kernel)


# Apply a median filter over image
median_simage = median(seasoned_image, 5)
median_gimage = median(gaussed_image, 5)

# Sobel edge detection filters
sx_kernel = [[-1,0,1],
             [-2,0,2],
             [-1,0,1]]

sy_kernel = [[1,2,1],
             [0,0,0],
             [-1,-2,-1]]

# Sobel edge filter
sx = np.abs(conv(seasoned_image, sx_kernel))
sy = np.abs(conv(seasoned_image, sy_kernel))
sobel_simage = sx + sy
sx = np.abs(conv(gaussed_image, sx_kernel))
sy = np.abs(conv(gaussed_image, sy_kernel))
sobel_gimage = sx + sy

if EXPORT_IMAGES:
    cv2.imwrite("./figures/part2/seasoned_pup.jpg", seasoned_image * 255)
    cv2.imwrite("./figures/part2/gaussed_pup.jpg", gaussed_image * 255)
    cv2.imwrite("./figures/part2/averaged_spup.jpg", averaged_simage * 255)
    cv2.imwrite("./figures/part2/averaged_gpup.jpg", averaged_gimage * 255)
    cv2.imwrite("./figures/part2/median_spup.jpg", median_simage * 255)
    cv2.imwrite("./figures/part2/median_gpup.jpg", median_gimage * 255)
    cv2.imwrite("./figures/part2/sobel_spup.jpg", sobel_simage * 255)
    cv2.imwrite("./figures/part2/sobel_gpup.jpg", sobel_gimage * 255)
