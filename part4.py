import numpy as np
import cv2
from matplotlib import pyplot as plt
from utils import mse
import pdb
from skimage.io import imread

image1 = cv2.imread("./data/n02091032_561.jpg",
                    cv2.IMREAD_GRAYSCALE) / 255.0
# image1 = imread('./data/test1.gif', as_gray=True)
fft1 = np.fft.fft2(image1)
fft1_shift = np.fft.fftshift(fft1)
fft1_real_log = 20*np.log(np.abs(fft1_shift))
fft1_imag_log = 20*np.log(fft1_shift.imag)


image2 = cv2.imread("./data/n02091134_755.jpg",
                    cv2.IMREAD_GRAYSCALE) / 255.0
# image2 = imread('./data/test2.gif', as_gray=True)
fft2 = np.fft.fft2(image2)
fft2_shift = np.fft.fftshift(fft2)
fft2_real_log = 20*np.log(np.abs(fft2_shift))
fft2_imag_log = 20*np.log(fft2_shift.imag)

### A
# plt.imshow(fft1_real_log, cmap = 'gray')
# plt.title('Image 1 Log Magnitude'), plt.xticks([]), plt.yticks([])
# plt.show()
# plt.imshow(fft1_imag_log, cmap = 'gray')
# plt.title('Image 1 Log Phase'), plt.xticks([]), plt.yticks([])
# plt.show()
#
# plt.imshow(fft2_real_log, cmap = 'gray')
# plt.title('Image 2 Log Magnitude'), plt.xticks([]), plt.yticks([])
# plt.show()
# plt.imshow(fft2_imag_log, cmap = 'gray')
# plt.title('Image 2 Log Phase'), plt.xticks([]), plt.yticks([])
# plt.show()

### B
recon_1 = np.abs(np.fft.ifft2(fft1.real + (1j * fft2.imag)))
recon_2 = np.abs(np.fft.ifft2(fft2.real + (1j * fft1.imag)))

plt.imshow(recon_1, cmap = 'gray')
plt.title('Image 1 Reconstructed'), plt.xticks([]), plt.yticks([])
plt.show()

plt.imshow(recon_2, cmap = 'gray')
plt.title('Image 2 Reconstructed'), plt.xticks([]), plt.yticks([])
plt.show()
