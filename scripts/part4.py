import numpy as np
import cv2
from matplotlib import pyplot as plt
from utils import mse

# Remove plt frame
x = plt.axes([0,0,1,1], frameon=False)
x.get_xaxis().set_visible(False)
x.get_yaxis().set_visible(False)

image1 = cv2.imread("../data/n02091032_561.jpg",
                    cv2.IMREAD_GRAYSCALE) / 255.0
fft1 = np.fft.fft2(image1)
fft1_shift = np.fft.fftshift(fft1)
fft1_real_log = 20*np.log(np.abs(fft1_shift))
fft1_imag_log = 20*np.log(fft1_shift.imag)


image2 = cv2.imread("../data/n02091134_755.jpg",
                    cv2.IMREAD_GRAYSCALE) / 255.0
fft2 = np.fft.fft2(image2)
fft2_shift = np.fft.fftshift(fft2)
fft2_real_log = 20*np.log(np.abs(fft2_shift))
fft2_imag_log = 20*np.log(fft2_shift.imag)

### A
# Image 1 FFT
plt.imshow(fft1_real_log, cmap = 'gray')
plt.xticks([]), plt.yticks([])
plt.savefig("../figures/part4/fft1_mag.png", transparent=True)

plt.imshow(fft1_imag_log, cmap = 'gray')
plt.xticks([]), plt.yticks([])
plt.savefig("../figures/part4/fft1_phase.png", transparent=True)

# Image 2 FFT
plt.imshow(fft2_real_log, cmap = 'gray')
plt.xticks([]), plt.yticks([])
plt.savefig("../figures/part4/fft2_mag.png", transparent=True)

plt.imshow(fft2_imag_log, cmap = 'gray')
plt.xticks([]), plt.yticks([])
plt.savefig("../figures/part4/fft2_phase.png", transparent=True)

### B
recon_1 = np.abs(np.fft.ifft2(fft2.real + (1j * fft1.imag)))
print("Recon 1 mse: " + str(mse(recon_1, image1)))
recon_2 = np.abs(np.fft.ifft2(fft1.real + (1j * fft2.imag)))
print("Recon 2 mse: " + str(mse(recon_2, image2)))

plt.imshow(recon_1, cmap = 'gray')
plt.xticks([]), plt.yticks([])
plt.savefig("../figures/part4/fft1_recon.png", transparent=True)

plt.imshow(recon_2, cmap = 'gray')
plt.xticks([]), plt.yticks([])
plt.savefig("../figures/part4/fft2_recon.png", transparent=True)
