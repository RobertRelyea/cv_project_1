import numpy as np
import cv2
from matplotlib import pyplot as plt
import csv
from utils import mse, circle_mask

image = cv2.imread("./data/pup.jpg", cv2.IMREAD_GRAYSCALE) / 255.0
fft = np.fft.fft2(image)

### A
# Unshifted
plt.imshow(np.abs(fft), cmap = 'gray')
plt.title('Magnitude'), plt.xticks([]), plt.yticks([])
plt.savefig("./figures/part3/fft.png")

# # Shifted
fft_shift = np.fft.fftshift(fft)
plt.imshow(np.abs(fft_shift), cmap = 'gray')
plt.title('Magnitude Shifted'), plt.xticks([]), plt.yticks([])
plt.savefig("./figures/part3/fft_shift.png")

# Log applied
fft_log = 20*np.log(np.abs(fft_shift))
plt.imshow(fft_log, cmap = 'gray')
plt.title('Log Magnitude Shifted'), plt.xticks([]), plt.yticks([])
plt.savefig("./figures/part3/fft_log.png")

### B
# Inverse
reconstructed = np.abs(np.fft.ifft2(fft_shift))
plt.imshow(reconstructed, cmap='gray')
plt.title('Reconstructed Image'), plt.xticks([]), plt.yticks([])
plt.savefig("./figures/part3/recon.png")

### C
# Radius of N/3
freq_3 = circle_mask(fft_shift, 3)
recon_3 = np.abs(np.fft.ifft2(freq_3))# + np.abs(fft_shift)))
plt.imshow(recon_3, cmap='gray')
plt.title('Reconstructed Image (N/3)'), plt.xticks([]), plt.yticks([])
plt.savefig("./figures/part3/recon_3.png")

# Radius of N/4
freq_4 = circle_mask(fft_shift, 4)
recon_4 = np.abs(np.fft.ifft2(freq_4))# + np.abs(fft_shift)))
plt.imshow(recon_4, cmap='gray')
plt.title('Reconstructed Image (N/4)'), plt.xticks([]), plt.yticks([])
plt.savefig("./figures/part3/recon_4.png")

# Radius of N/8
freq_8 = circle_mask(fft_shift, 8)
recon_8 = np.abs(np.fft.ifft2(freq_8))# + np.abs(fft_shift)))
plt.imshow(recon_8, cmap='gray')
plt.title('Reconstructed Image (N/8)'), plt.xticks([]), plt.yticks([])
plt.savefig("./figures/part3/recon_8.png")

# Radius of N/16
freq_16 = circle_mask(fft_shift, 16)
recon_16 = np.abs(np.fft.ifft2(freq_16))# + np.abs(fft_shift)))
plt.imshow(recon_16, cmap='gray')
plt.title('Reconstructed Image (N/16)'), plt.xticks([]), plt.yticks([])
plt.savefig("./figures/part3/recon_16.png")

# Compile MSE results
mse_results = []
mse_results.append(("Unaltered",
                    '{:0.3e}'.format(mse(image, reconstructed))))
mse_results.append(("N/3", '{:0.3e}'.format(mse(image, recon_3))))
mse_results.append(("N/4", '{:0.3e}'.format(mse(image, recon_4))))
mse_results.append(("N/8", '{:0.3e}'.format(mse(image, recon_8))))
mse_results.append(("N/16", '{:0.3e}'.format(mse(image, recon_16))))

# Export results to csv
with open('./figures/part3/results.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(['reconstruction', 'mse'])
    for (recon, value) in mse_results:
       writer.writerow([recon, value])
