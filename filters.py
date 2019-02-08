import numpy as np
from math import log10

# Convolve over image
def conv(image, kernel):
    # Assuming MxM kernel
    M = len(kernel)
    if M < 1:
        return image

    output = np.zeros_like(image)

    # Apply edge padding
    # Duplicates the last element in each axis by padding amount
    original_shape = image.shape
    image = np.pad(image, [(M/2,M/2) ,(M/2,M/2)] , 'edge')

    # Convolve filter
    kernel = np.flip(kernel, axis=0)
    kernel = np.flip(kernel, axis=1)

    # Operate over entire image
    for x in range(original_shape[0]):
        for y in range(original_shape[1]):
            mult = np.multiply(kernel, image[x:x+M, y:y+M])
            output[x,y] = np.sum(mult, axis=(0,1))

    return output

def median(image, M):
    if M < 1:
        return image

    output = np.zeros_like(image)

    # Apply edge padding
    # Duplicates the last element in each axis by padding amount
    original_shape = image.shape
    image = np.pad(image, [(M/2,M/2) ,(M/2,M/2)] , 'edge')

    # Operate over entire image
    for x in range(original_shape[0]):
        for y in range(original_shape[1]):
            output[x,y] = np.median(image[x:x+M, y:y+M])

    return output

def snr(original, noisy):
    original_var = np.var(original)
    noise_var = np.var(noisy)
    return 10 * log10(original_var/noise_var)
