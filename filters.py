import numpy as np

# Convolve over image
def conv(image, kernel, mean_center=False):
    # Assuming MxM kernel
    M = len(kernel)
    if M < 1:
        return image

    output = np.zeros_like(image)

    # Apply edge padding
    # Duplicates the last element in each axis by padding amount
    original_shape = image.shape
    image = np.pad(image, [(M/2,M/2) ,(M/2,M/2), (0,0)] , 'edge')

    # Convolve filter
    kernel = np.flip(kernel, axis=0)
    kernel = np.flip(kernel, axis=1)
    kernel = np.tile(kernel[:,:,None], [1,1,original_shape[2]])

    # Operate over entire image
    for x in range(original_shape[0]):
        for y in range(original_shape[1]):
            mult = np.multiply(kernel, image[x:x+M, y:y+M])
            output[x,y] = np.sum(mult, axis=(0,1))

    return output