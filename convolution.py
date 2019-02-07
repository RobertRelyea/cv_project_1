import numpy as np
import pdb

# Convolve over image
def conv(image, kernel, M=5):
    # Preconditions:
    if M < 1:
        return image

    output = np.zeros_like(image)

    # Apply zero padding
    original_shape = image.shape
    image = np.pad(image, [(M/2,M/2) ,(M/2,M/2), (0,0)] , 'constant')

    # Convolve filter
    kernel = np.flip(kernel, axis=0)
    kernel = np.flip(kernel, axis=1)
    kernel = np.tile(kernel[:,:,None], [1,1,3])

    for x in range(original_shape[0]):
        for y in range(original_shape[1]):
            # pdb.set_trace()
            mult = np.multiply(kernel, image[x:x+M, y:y+M])
            output[x,y] = min( 1.0, np.sum(mult) / 9)

    return output
