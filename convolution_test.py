import numpy as np
import cv2
from convolution import conv



image = np.array([[255, 255, 255, 255],
                  [255, 255, 255, 4],
                  [255, 255, 255, 4],
                  [255, 2, 3, 4]]) / 255.0
image = np.tile(image[:,:,None], [1,1,1])

image = cv2.imread("pup.jpg") / 255.0
print(image)

print(image.shape)

kernel = np.array([[1,1,1],
                   [1,1,1],
                   [1,1,1]]) * (1.0/9)

convolved = conv(image, kernel, M=len(kernel))
cv2.imshow("original", image)
cv2.imshow("filtered", convolved)
cv2.waitKey(0)
# print(image)
print(convolved)

# print(convolved)
