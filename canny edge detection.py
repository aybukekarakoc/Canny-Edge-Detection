import numpy as np
from scipy import ndimage
""" image okutma"""
import cv2
import matplotlib.pyplot as plt

# Load image
"""img = ndimage.imread("formula.jpg", flatten=True)"""
img=cv2.imread('formula.jpg')

# x
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
grad_x = ndimage.convolve(img, sobel_x)

# y
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
grad_y = ndimage.convolve(img, sobel_y)


grad = np.sqrt(np.power(grad_x, 2) + np.power(grad_y, 2))


plt.figure()
plt.subplot(121)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.subplot(122)
plt.imshow(grad, cmap="gray")
plt.title("Edge-detected")
plt.show()




"""def sobel_edge_detection(image):
    
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    
    filtered_x = np.abs(np.convolve(image, kernel_x, mode='same'))
    filtered_y = np.abs(np.convolve(image, kernel_y, mode='same'))


    filtered = np.sqrt(np.square(filtered_x) + np.square(filtered_y))

    return filtered

sobel_edge_detection()"""


