import numpy as np

def gaussian_blur(img, kernel_size, sigma):
    """Applies a Gaussian filter to the input image"""
    k = cv2.getGaussianKernel(kernel_size, sigma)
    return cv2.sepFilter2D(img, -1, k, k)

def sobel_filters(img):
    """Calculates gradient magnitude and direction using Sobel filters"""
    # Apply the x and y Sobel filters
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    filtered_x = cv2.filter2D(img, cv2.CV_64F, sobel_x)
    filtered_y = cv2.filter2D(img, cv2.CV_64F, sobel_y)
    # Calculate the gradient magnitude and direction
    gradient_magnitude = np.sqrt(np.power(filtered_x, 2) + np.power(filtered_y, 2))
    gradient_direction = np.arctan2(filtered_y, filtered_x)
    return gradient_magnitude, gradient_direction

def non_maximum_suppression(gradient_magnitude, gradient_direction):
    """Thins the edges to a single pixel by suppressing non-maxima along the gradient direction"""
    # Round the gradient direction to one of four possible values
    rounded_direction = np.round(gradient_direction * 4 / np.pi) % 4
    # Create a copy of the gradient magnitude to modify
    thinned = np.copy(gradient_magnitude)
    # Loop through the gradient magnitude
    h, w = gradient_magnitude.shape
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            direction = rounded_direction[y, x]
            if direction == 0:
                if gradient_magnitude[y, x] <= gradient_magnitude[y, x - 1] or gradient_magnitude[y, x] <= gradient_magnitude[y, x + 1]:
                    thinned[y, x] = 0
            elif direction == 1:
                if gradient_magnitude[y, x] <= gradient_magnitude[y - 1, x + 1] or gradient_:
                    thinned[y,x]= 1