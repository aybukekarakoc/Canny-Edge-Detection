import numpy as np

def canny_edge_detection(img, sigma=1, low_threshold=20, high_threshold=100):
    # Step 1: Gaussian filter
    img = np.array(img, dtype=np.float32)
    rows, cols = img.shape
    filter_size = int(6 * sigma) | 1
    pad_size = filter_size // 2
    img = np.pad(img, [(pad_size, pad_size), (pad_size, pad_size)], 'reflect')
    kernel = np.zeros((filter_size, filter_size), dtype=np.float32)
    for i in range(-pad_size, pad_size + 1):
        for j in range(-pad_size, pad_size + 1):
            kernel[i + pad_size, j + pad_size] = np.exp(-(i * 2 + j * 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)
    img = np.abs(np.convolve(img, kernel, mode='valid'))

    # Step 2: Gradient computation
    gradient_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    gradient_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gradient_x = np.convolve(img, gradient_x, mode='valid')
    gradient_y = np.convolve(img, gradient_y, mode='valid')
    gradient_magnitude = np.sqrt(gradient_x * 2 + gradient_y * 2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)

    # Step 3: Non-maximum suppression
    gradient_direction = (gradient_direction + np.pi) / (2 * np.pi) * 8
    gradient_direction = gradient_direction.round().astype(int) % 8
    gradient_magnitude = gradient_magnitude[1:-1, 1:-1]
    gradient_direction = gradient_direction[1:-1, 1:-1]
    suppressed_magnitude = np.zeros((rows, cols), dtype=np.float32)
