import sys
sys.path.append('..')

import argparse
import numpy as np
from scipy import signal
from scipy.ndimage.filters import gaussian_filter

from common import Img

# Gaussian smoothing
def smooth_gauss(img, sigma):
    return gaussian_filter(img, np.sqrt(sigma))

# Laplacian mask
laplacian_mask = np.array([
    [1, 1, 1],
    [1, -8, 1],
    [1, 1, 1]
])

# Marr-Hildreth
def marr_hildreth(img, sigma):

    def zero_crossing(img, threshold=12.5):

        def cross(lhs, rhs):
            return np.sign(lhs) != np.sign(rhs) and np.abs(rhs - lhs) > threshold

        output = np.zeros(img.shape, dtype=np.uint8)

        M, N = img.shape
        for (x, y), n in np.ndenumerate(img):
            if x == 0 or x == (M-1) or y == 0 or y == (N-1): continue

            if   cross(img[x-1, y], img[x+1, y]): output[x, y] = 255
            elif cross(img[x, y-1], img[x, y+1]): output[x, y] = 255
            elif cross(img[x-1, y-1], img[x+1, y+1]): output[x, y] = 255
            elif cross(img[x-1, y+1], img[x+1, y-1]): output[x, y] = 255

        return output


    smoothed  = smooth_gauss(img, sigma)
    laplacian = signal.convolve2d(smoothed, laplacian_mask, mode='same')

    return zero_crossing(laplacian)

# Main
if __name__ == "__main__":

    # Available args
    parser = argparse.ArgumentParser(description='Image segmentation - Marr-Hildreth edge detector')

    parser.add_argument('image_path', type=str, help='Image path')

    parser.add_argument('-s', dest='sigma', type=float, default=4,
        help='Variance of the Gaussian smoothing kernel'
    )

    # Parse args
    args = parser.parse_args()

    image = Img.load(args.image_path)

    # Marr-Hildreth
    mh = marr_hildreth(image, args.sigma)
    Img.show(mh)
