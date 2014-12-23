import sys
sys.path.append('..')

import argparse
import numpy as np

from common import Img

# Global thresholding
def global_thresholding(img, epsilon=0.01):

    threshold     = np.percentile(img, 1.0)
    new_threshold = np.percentile(img, 99.0)

    while np.abs(threshold - new_threshold) > epsilon:
        threshold = new_threshold
        m1 = np.mean(img[img > threshold])
        m2 = np.mean(img[img <= threshold])
        new_threshold = 0.5 * (m1 + m2)

    output = np.zeros(img.shape, np.uint8)
    output[img > new_threshold] = 255

    return output

# Otsu
def otsu(img):

    M, N = img.shape

    # Normalized histogram
    histogram = np.array([0.0 for _ in xrange(256)], dtype=np.float)
    for _, x in np.ndenumerate(img):
        histogram[x] += 1

    histogram = histogram / (M * N)

    # Cumulative sums
    cumul_sums = np.copy(histogram)
    for k in xrange(1, 256):
        cumul_sums[k] = cumul_sums[k-1] + histogram[k]

    # Cumulative means
    cumul_means = np.zeros(histogram.shape)
    for k in xrange(1, 256):
        cumul_means[k] = cumul_means[k-1] + k * histogram[k]

    # Global mean
    mg = cumul_means[-1]

    # In between class variances
    sigma = np.zeros(histogram.shape)
    for k in xrange(256):
        P1k = cumul_sums[k]
        mk = cumul_means[k]
        sigma[k] = (mg * P1k - mk) ** 2.0 / (P1k * (1.0 - P1k)) if P1k != 0 else -float('inf')

    # Otsu threshold
    k_max = np.argmax(sigma, axis=0)

    # Separability measure
    nu = sigma[k_max] / np.var(img)

    output = np.zeros(img.shape, np.uint8)
    output[img > k_max] = 255

    return output


# Main
if __name__ == "__main__":

    # Available args
    parser = argparse.ArgumentParser(description='Image segmentation - Otsu\'s method of thresholding segmentation')

    parser.add_argument('-o', '--otsu', action='store_true', help='Use Otsu\'s method of thresholding segmentation')
    parser.add_argument('-g', '--gthreshold', action='store_true', help='Use global thresholding')

    parser.add_argument('image_path', type=str, help='Image path')


    # Parse args
    args = parser.parse_args()

    image = Img.load(args.image_path)

    # Otsu
    if args.otsu:
        o = otsu(image)
        Img.show(o)

    # Global thresholding
    if args.gthreshold:
        gt = global_thresholding(image)
        Img.show(gt)
