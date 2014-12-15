#!/usr/bin/python

import argparse, os, sys
from PIL import Image
import numpy as np
from scipy import signal, misc
from scipy.ndimage.filters import gaussian_filter


# Show an image
def show(img_data):
    Image.fromarray(img_data).show()

# Gaussian smoothing
def smooth_gauss(img, sigma):
    return gaussian_filter(img, np.sqrt(sigma))

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
    show(np.array(smoothed, dtype=np.uint8))
    laplacian = signal.convolve2d(smoothed, laplacian_mask, mode='same')

    return zero_crossing(laplacian)


# Masks

roberts_masks = (
    np.array([[0, -1], [1, 0]]), \
    np.array([[-1, 0], [0, 1]])
)

prewitt_masks = (
    np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]), \
    np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
)

sobel_masks = (
    np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]), \
    np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
)


# Compute the gradient magnitudes using a mask
def gradient_magnitude(img, mask_type='roberts'):

    mask_types = {'roberts': roberts_masks, 'prewitt': prewitt_masks, 'sobel': sobel_masks}

    mask_x, mask_y = mask_types[mask_type]

    img = np.asarray(img, dtype=np.int)

    gx = np.array(signal.convolve2d(img, mask_x, mode='same'), dtype=np.int)
    gy = np.array(signal.convolve2d(img, mask_y, mode='same'), dtype=np.int)

    M = np.sqrt(gx*gx + gy*gy)
    #M *= 255.0 / np.max(M) # Normalize
    M *= 1.0 / np.max(M) # Normalize

    return M, gx, gy

# Compute the gradient angles
def gradient_angle(gx, gy):
    return np.arctan2(gy, gx)

def to_degree(input):
    return input * 180.0 / np.pi

# Perform non-maxima suppression
def non_maxima_suppression(img, magnitudes, angles):
    """
        img:        Input image
        magnitudes: Gradient magnitudes
        angles:     Gradient angles
    """

    assert img.shape == magnitudes.shape, \
        "Image and gradient magnitudes must have the same shape: {} != {}".format(img.shape, magnitudes.shape)
    assert img.shape == angles.shape, \
        "Image and gradient angles must have the same shape: {} != {}".format(img.shape, angles.shape)

    M, N = img.shape

    def get(x, y, array):
        return None if x < 0 or x >= M or y < 0 or y >= N else array[x, y]

    # Rounded gradient angle = 0 degree
    def get_e_w_magnitudes(x, y):
        east = get(x+1, y, magnitudes)
        west = get(x-1, y, magnitudes)
        return east, west

    # Rounded gradient angle = 90 degree
    def get_n_s_magnitudes(x, y):
        north = get(x, y-1, magnitudes)
        south = get(x, y+1, magnitudes)
        return north, south

    # Rounded gradient angle = 135 degree
    def get_nw_se_magnitudes(x, y):
        nw = get(x-1, y-1, magnitudes)
        se = get(x+1, y+1, magnitudes)
        return nw, se

    # Rounded gradient angle = 45 degree
    def get_ne_sw_magnitudes(x, y):
        ne = get(x+1, y-1, magnitudes)
        sw = get(x-1, y+1, magnitudes)
        return ne, sw

    def interpolate_magnitude(angle, a_min, a_max, m_min, m_max):
        i = (angle - a_min) / (a_max - a_min)
        return (1 - i) * m_min + i * m_max

    output_magnitudes = np.zeros(img.shape)

    for (x, y), n in np.ndenumerate(img):

        if x == 0 or x == (M-1) or y == 0 or y == (N-1): continue

        # Gradient magnitude
        m = magnitudes[x, y]

        # Gradient angle
        angle = np.abs(angles[x, y])

        if 0 <= angle <= 45:

            east, west = get_e_w_magnitudes(x, y)
            ne, sw = get_ne_sw_magnitudes(x, y)
            m1 = interpolate_magnitude(angle, 0, 45, min(east, ne), max(east, ne))
            if m >= m1:
                m2 = interpolate_magnitude(angle, 0, 45, min(west, sw), max(west, sw))
                if m >= m2:
                    output_magnitudes[x, y] = m

        elif 45 < angle <= 90:

            ne, sw = get_ne_sw_magnitudes(x, y)
            north, south = get_n_s_magnitudes(x, y)
            m1 = interpolate_magnitude(angle, 45, 90, min(north, ne), max(north, ne))
            if m >= m1:
                m2 = interpolate_magnitude(angle, 45, 90, min(south, sw), max(south, sw))
                if m >= m2:
                    output_magnitudes[x, y] = m

        elif 90 < angle <= 135:

            north, south = get_n_s_magnitudes(x, y)
            nw, se = get_nw_se_magnitudes(x, y)
            m1 = interpolate_magnitude(angle, 90, 135, min(north, nw), max(north, nw))
            if m >= m1:
                m2 = interpolate_magnitude(angle, 90, 135, min(south, se), max(south, se))
                if m >= m2:
                    output_magnitudes[x, y] = m

        elif 135 < angle <= 180:

            nw, se = get_nw_se_magnitudes(x, y)
            east, west = get_e_w_magnitudes(x, y)
            m1 = interpolate_magnitude(angle, 135, 180, min(east, se), max(east, se))
            if m >= m1:
                m2 = interpolate_magnitude(angle, 135, 180, min(west, nw), max(west, nw))
                if m >= m2:
                    output_magnitudes[x, y] = m

    return output_magnitudes

# Hysteresis thresholding
def hysteresis_thresholding(magnitudes, T_H, T_L):
    """
    magnitudes: Gradient magnitudes
    T_H:        High threshold
    T_L:        Low threshold
    """

    def thresholding(input, threshold):
        return np.vectorize(lambda x: x if x >= threshold else 0, otypes=[input.dtype])(input)

    g_NH = thresholding(magnitudes, T_H)
    g_NL = thresholding(magnitudes, T_L)

    g_NL = g_NL - g_NH

    #show(g_NH)
    #show(g_NL)

    g_NL_valid = np.zeros(g_NL.shape)

    neighbors = np.array([
        [+0, +0],
        [-1, +0],
        [+1, +0],
        [+0, -1],
        [+0, +1],
        [-1, -1],
        [-1, +1],
        [+1, -1],
        [+1, +1],
    ]);

    M, N = magnitudes.shape

    for (x, y), p in np.ndenumerate(g_NH):

        if p == 0: continue # Non-edge pixel
        if x == 0 or x == (M-1) or y == 0 or y == (N-1): continue

        for offset in neighbors:
            dx, dy = offset[0], offset[1]
            v = g_NL[x+dx, y+dy]
            if v > 0: g_NL_valid[x+dx, y+dy] = v

    output = g_NH + g_NL_valid

    #show(np.array(((output - g_NH) > 0) * 255, dtype=np.uint8))

    return output

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

def otsu(img):

    M, N = img.shape

    # Normalized histogram
    histogram = np.array([0.0 for _ in xrange(255)], dtype=np.float)
    for _, x in np.ndenumerate(img):
        histogram[x] += 1

    histogram = histogram / (M * N)

    # Cumulative sums
    cumul_sums = np.copy(histogram)
    for k in xrange(1, 255):
        cumul_sums[k] = cumul_sums[k-1] + histogram[k]

    # Cumulative means
    cumul_means = np.zeros(histogram.shape)
    for k in xrange(1, 255):
        cumul_means[k] = cumul_means[k-1] + k * histogram[k]

    # Global mean
    mg = cumul_means[-1]

    # In between class variances
    sigma = np.zeros(histogram.shape)
    for k in xrange(255):
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
    parser = argparse.ArgumentParser(description='Image segmentation')

    parser.add_argument('image_path', type=str, help='Image path')

    parser.add_argument('-s', dest='sigma', type=float, default=4,
        help='Variance of the Gaussian smoothing kernel'
    )

    parser.add_argument('-m', type=str, default=None,
        help='Post non-maxima suppression image path'
    )

    parser.add_argument('--th', type=float, default=0.10,
        help='High threshold for hysteresis thresholding'
    )

    parser.add_argument('--tl', type=float, default=0.04,
        help='Low threshold for hysteresis thresholding'
    )

    edge_masks = parser.add_mutually_exclusive_group(required=True)
    edge_masks.add_argument('--roberts', action='store_true', help='Use Roberts mask')
    edge_masks.add_argument('--sobel', action='store_true', help='Use Sobel mask')
    edge_masks.add_argument('--prewitt', action='store_true', help='Use Prewitt mask')

    parser.add_argument('--dump', type=str, default=None, help='Dump the magnitudes array')

    parser.add_argument('--cheat', action='store_true', help='Use cv2.Canny')

    # Parse args
    args = parser.parse_args()

    image = Image.open(args.image_path).convert('L')
    data = np.array(image, dtype=int)

    # Otsu
    o = otsu(data)
    show(o)
    sys.exit(0)

    # Global thresholding
    gt = global_thresholding(data) #, args.epsilon)
    show(gt)
    sys.exit(0)

    # Marr-Hildreth
    mh = marr_hildreth(data, args.sigma)
    misc.imsave('marr_hildreth.tif', mh)
    show(mh)

    sys.exit(0)

    if args.cheat:
        import cv2
        img = cv2.imread(args.image_path, 0)
        show(cv2.Canny(img, 100, 200))
        sys.exit(0)

    magnitudes = None
    if args.m is None:

        image = Image.open(args.image_path).convert('L')
        data = np.array(image, dtype=int)
        shape = data.shape

        # Gaussian smoothing
        print "Smoothing input image..."
        smoothed = np.array(smooth_gauss(data, args.sigma), dtype=np.uint8)
        #show(smoothed)

        mask = None
        if args.roberts: mask = 'roberts'
        elif args.sobel: mask = 'sobel'
        elif args.prewitt: mask = 'prewitt'

        # Gradient magnitudes
        print "Computing gradient magnitudes using the {} mask...".format(mask)
        M, gx, gy = gradient_magnitude(smoothed, mask)
        #show(M * 255)

        # Gradient angles
        print "Computing gradient angles..."
        alpha = to_degree(gradient_angle(gx, gy))

        # Non-maxima suppression
        print "Performing non-maxima suppression..."
        magnitudes = non_maxima_suppression(data, M, alpha)
        #show(magnitudes)

        # Normalization
        print "Normalizing magnitudes..."
        magnitudes *= 1.0 / np.max(magnitudes)
        #magnitudes *= 255.0 / np.max(magnitudes)
        #show(magnitudes)

        # Dump the magnitudes into a file
        if args.dump is not None:
            magnitudes.dump(args.dump)

    else:
        print "Loading precomputed post-non-maxima-suppression magnitudes..."
        magnitudes = np.load(args.m)
        show(magnitudes * 255)

    #print np.min(magnitudes), np.max(magnitudes)
    #show(np.array((magnitudes > args.th) * 255, dtype=np.int8))

    print "Performing hysteresis thresholding..."
    final_img = hysteresis_thresholding(magnitudes, args.th, args.tl)

    show(np.array((final_img > 0) * 255, dtype=np.int8))