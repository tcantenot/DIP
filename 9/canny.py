#!/usr/bin/python

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

    #Img.show(g_NH)
    #Img.show(g_NL)

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

    #Img.show(np.array(((output - g_NH) > 0) * 255, dtype=np.uint8))

    return output


# Main
if __name__ == "__main__":

    # Available args
    parser = argparse.ArgumentParser(description='Image segmentation - Canny edge detector')

    parser.add_argument('image_path', type=str, help='Image path')

    edge_masks = parser.add_mutually_exclusive_group(required=True)
    edge_masks.add_argument('--roberts', action='store_true', help='Use Roberts mask')
    edge_masks.add_argument('--sobel', action='store_true', help='Use Sobel mask')
    edge_masks.add_argument('--prewitt', action='store_true', help='Use Prewitt mask')

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

    parser.add_argument('--dump', type=str, default=None, help='Dump the magnitudes array')

    # Parse args
    args = parser.parse_args()

    image = Img.load(args.image_path)

    magnitudes = None
    if args.m is None:

        image = np.array(image, dtype=int)
        shape = image.shape

        # Gaussian smoothing
        print "Smoothing input image..."
        smoothed = np.array(smooth_gauss(image, args.sigma), dtype=np.uint8)
        #Img.show(smoothed)

        mask = None
        if args.roberts: mask = 'roberts'
        elif args.sobel: mask = 'sobel'
        elif args.prewitt: mask = 'prewitt'

        # Gradient magnitudes
        print "Computing gradient magnitudes using the {} mask...".format(mask)
        M, gx, gy = gradient_magnitude(smoothed, mask)
        #Img.show(M * 255)

        # Gradient angles
        print "Computing gradient angles..."
        alpha = to_degree(gradient_angle(gx, gy))

        # Non-maxima suppression
        print "Performing non-maxima suppression..."
        magnitudes = non_maxima_suppression(image, M, alpha)
        #Img.show(magnitudes)

        # Normalization
        print "Normalizing magnitudes..."
        magnitudes *= 1.0 / np.max(magnitudes)
        #Img.show(magnitudes)

        # Dump the magnitudes into a file
        if args.dump is not None:
            magnitudes.dump(args.dump)

    else:
        print "Loading precomputed post-non-maxima-suppression magnitudes..."
        magnitudes = np.load(args.m)
        Img.show(magnitudes * 255)

    #print np.min(magnitudes), np.max(magnitudes)
    #Img.show(np.array((magnitudes > args.th) * 255, dtype=np.int8))

    print "Performing hysteresis thresholding..."
    final_img = hysteresis_thresholding(magnitudes, args.th, args.tl)

    Img.show(np.array((final_img > 0) * 255, dtype=np.int8))
