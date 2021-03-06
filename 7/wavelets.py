import sys
sys.path.append('..')

import argparse
import numpy as np
from scipy import misc

from common import Img

# Wavelets

haar_h0 = np.array([
    1 / np.sqrt(2),
    1 / np.sqrt(2)
])

haar_h1 = np.array([
    1 / np.sqrt(2),
    -1 / np.sqrt(2)
])

daubechies_g0 = np.array([
    0.23037781, 0.71484657, 0.63088076, -0.02798376,
    -0.18703481, 0.03084138, 0.03288301, -0.0105940
])

symlet_g0 = np.array([
    0.0322, -0.0126, -0.0992, 0.2979,
    0.8037, 0.4976, -0.0296, -0.0758
])

cohen_h0 = np.array([
    0, 0.0019, -0.0019, -0.017, 0.0119, 0.0497, -0.0773,
    -0.0941, 0.4208, 0.8259, 0.4208, -0.0941, -0.0773,
    0.0497, 0.0119, -0.017, -0.0019, 0.0010
])

cohen_h1 = np.array([
    0, 0, 0, 0.0144, -0.0145, -0.0787,
    0.0404, 0.4178, -0.7589, 0.4178, 0.0404,
    -0.0787, -0.0145, 0.0144, 0, 0, 0, 0
])


def encode_wavelet(img, kernel, level):
    output = np.copy(img)
    for _ in xrange(level):
        output = np.dot(kernel.T, np.dot(output, kernel))
    return output

def decode_wavelet(img, kernel, level):
    output = np.copy(img)
    for _ in xrange(level):
        output = np.dot(kernel, np.dot(output, kernel.T))
    return output

def haar_kernel(shape):
    M, _ = shape
    return (np.vstack((np.kron(np.eye(M / 2), haar_h0), np.kron(np.eye(M / 2), haar_h1))).T)

def wavelet_kernel(h0, h1, shape):
    M, _ = shape
    fwd = np.zeros(shape)
    for i in xrange(0, M / 2):
        for j in xrange(len(h0)):
            fwd[i, (2 * i + j) % M] = h0[j]
            fwd[(M / 2) + i, (2 * i + j) % M] = h1[j]
    return fwd.T

def daubechies_kernel(shape):
    h0 = daubechies_g0[::-1]
    g1 = [h * (-1.)**i for i, h in enumerate(h0)]
    h1 = g1[::-1]
    return wavelet_kernel(h0, h1, shape)

def symlet_kernel(shape):
    h0 = symlet_g0[::-1]
    g1 = [h * (-1.)**i for i, h in enumerate(h0)]
    h1 = g1[::-1]
    return wavelet_kernel(h0, h1, shape)

def cohen_kernel(shape):
    return wavelet_kernel(cohen_h0, cohen_h1, shape)


if __name__ == '__main__':

    # Available args
    parser = argparse.ArgumentParser(description='Transform image compression')

    parser.add_argument('image_path', type=str, help='Image path')

    wavelets = parser.add_mutually_exclusive_group()
    wavelets.add_argument('--haar', action='store_true', help='Use Haar wavelets')
    wavelets.add_argument('--daub', action='store_true', help='Use Daubechies wavelets')
    wavelets.add_argument('--symlet', action='store_true', help='Use Symlet wavelets')
    wavelets.add_argument('--cohen', action='store_true', help='Use Cohen wavelets')

    parser.add_argument('-l', dest='level', type=int, default=1, help='Number of Wavelet levels')

    parser.add_argument('-t', dest='threshold', type=float, default=0, help='Threshold use to truncate')


    # Parse args
    args = parser.parse_args()

    img = Img.load(args.image_path, np.float)
    level = args.level

    kernel = None
    if args.haar:
        kernel = haar_kernel(img.shape)
    elif args.daub:
        kernel = daubechies_kernel(img.shape)
    elif args.symlet:
        kernel = symlet_kernel(img.shape)
    elif args.cohen:
        kernel = cohen_kernel(img.shape)

    encoded = encode_wavelet(img, kernel, level)
    Img.show(encoded)

    threshold = args.threshold
    encoded[encoded < threshold] = 0.

    decoded = decode_wavelet(encoded, kernel, level)

    Img.show(Img.scale(decoded), np.uint8)
    Img.show_diff(decoded, img)
