#!/usr/bin/python

import argparse, os, sys
from math import exp, pow, sqrt
from PIL import Image
import numpy as np


# Euclidean distance from the center of the image
def D(u, v, M, N):
	return sqrt(pow(u - float(M)/2, 2) + pow(v - float(N)/2, 2))


# Low-pass Ideal filter
def ideal_low_pass(D0, M, N):
	def ideal(u, v):
		return 1. if D(u, v, M, N) <= D0 else 0.
	return ideal

# High-pass Ideal filter
def ideal_high_pass(D0, M, N):
	def ideal(u, v):
		return 1. - ideal_low_pass(D0, M, N)(u, v)
	return ideal


# Low-pass Butterworth filter
def butterworth_low_pass(D0, n, M, N):
	def butterworth(u, v):
		return float(1) / (1 + pow(D(u, v, M, N)/D0, 2*n))
	return butterworth

# High-pass Butterworth filter
def butterworth_high_pass(D0, n, M, N):
	def butterworth(u, v):
		return 1. - butterworth_low_pass(D0, n, M, N)(u, v)
	return butterworth


# Low-pass Gaussian filter
def gaussian_low_pass(D0, M, N):
	def gauss(u, v):
		return exp(-pow(D(u,v,M,N), 2)/(2 * pow(D0, 2)))
	return gauss

# High-pass Gaussian filter
def gaussian_high_pass(D0, M, N):
	def gauss(u, v):
		return 1. - gaussian_low_pass(D0, M, N)(u, v)
	return gauss


# Apply the filter to the given FFT data
def apply_filter(fft, filter):
    return np.asarray([x * filter(i[0], i[1]) for i, x in np.ndenumerate(fft)]).reshape(fft.shape)

# Preprocessing
def preprocess(data, w, h):
    new_data = [0] * w * h
    for x in xrange(w):
        for y in xrange(h):
            idx = y * w + x
            new_data[idx] = data[idx] * ((-1)**(x+y))
    return new_data

# Postprocessing
def postprocess(data, w, h):
    return preprocess(data, w, h)


# Create and return a new image of size (w, h) from the given linear array of pixels
def new_image(pixels, w, h):
    image = Image.new("L", [w, h])
    data = image.load()
    for x in xrange(w):
        for y in xrange(h):
            data[x, y] = pixels[y*w+x]
    return image


# Main
if __name__ == "__main__":

    # Available args
    parser = argparse.ArgumentParser(description=
        'Filtering in the frequency domain'
    )

    parser.add_argument('image_path', type=str, help='Image path')

    filter_types = parser.add_argument_group('Filter types', 'Type of the filter to apply')
    filter_types.add_argument('--ideal', action='store_true', help='Use the Ideal filter')
    filter_types.add_argument('--butterworth', action='store_true', help='Use the Butterworth filter')
    filter_types.add_argument('--gaussian', action='store_true', help='Use the Gaussian filter')

    filter_pass_types = parser.add_mutually_exclusive_group(required=True)
    filter_pass_types.add_argument('--low', action='store_true', help='Low pass')
    filter_pass_types.add_argument('--high', action='store_true', help='High pass')

    prepostprocess = parser.add_argument('--npp', action='store_true',
        help='Disable the pre- and post-processing of the image ((-1)^(x+y))'
    )

    parser.add_argument('-d', dest='d', type=float, default=10, help='"D0" parameter: cutoff frequency')
    parser.add_argument('-n', dest='n', type=float, default=2, help='Butterworth filter order')

    # Parse args
    args = parser.parse_args()

    # Image path
    image_path = args.image_path

    # Filter parameters
    D0 = args.d
    n = args.n

    # Check that the image exists
    if not os.path.isfile(image_path):
        print "Could not find image '{}'".format(image_path)
        sys.exit(-1)

    # Open image
    image = Image.open(image_path)
    if image == None:
        print "Failed to open image '{}'".format(image_path)
        sys.exit(-2)

    # Make sure the image is a gray scale image
    image = image.convert('L')
    image.show()

    M, N = image.size

    # Select filter
    filter = None
    if args.ideal:
        if args.low:
            filter = ideal_low_pass(D0, M, N)
        elif args.high:
            filter = ideal_high_pass(D0, M, N)
    elif args.butterworth:
        if args.low:
            filter = butterworth_low_pass(D0, n, M, N)
        elif args.high:
            filter = butterworth_high_pass(D0, n, M, N)
    elif args.gaussian:
        if args.low:
            filter = gaussian_low_pass(D0, M, N)
        elif args.high:
            filter = gaussian_high_pass(D0, M, N)
    else:
        print "No filter selected"
        sys.exit(-3)

    # Image's pixels
    data = image.getdata()

    # Preprocess the data: multiply by (-1)^(x+y)
    if not args.npp:
        data = preprocess(data, M, N)

    # 2D Fast Fourier Transform
    fft = np.fft.fft2(np.asarray(data).reshape((M, N)))

    # Apply the selected filter
    array = apply_filter(fft, filter)

    # 2D Inverse Fast Fourier Transform
    enhanced_img = np.fft.ifft2(np.asarray(array).reshape((M, N))).ravel()

    # Retrieve the real part of the IFFT
    #real_img = [sqrt(x.real**2 + x.imag**2) for x in enhanced_img]
    real_img = [x.real for x in enhanced_img]

    # Postprocess the data: multiply by (-1)^(x+y)
    if not args.npp:
        real_img = postprocess(real_img, M, N)

    # Final image
    final_img = new_image(real_img, M, N)
    final_img.show()
