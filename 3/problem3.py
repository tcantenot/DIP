#!/usr/bin/python

import sys
sys.path.append('..')

import argparse
import numpy as np

from common import Img


# Euclidean distance from the center of the image
def D(u, v, M, N):
	return np.sqrt((u - M/2.)**2 + (v - N/2.)**2)


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
		return 1. / (1. + np.power(D(u, v, M, N)/D0, 2*n))
	return butterworth

# High-pass Butterworth filter
def butterworth_high_pass(D0, n, M, N):
	def butterworth(u, v):
		return 1. - butterworth_low_pass(D0, n, M, N)(u, v)
	return butterworth


# Low-pass Gaussian filter
def gaussian_low_pass(D0, M, N):
	def gauss(u, v):
		return exp(-D(u,v,M,N)**2/(2. * D0**2))
	return gauss

# High-pass Gaussian filter
def gaussian_high_pass(D0, M, N):
	def gauss(u, v):
		return 1. - gaussian_low_pass(D0, M, N)(u, v)
	return gauss


# Apply the filter to the given FFT data
def apply_filter(fft, filter):
    return np.array([x * filter(u, v) for (u, v), x in np.ndenumerate(fft)]).reshape(fft.shape)

# Preprocessing
def preprocess(data):
    x, y = np.ogrid[0:data.shape[0], 0:data.shape[1]]
    return data * ((-1)**(x+y))

# Postprocessing
def postprocess(data):
    return preprocess(data)


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

    parser.add_argument('-d', dest='d', type=float, default=10, help='"D0" parameter: cutoff frequency')
    parser.add_argument('-n', dest='n', type=float, default=2, help='Butterworth filter order')

    # Parse args
    args = parser.parse_args()

    # Filter parameters
    D0 = args.d
    n = args.n

    # Load image
    image = Img.load(args.image_path)
    M, N = image.shape

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

    data = image

    # Preprocess the data: multiply by (-1)^(x+y)
    data = preprocess(data)

    # 2D Fast Fourier Transform
    data = np.fft.fft2(data)

    # Apply the selected filter
    data = apply_filter(data, filter)

    # 2D Inverse Fast Fourier Transform
    data = np.fft.ifft2(data)

    # Retrieve the real part of the IFFT
    data = data.real

    # Postprocess the data: multiply by (-1)^(x+y)
    data = postprocess(data)

    # Final image
    final_img = data
    Img.show(final_img)
