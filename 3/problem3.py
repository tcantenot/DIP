#!/usr/bin/python

import argparse, os, sys
from math import exp, pow, sqrt
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def D(u, v, M, N):
	return sqrt(pow(u - float(M)/2, 2) + pow(v - float(N)/2, 2))


def ideal_low_pass(D0, M, N):
	def ideal(u, v):
		return 1. if D(u, v, M, N) <= D0 else 0.
	return ideal

def ideal_high_pass(D0, M, N):
	def ideal(u, v):
		return 1. - ideal_low_pass(D0, M, N)(u, v)
	return ideal

def butterworth_low_pass(D0, n, M, N):
	def butterworth(u, v):
		return float(1) / (1 + pow(D(u, v, M, N)/D0, 2*n))
	return butterworth

def butterworth_high_pass(D0, n, M, N):
	def butterworth(u, v):
		return 1. - butterworth_low_pass(D0, n, M, N)(u, v)
	return butterworth


def gaussian_low_pass(D0, M, N):
	def gauss(u, v):
		return exp(-pow(D(u,v,M,N), 2)/(2 * pow(D0, 2)))
	return gauss


def gaussian_high_pass(D0, M, N):
	def gauss(u, v):
		return 1. - gaussian_low_pass(D0, M, N)(u, v)
	return gauss

def apply_filter(fft, filter):
    return np.asarray([x * filter(i[0], i[1]) for i, x in np.ndenumerate(fft)]).reshape(fft.shape)

# Create and return a new image of size (w, h) from the given linear array of pixels
def new_image(pixels, w, h):
    image = Image.new("L", [w, h])
    data = image.load()
    for x in xrange(w):
        for y in xrange(h):
            data[x, y] = pixels[y*w+x]
    return image

def preprocess(data, w, h):
    new_data = [0] * w * h
    for x in xrange(w):
        for y in xrange(h):
            idx = y * w + x
            new_data[idx] = data[idx] * ((-1)**(x+y))
    return new_data

def postprocess(data, w, h):
    return preprocess(data, w, h)


# Main
if __name__ == "__main__":

    # Available args
    parser = argparse.ArgumentParser(description=
        'Combination of spatial enhancement methods'
    )

    parser.add_argument('path', type=str, help='Image path')

    # Parse args
    args = parser.parse_args()

    # Image path
    image_path = args.path

    # Check that the image exists
    if not os.path.isfile(image_path):
        print "Could not find image '{}'".format(image_path)
        sys.exit(-1)

    # Open image
    image = Image.open(image_path)
    if image == None:
        print "Failed to open image '{}'".format(image_path)
        sys.exit(-2)

    image = image.convert('L')
    image.show()

    M, N = image.size
    print "Image size: ({}, {})".format(M, N)

    data = image.getdata()

    data = preprocess(data, M, N)

    data = np.asarray(data).reshape((M, N))

    fft = np.fft.fft2(data)

    filter = ideal_low_pass(10, M, N)
    filter = ideal_high_pass(10, M, N)
    filter = butterworth_low_pass(10, 2, M, N)
    filter = gaussian_low_pass(10, M, N)


    array = apply_filter(fft, filter)


    enhanced_img = np.fft.ifft2(np.asarray(array).reshape((M, N))).ravel()

    #real_img = [sqrt(x.real**2 + x.imag**2) for x in enhanced_img]
    real_img = [x.real for x in enhanced_img]
    print "Real image: ({}, {})".format(len(real_img), 0)

    real_img = postprocess(real_img, M, N)

    final_img = new_image(real_img, M, N)

    final_img.show()
