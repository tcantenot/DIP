#!/usr/bin/python

import argparse, os, sys
from math import exp, pow, sqrt
import math
import cmath
from PIL import Image
import numpy as np

# Blurring function
def blurring(a, b, T):
    def _blurring(u, v):
        if u == 0 and v == 0 or \
           a == 0 and b == 0 or \
           a == 0 and v == 0 or \
           b == 0 and u == 0:
            return 1
        else:
            u, v = float(u), float(v)
            return float(T) / (cmath.pi * (u*a + v*b)) * cmath.sin(cmath.pi * (u*a + v*b)) \
                    * cmath.exp(-1j*cmath.pi*(u*a+v*b))

    return _blurring


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

def scale_filtered_data(pixels, w, h):

    min_value = min(pixels)
    for x in xrange(w):
        for y in xrange(h):
            pixels[y*w+x] -= min_value

    max_value = max(pixels)

    for x in xrange(w):
        for y in xrange(h):
            pixels[y*w+x] = pixels[y*w+x] * (float(255) / max_value)

    return pixels



# Main
if __name__ == "__main__":

    # Available args
    parser = argparse.ArgumentParser(description=
        'Filtering in the frequency domain'
    )

    parser.add_argument('image_path', type=str, help='Image path')

    # Parse args
    args = parser.parse_args()

    # Image path
    image_path = args.image_path

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
    filter = blurring(a=0.005, b=0.005, T=1)

    # Image's pixels
    data = image.getdata()

    # Preprocess the data: multiply by (-1)^(x+y)
    #data = preprocess(data, M, N)

    # 2D Fast Fourier Transform
    fft = np.fft.fft2(np.asarray(data).reshape((M, N)))

    # Print the FFT
    fft_data = [x.real for x in fft.ravel()]
    fft_data[0] = 0
    new_image(fft_data, M, N).show()

    # Apply the selected filter
    array = apply_filter(fft, filter)

    # Print the FFT after transformation
    fft_data = [x.real for x in array.ravel()]
    fft_data[0] = 0
    new_image(fft_data, M, N).show()

    # 2D Inverse Fast Fourier Transform
    enhanced_img = np.fft.ifft2(np.asarray(array).reshape((M, N))).ravel()

    # Retrieve the real part of the IFFT
    #real_img = [sqrt(x.real**2 + x.imag**2) for x in enhanced_img]
    real_img = [x.real for x in enhanced_img]

    # Postprocess the data: multiply by (-1)^(x+y)
    #real_img = postprocess(real_img, M, N)
    real_img = scale_filtered_data(real_img, M, N)

    # Final image
    final_img = new_image(real_img, M, N)
    final_img.show()
