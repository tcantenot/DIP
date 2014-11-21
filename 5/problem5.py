#!/usr/bin/python

import argparse, os, sys
from math import exp, pow, sqrt
import math
import cmath
from PIL import Image
import numpy as np

from scipy import misc


def blurring_filter(a, b, T, shape):
    m, n = shape
    u, v = np.ogrid[0.:m, 0.:n]
    u[0, 0], v[0, 0] = 0.00001, 0.00001
    return T/(np.pi*(u*a+v*b)) * np.sin(np.pi*(u*a+v*b)) * np.exp(-1j*np.pi*(u*a+v*b))

def inv_blurring_filter(a, b, T, shape):
    return 1./blurring_filter(a, b, T, shape)

def wiener_filter(H):
    K = 0.1
    return H.T / ((H * H) + K)


# Gaussian noise
def gaussian_noise(mu, sigma, shape):
    return np.random.normal(mu, sigma, shape)

# Preprocessing
def preprocess(data):
    x, y = np.ogrid[0:data.shape[0], 0:data.shape[1]]
    return data * ((-1)**(x+y))

# Postprocessing
def postprocess(data):
    return preprocess(data)

# Scale the data between 0 and 255
def scale_data(data):
    min_value = np.min(data)
    scaled_data = data - np.full(data.shape, min_value)
    max_value = np.max(scaled_data)
    scaled_data = scaled_data * np.full(data.shape, 255./max_value)
    return scaled_data


def extract_real(img):
    return np.array([x.real for _, x in np.ndenumerate(img)]).reshape(M, N)

# Create and return a new image of size (w, h) from the given linear array of pixels
def new_image(pixels, w, h):
    image = Image.new("L", [w, h])
    data = image.load()
    for x in xrange(w):
        for y in xrange(h):
            data[x, y] = pixels[y*w+x]
    return image


def show(img_data):
    w, h = img_data.shape
    new_image(img_data.ravel(), w, h).show()

def show_fourier(fft_data):
    fft_img = fft_data.ravel()
    fft_img[0] = 0
    new_image(fft_img, M, N).show()


def process_image(img, filter, pp=True, scale=True, imshow_f=False):

    # Preprocess the data: multiply by (-1)^(x+y)
    if pp: img = preprocess(img)

    # 2D Fast Fourier Transform
    fft_data = np.fft.fft2(img)
    if imshow_f: show_fourier(fft_data)

    # Apply the filter
    filtered_data = fft_data * filter
    if imshow_f: show_fourier(filtered_data)

    # 2D Inverse Fast Fourier Transform
    filtered_img = np.fft.ifft2(filtered_data)

    # Retrieve the real part of the IFFT
    real_img = extract_real(filtered_img)

    # Postprocess the data: multiply by (-1)^(x+y)
    if pp: real_img = postprocess(real_img)

    # Scale the pixels values
    if scale: real_img = scale_data(real_img)

    return real_img

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

    a, b, T = 0.1, 0.1, 1

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

    # Image's pixels
    data = np.array(image.getdata()).reshape((M, N))

    # Blurring filter
    bf = blurring_filter(a, b, T, data.shape)

    # Apply the blurring filter
    blurred_image = process_image(data, bf, pp=True, imshow_f=False)
    show(blurred_image)


    # Add Gaussian noise
    #blurred_image_w_noise = blurred_image + gaussian_noise(0, 650, blurred_image.shape)
    #blurred_image_w_noise = scale_data(blurred_image_w_noise)
    #show(blurred_image_w_noise)

    # Inverse blurring filter
    #ibf = inv_blurring_filter(a, b, T, data.shape)

    # Apply the inverse blurring filter on the blurred image
    #restored_image = process_image(blurred_image, ibf, imshow_f=False)
    #show(restored_image)

    # Apply the inverse blurring filter on the blurred and noisy image
    #restored_image = process_image(blurred_image_w_noise, ibf, imshow_f=False)
    #show(restored_image)

    # Wiener filter
    wf = wiener_filter(bf)

    # Apply the Wiener deconvolution filter on the blurred image
    restored_image = process_image(blurred_image, wf, imshow_f=False)
    show(restored_image)

    # Apply the Wiener deconvolution filter on the blurred and noisy image
    #restored_image = process_image(blurred_image_w_noise, wf, imshow_f=False)
    #show(restored_image)

    sys.exit(0)
