#!/usr/bin/python

import argparse, os, sys
from PIL import Image
import numpy as np
from scipy import fftpack

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


# Generator yielding 8x8 subimages
def subimages(img, size):
    w, h = img.shape
    for x in xrange(0, w, size):
        for y in xrange(0, h, size):
            yield img[x:x+size, y:y+size]

# Textbook p598: Zonal mask
def zonal_mask(n, s=8):
    if n <= s:
        mask = np.zeros((s, s))
        for i in xrange(len(mask)):
            for j in reversed(xrange(n-i)):
                mask[i, j] = 1
    elif n <= s * 2:
        mask = np.zeros((n, n))
        for i in xrange(len(mask)):
            for j in reversed(xrange(n-i)):
                mask[i, j] = 1
    else:
        mask = np.ones((s, s))

    return mask[:s, :s]


# Textbook p598: threshold mask
def threshold_mask():
    return np.array([
    1, 1, 0, 1, 1, 0, 0, 0,
    1, 1, 1, 1, 0, 0, 0, 0,
    1, 1, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0 ]).reshape(8, 8)

def apply_mask(subimg, mask):
    return subimg * mask

def dct2(data):
    return fftpack.dct(fftpack.dct(data.T, norm='ortho').T, norm='ortho')

def idct2(data):
    return fftpack.idct(fftpack.idct(data.T, norm='ortho').T, norm='ortho')


def compress_image(img, mask):

    MASK_SIZE = 8

    w, h = img.shape
    chunks = [[] for i in xrange(w/MASK_SIZE)]

    for i, subimg in enumerate(subimages(img, MASK_SIZE)):
        dct_data = dct2(subimg)
        dct_data = apply_mask(dct_data, mask)
        compress_img = idct2(dct_data)
        chunks[i/MASK_SIZE/MASK_SIZE].append(compress_img)

    return chunks

def reconstruct_image(img_chunks, w, h):

    img = np.empty((w, h))
    sx, sy = img_chunks[0][1].shape

    for r, chunks in enumerate(img_chunks):
        for c, ch in enumerate(chunks):
            for i in xrange(sx):
                x = r * sx + i
                for j in xrange(sy):
                    y = c * sy + j
                    img[x, y] = ch[i, j]

    return img


# Main
if __name__ == "__main__":

    # Available args
    parser = argparse.ArgumentParser(description=
        'Filtering in the frequency domain'
    )

    parser.add_argument('image_path', type=str, help='Image path')

    parser.add_argument('--quant', type=str, help='Image path')

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
    data = np.array(image.getdata(), dtype=float).reshape((M, N))

    # Compress the image
    chunks = compress_image(data, zonal_mask(15))

    # Reconstruct the image
    final_img = reconstruct_image(chunks, M, N)

    show(final_img)

    # Difference between original and reconstructed
    show(data - final_img)
