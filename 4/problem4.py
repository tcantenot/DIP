import argparse, os, sys
from math import sqrt, exp, pi
from PIL import Image
import random
import numpy as np
from matplotlib import pyplot

# Gaussian distribution
def gaussian(z, mu=0, sigma=1):
    return (1./sqrt(2.*pi*sigma**2))*exp(-((z-mu)**2)/(2*sigma**2))

def default_gaussian(z, mu=0, sigma=1):
    return random.normalvariate(mu, sigma)

def rayleigh(z, a=0, b=1):
    #return 2./b*(z-a)*exp(-((z-a)**2)/b)*u(z-a)
    pass


def scale(z, min=0, max=255):
    return min + (max - min) * z

# Create and return a new image of size (w, h) from the given linear array of pixels
def new_image(pixels, w, h):
    image = Image.new("L", [w, h])
    data = image.load()
    for x in xrange(w):
        for y in xrange(h):
            data[x, y] = pixels[y*w+x]
    return image

# Scale the pixels data to the range [0, 255]
def scale_pixel_data(pixels, w, h):

    min_value = min(pixels)
    for x in xrange(w):
        for y in xrange(h):
            pixels[y*w+x] -= min_value

    max_value = max(pixels)
    for x in xrange(w):
        for y in xrange(h):
            pixels[y*w+x] = pixels[y*w+x] * (float(255) / max_value)

    return pixels


def apply_noise(image, noise_gen):
    w, h = image.size
    pixels = image.getdata()
    data = [0] * w * h
    for i in xrange(len(data)):
        data[i] = pixels[i] + scale(noise_gen(random.random()))

    scale_pixel_data(data, w, h)

    return new_image(data, w, h)

def arithmetic_mean_filter(x, y, w, h, pixels):

    def get_pixel(x, y):
        p = 0 if x<0 or x>(w-1) or y<0 or y>(h-1) else pixels[y*w+x]
        return float(p)

    mask_w = 3
    mask_h = 3

    value = 0

    for i in xrange(mask_w):
        for j in xrange(mask_h):
            img_x = (x - mask_w / 2 + i)
            img_y = (y - mask_h / 2 + j)
            value += get_pixel(img_x, img_y)

    value /= (float(mask_w) * float(mask_h))

    return value


def apply_restoration_filter(image, filter):
    w, h = image.size
    pixels = image.getdata()
    data = [0] * w * h
    for x in xrange(w):
        for y in xrange(h):
            data[y*w+x] = filter(x, y, w, h, pixels)

    return new_image(data, w, h)


# Main
if __name__ == "__main__":

    # Available args
    parser = argparse.ArgumentParser(description=
        'Noise generation and noise reduction'
    )

    parser.add_argument('image_path', type=str, help='Image path')

    #noise_types = parser.add_argument_group('Noise types', 'Type of noise to apply')
    #noise_types = parser.add_mutually_exclusive_group(required=True)
    #noise_types.add_argument('--uniform', action='store_true', help='Use Uniform noise')
    #noise_types.add_argument('--gaussian', action='store_true', help='Use Gaussian noise')

    #parser.add_argument('-d', dest='d', type=float, default=10, help='"D0" parameter: cutoff frequency')
    #parser.add_argument('-n', dest='n', type=float, default=2, help='Butterworth filter order')

    # Parse args
    args = parser.parse_args()

    # Image path
    image_path = args.image_path

    #if args.uniform:
        #pass
    #elif args.gaussian:
        #pass

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
    pyplot.plot(image.histogram())
    pyplot.draw()

    #noise_image = apply_noise(image, gaussian)
    noise_image = apply_noise(image, default_gaussian)
    noise_image.show()
    pyplot.plot(noise_image.histogram())

    restored_image = apply_restoration_filter(noise_image, arithmetic_mean_filter)
    restored_image.show()
    pyplot.plot(restored_image.histogram())

    pyplot.show()

