#!/usr/bin/python

import argparse, os, sys
from PIL import Image
from matplotlib import pyplot

from noise import GaussianNoise, UniformNoise
from meanfilter import ArithmeticMeanFilter, GeometricMeanFilter
from meanfilter import HarmonicMeanFilter, ContraHarmonicMeanFilter
from osfilter import MedianFilter, MaxFilter, MinFilter, MidpointFilter, AlphaTrimmedFilter


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

# Add noise to the given image using the given noise generator
def apply_noise(image, noise_gen):
    w, h = image.size
    pixels = image.getdata()
    data = [0 for i in xrange(w * h)]

    for x in xrange(w):
        for y in xrange(h):
            data[y*w+x] = pixels[y*w+x] + noise_gen.sample(x, y)

    scale_pixel_data(data, w, h)

    return new_image(data, w, h)

# Apply the given restoration filter to the given image
def apply_restoration_filter(image, filter):
    w, h = image.size
    pixels = image.getdata()
    data = [0 for i in xrange(w * h)]
    for x in xrange(w):
        for y in xrange(h):
            data[y*w+x] = filter.apply(x, y, w, h, pixels)

    return new_image(data, w, h)


# Main
if __name__ == "__main__":

    # Available args
    parser = argparse.ArgumentParser(description=
        'Noise generation and noise reduction'
    )

    parser.add_argument('image_path', type=str, help='Image path')

    noise_types = parser.add_mutually_exclusive_group(required=True)
    noise_types.add_argument('--uniform', action='store_true', help='Use Uniform noise')
    noise_types.add_argument('--gaussian', action='store_true', help='Use Gaussian noise')

    parser.add_argument('--histogram', action='store_true', help='Show histogram of the noisy image')

    mf_types = parser.add_argument_group('Mean filter types', 'Type of Mean filter to apply')
    mf_types.add_argument('--arithmetic', action='store_true', help='Use Arithmetic mean filter')
    mf_types.add_argument('--geometric', action='store_true', help='Use Geometric mean filter')
    mf_types.add_argument('--harmonic', action='store_true', help='Use Harmonic mean filter')
    mf_types.add_argument('--contraharmonic', action='store_true', help='Use Contra-harmonic mean filter')

    parser.add_argument('-q', dest='q', type=float, default=1, \
        help='"Q" parameter for the Contrat-harmonic mean filter'
    )

    osf_types = parser.add_argument_group(
        'Order-statistic filter types',
        'Type of Order-statistic filter to apply'
    )
    osf_types.add_argument('--median', action='store_true', help='Use Median filter')
    osf_types.add_argument('--max', action='store_true', help='Use Max filter')
    osf_types.add_argument('--min', action='store_true', help='Use Min filter')
    osf_types.add_argument('--midpoint', action='store_true', help='Use Midpoint filter')
    osf_types.add_argument('--alpha', action='store_true', help='Use Alpha-trimmed filter')

    parser.add_argument('-d', dest='d', type=int, default=2, \
        help='"D" parameter for the Alpha-trimmed filter'
    )


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

    if args.histogram:
        fig = pyplot.figure()
        fig.suptitle("Histogram of orginal image")
        pyplot.plot(image.histogram())
        pyplot.show()


    noise_image = None

    # Noise generators

    if args.gaussian:
        noise_image = apply_noise(image, GaussianNoise(0, 20))
    elif args.uniform:
        noise_image = apply_noise(image, UniformNoise())

    noise_image.show()

    if args.histogram:
        fig = pyplot.figure()
        fig.suptitle("Histogram of original image with noise")
        pyplot.plot(noise_image.histogram())
        pyplot.show()


    # Mean filters

    if args.arithmetic:
        restored_image = apply_restoration_filter(noise_image, ArithmeticMeanFilter(3,3))
        restored_image.show()

    if args.geometric:
        restored_image = apply_restoration_filter(noise_image, GeometricMeanFilter(3,3))
        restored_image.show()

    if args.harmonic:
        restored_image = apply_restoration_filter(noise_image, HarmonicMeanFilter(3,3))
        restored_image.show()

    if args.contraharmonic:
        restored_image = apply_restoration_filter(noise_image, ContraHarmonicMeanFilter(3,3,args.q))
        restored_image.show()


    # Order-statistic filters

    if args.median:
        restored_image = apply_restoration_filter(noise_image, MedianFilter(3,3))
        restored_image.show()

    if args.max:
        restored_image = apply_restoration_filter(noise_image, MaxFilter(3,3))
        restored_image.show()

    if args.min:
        restored_image = apply_restoration_filter(noise_image, MinFilter(3,3))
        restored_image.show()

    if args.midpoint:
        restored_image = apply_restoration_filter(noise_image, MidpointFilter(3,3))
        restored_image.show()

    if args.alpha:
        restored_image = apply_restoration_filter(noise_image, AlphaTrimmedFilter(3,3,args.d))
        restored_image.show()

    if args.histogram:
        fig = pyplot.figure()
        fig.suptitle("Histogram of restored image")
        pyplot.plot(restored_image.histogram())
        pyplot.show()
