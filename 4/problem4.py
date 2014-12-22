#!/usr/bin/python

import sys
sys.path.append('..')

import argparse
import numpy as np
from matplotlib import pyplot

from common import Img

from noise import GaussianNoise, UniformNoise
from meanfilter import ArithmeticMeanFilter, GeometricMeanFilter
from meanfilter import HarmonicMeanFilter, ContraHarmonicMeanFilter
from osfilter import MedianFilter, MaxFilter, MinFilter, MidpointFilter, AlphaTrimmedFilter

# Compute histogram
def histogram(img):
    histogram = np.array([0.0 for _ in xrange(256)], dtype=np.float)
    for _, x in np.ndenumerate(img):
        histogram[x] += 1
    return histogram

# Add noise to the given image using the given noise generator
def apply_noise(image, noise):
    return Img.scale(image + noise)

# Apply the given restoration filter to the given image
def apply_restoration_filter(image, filter):
    w, h = image.shape
    data = np.empty(image.shape)
    for x in xrange(w):
        for y in xrange(h):
            data[x, y] = filter.apply(x, y, image)

    return Img.scale(data)

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

    # Load image
    image = Img.load(args.image_path)

    if args.histogram:
        fig, plot = pyplot.subplots()
        plot.set_xlim(0, 255)
        fig.suptitle("Histogram of orginal image")
        hist = histogram(image)
        pyplot.plot(hist)
        pyplot.show()


    # Noise generators

    noisy_image = None

    if args.gaussian:
        noisy_image = apply_noise(image, GaussianNoise(0, 20, image.shape))
    elif args.uniform:
        noisy_image = apply_noise(image, UniformNoise(0, 128, image.shape))

    Img.show(noisy_image)

    if args.histogram and noisy_image is not None:
        fig, plot = pyplot.subplots()
        plot.set_xlim(0, 255)
        fig.suptitle("Histogram of original image with noise")
        hist = histogram(noisy_image)
        pyplot.plot(hist)
        pyplot.show()


    # Mean filters

    restored_image = None

    if args.arithmetic:
        restored_image = apply_restoration_filter(noisy_image, ArithmeticMeanFilter(3,3))
        Img.show(restored_image)

    if args.geometric:
        restored_image = apply_restoration_filter(noisy_image, GeometricMeanFilter(3,3))
        Img.show(restored_image)

    if args.harmonic:
        restored_image = apply_restoration_filter(noisy_image, HarmonicMeanFilter(3,3))
        Img.show(restored_image)

    if args.contraharmonic:
        restored_image = apply_restoration_filter(noisy_image, ContraHarmonicMeanFilter(3,3,args.q))
        Img.show(restored_image)


    # Order-statistic filters

    if args.median:
        restored_image = apply_restoration_filter(noisy_image, MedianFilter(3,3))
        Img.show(restored_image)

    if args.max:
        restored_image = apply_restoration_filter(noisy_image, MaxFilter(3,3))
        Img.show(restored_image)

    if args.min:
        restored_image = apply_restoration_filter(noisy_image, MinFilter(3,3))
        Img.show(restored_image)

    if args.midpoint:
        restored_image = apply_restoration_filter(noisy_image, MidpointFilter(3,3))
        Img.show(restored_image)

    if args.alpha:
        restored_image = apply_restoration_filter(noisy_image, AlphaTrimmedFilter(3,3,args.d))
        Img.show(restored_image)

    if args.histogram and restored_image is not None:
        fig, plot = pyplot.subplots()
        plot.set_xlim(0, 255)
        fig.suptitle("Histogram of restored image")
        hist = histogram(restored_image)
        pyplot.plot(hist)
        pyplot.show()
