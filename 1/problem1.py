#!/usr/bin/python

import sys
sys.path.append('..')

import argparse, os
import numpy as np
from matplotlib import pyplot

from common import Img


# Hexadecimal encoding from a RGB triplet
def hexencode(r, g, b):
    return '#%02x%02x%02x' % (r, g, b)

# Compute and return the histogram of the given image
def histogram(image):
    hist = np.zeros((256, 1))
    for _, c in np.ndenumerate(image): hist[c] += 1
    return hist

# Compute and return the cumulated histogram from the given histogram
def cumulated_histogram(hist):
    cumul_hist_plot = np.zeros((256, 1))
    for i, h in enumerate(hist):
        cumul_hist_plot[i] = (0 if i == 0 else cumul_hist_plot[i-1]) + hist[i]
    return cumul_hist_plot


# Display the given histogram
def show_histogram(hist, title=None):

    figure, hist_plot = pyplot.subplots()
    figure.patch.set_facecolor(hexencode(255, 255, 255))

    if title != None: figure.suptitle(title, fontsize=18)

    for i, count in enumerate(hist):
        color = hexencode(i, i, i)
        hist_plot.bar(i, count, color=color, edgecolor=color, facecolor=color, label='Histogram')

    hist_plot.patch.set_facecolor(hexencode(217, 224, 255))
    hist_plot.set_xlim(0, 255)

    cumul_hist_plot = hist_plot.twinx()
    cumul_hist_color = hexencode(255, 0, 0)
    cumul_hist_plot.plot(xrange(256), cumulated_histogram(hist), \
        color=cumul_hist_color, label='Cumulative histogram')
    cumul_hist_plot.set_xlim(0, 255)
    cumul_hist_plot.legend(loc='upper right')
    cumul_hist_plot.spines['right'].set_color(cumul_hist_color)
    cumul_hist_plot.tick_params(axis='y', colors=cumul_hist_color)

    pyplot.draw()


def histogram_equalization(image, hist):

    M, N = image.shape

    # Normalized histogram (probability)
    norm_hist = hist / float(M * N)

    # Generate the cumulative distribution function lookup table
    L = 256
    cdfs = np.zeros((L, 1))
    sums = np.zeros((L, 1))
    for i in xrange(L):
        sums[i] = (sums[i-1] if i > 0 else 0) + norm_hist[i]
        cdfs[i] = round(sums[i] * (L-1))

    # Generate the new enhanced image
    return np.vectorize(lambda x: cdfs[x])(image)


# Main
if __name__ == "__main__":

    # Available args
    parser = argparse.ArgumentParser(description=
        'Generate the histogram of a gray-scale image and \
        perform histogram equalization'
    )

    parser.add_argument('image_path', type=str, help='Image path')

    # Parse args
    args = parser.parse_args()

    # Load image
    image_path = args.image_path
    image = Img.load(image_path)

    # Compute the histogram of the image
    print "Computing histogram of '{}'...".format(image_path)
    hist = histogram(image)
    Img.show(image)
    show_histogram(hist, "Histogram of '{}'".format(image_path))

    # Perform the histogram equalization and generate the enhanced image
    print "Performing histogram equalization..."
    new_image = histogram_equalization(image, hist)

    # Compute the histogram of the new image
    print "Computing histogram of enhanced '{}'...".format(image_path)
    new_hist = histogram(new_image)
    Img.show(new_image)
    show_histogram(new_hist, "Histogram of enhanced '{}'".format(image_path))

    # Ensure that the matplotlib windows stay open
    pyplot.show()
