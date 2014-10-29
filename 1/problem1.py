#!/usr/bin/python

import argparse, os, sys
from PIL import Image
from matplotlib import pyplot

# Hexadecimal encoding from a RGB triplet
def hexencode(r, g, b):
    return '#%02x%02x%02x' % (r, g, b)


# Compute and return the histogram of the given image
def histogram(image):

    pixels = list(image.getdata())

    # Gray-scale image
    if image.mode == "L":
        hist = [0] * 256
        for c in pixels: hist[c] += 1
        return hist

    # RGB image
    elif image.mode == "RGB":
        hist = [0] * 256
        for c in pixels: hist[c[0]] += 1
        return hist

    # Not supported format
    else: print "Image mode not supported: '{}'".format(img.mode)

    return None


# Compute and return the cumulated histogram from the given histogram
def cumulated_histogram(hist):
    cumul_hist_plot = [0] * len(hist)

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

    # Normalized histogram (probability)
    def p(i, image, hist):
        return hist[i] / float(image.size[0] * image.size[1])

    # Generate the cumulative distribution function lookup table

    L = 256
    cdfs = [0] * L
    for i in xrange(L):
        for j in xrange(i+1):
            cdfs[i] += p(j, image, hist)
        cdfs[i] = round(cdfs[i] * (L-1))


    # Generate the new enhanced image

    w, h = image.size
    pixels = list(image.getdata())

    new_image = Image.new("L", image.size)
    new_pixels = new_image.load()

    for x in xrange(w):
        for y in xrange(h):
            if image.mode == "L": # Grey scale image
                new_pixels[x, y] = cdfs[pixels[y * w + x]]
            elif image.mode == "RGB": # RGB image
                new_pixels[x, y] = cdfs[pixels[y * w + x][0]]

    return new_image


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

    # Compute the histogram of the image
    print "Computing histogram of '{}'...".format(image_path)
    hist = histogram(image)
    image.show()
    show_histogram(hist, "Histogram of '{}'".format(image_path))

    # Perform the histogram equalization and generate the enhanced image
    print "Performing histogram equalization..."
    new_image = histogram_equalization(image, hist)

    # Compute the histogram of the new image
    print "Computing histogram of enhanced '{}'...".format(image_path)
    new_hist = histogram(new_image)
    new_image.show()
    show_histogram(new_hist, "Histogram of enhanced '{}'".format(image_path))

    # Ensure that the matplotlib windows stay open
    pyplot.show()
