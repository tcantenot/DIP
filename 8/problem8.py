#!/usr/bin/python

import argparse, os, sys
from PIL import Image
import numpy as np
from scipy import ndimage

B = 255

test = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, B, B, B, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, B, B, B, B, B, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, B, B, B, B, B, 0, 0, 0, 0, B, B, B, 0, 0],
    [0, 0, B, B, B, B, 0, 0, 0, 0, B, B, B, B, 0, 0],
    [0, 0, 0, B, B, 0, 0, 0, 0, B, B, B, B, B, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, B, B, B, B, B, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, B, B, B, B, B, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, B, B, B, B, B, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, B, B, B, B, B, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, B, B, B, B, B, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, B, B, B, B, B, B, B, B, 0, 0, 0, 0],
    [0, 0, 0, 0, B, B, B, B, B, B, B, B, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, B, B, B, B, B, B, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]).reshape((16, 16))


# Scale the data between 0 and 255
def scale_data(data):
    min_value = np.min(data)
    scaled_data = data - np.full(data.shape, min_value)
    max_value = np.max(scaled_data)
    scaled_data = scaled_data * np.full(data.shape, 255./max_value)
    return scaled_data

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

def erosion(data, mask):

    new_data = np.zeros(data.shape, dtype=int)

    w, h = data.shape
    mask_w2, mask_h2 = [s // 2 for s in mask.shape]

    zeros = np.zeros(mask.shape)

    for x in xrange(mask_w2, w-mask_w2):
        for y in xrange(mask_h2, h-mask_h2):
            neigh = data[x-mask_w2:x+mask_w2+1, y-mask_h2:y+mask_h2+1]
            new_data[x,y] = data[x,y] if np.array_equal(neigh - mask, zeros) else 0

    return new_data#[mask_w2:w-mask_w2, mask_h2:h-mask_h2]

    return np.array([data[i[0], i[1]] if np.array_equal(data[i[0]-(mask.shape[0]/2):i[0]+(mask.shape[0]/2)+1, i[1]-(mask.shape[1]/2):i[1]+(mask.shape[1]/2)+1] - mask, np.zeros(mask.shape)) else 0 for i, d in np.ndenumerate(data) if i[0] >= (mask.shape[0]/2) and i[0] < data.shape[0]-(mask.shape[0]/2) and i[1] >= (mask.shape[1]/2) and i[1] < data.shape[1]-(mask.shape[1]/2)]).reshape((data.shape[0]-mask.shape[0]+1, data.shape[1]-mask.shape[1]+1))

def dilation(data, mask, b=B):

    w, h = data.shape
    mask_w, mask_h = mask.shape
    mask_w2, mask_h2 = mask_w / 2, mask_h / 2

    new_data = np.zeros(data.shape, dtype=int)

    tmp = np.zeros((w+mask_w-1, h+mask_h-1), dtype=int)

    for x in xrange(w):
        for y in xrange(h):
            tmp[x+mask_w2, y+mask_h2] = data[x, y]

    zeros = np.zeros(mask.shape)

    for x in xrange(w):
        for y in xrange(h):
            neigh = tmp[x+mask_w2-mask_w2:x+mask_w2+mask_w2+1, y+mask_h2-mask_h2:y+mask_h2+mask_h2+1]
            if np.array_equal(neigh, zeros):
                new_data[x, y] = tmp[x + mask_w2, y + mask_h2]
            else:
                dilate = False
                for i in xrange(mask_w):
                    xx = x - mask_w2 + i
                    for j in xrange(mask_h):
                        yy = y - mask_h2 + j
                        if tmp[xx + mask_w2, yy + mask_h2] == mask[i, j]:
                            dilate = True
                            break
                if dilate:
                    new_data[x, y] = b

    return new_data


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
    #image.show()

    M, N = image.size

    # Image's pixels
    data = np.array(image.getdata()).reshape((M, N))

    mask = np.full((3, 3), B, dtype=int)

    data = test

    print data
    print ""

    # Erosion

    ref = 255 * ndimage.binary_erosion(data, structure=mask).astype('uint8')

    eroded = erosion(data, mask)

    print ref
    print ""
    print eroded
    print ""

    assert np.array_equal(ref, eroded), "Erosion is different from Scipy's"

    # Dilation

    ref = 255 * ndimage.binary_dilation(data, structure=mask).astype('uint8')

    dilated = dilation(data, mask)

    print ref
    print ""
    print dilated
    print ""

    assert np.array_equal(ref, dilated), "Dilation is different from Scipy's"
