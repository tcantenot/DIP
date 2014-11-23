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
])

boundary_test = np.array([
    [B, B, B, 0, B, B, B, B, B, 0],
    [B, B, B, 0, B, B, B, B, B, 0],
    [B, B, B, B, B, B, B, B, B, B],
    [B, B, B, B, B, B, B, B, B, B],
    [B, B, B, B, B, B, B, B, B, B]
])

filling_test = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, B, B, 0, 0, 0],
    [0, B, 0, 0, B, 0, 0],
    [0, B, 0, 0, B, 0, 0],
    [0, 0, B, 0, B, 0, 0],
    [0, 0, B, 0, B, 0, 0],
    [0, B, 0, 0, 0, B, 0],
    [0, B, 0, 0, 0, B, 0],
    [0, B, B, B, B, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
])

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


def intersection(lhs, rhs, bg=0):
    assert lhs.shape == rhs.shape, "Shapes must be equal"
    return np.array(
        [lhs[i] if lhs[i] == rhs[i] else bg for i, _ in np.ndenumerate(lhs)]
    ).reshape(lhs.shape)

def union(lhs, rhs, fg=B):
    assert lhs.shape == rhs.shape, "Shapes must be equal"
    return np.array(
        [fg if lhs[i] != rhs[i] else lhs[i] for i, _ in np.ndenumerate(lhs)]
    ).reshape(lhs.shape)



def erosion(data, mask, fg=B, bg=0):

    new_data = np.zeros(data.shape, dtype=int)

    w, h = data.shape
    mask_w2, mask_h2 = [s // 2 for s in mask.shape]

    zeros = np.zeros(mask.shape)

    for x in xrange(mask_w2, w-mask_w2):
        for y in xrange(mask_h2, h-mask_h2):
            neigh = data[x-mask_w2:x+mask_w2+1, y-mask_h2:y+mask_h2+1]
            new_data[x,y] = data[x,y] if np.array_equal(neigh - mask, zeros) else bg

    return new_data

    return np.array([data[i[0], i[1]] if np.array_equal(data[i[0]-(mask.shape[0]/2):i[0]+(mask.shape[0]/2)+1, i[1]-(mask.shape[1]/2):i[1]+(mask.shape[1]/2)+1] - mask, np.zeros(mask.shape)) else bg for i, d in np.ndenumerate(data) if i[0] >= (mask.shape[0]/2) and i[0] < data.shape[0]-(mask.shape[0]/2) and i[1] >= (mask.shape[1]/2) and i[1] < data.shape[1]-(mask.shape[1]/2)]).reshape((data.shape[0]-mask.shape[0]+1, data.shape[1]-mask.shape[1]+1))


def dilation(data, mask, fg=B, bg=0):

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
        tx = x + mask_w2
        for y in xrange(h):
            ty = y + mask_h2
            new_data[x, y] = tmp[tx, ty]

            neigh = tmp[tx-mask_w2:tx+mask_w2+1, ty-mask_h2:ty+mask_h2+1]

            if tmp[x + mask_w2, y + mask_h2] == fg or np.array_equal(neigh, zeros):
                new_data[x, y] = tmp[tx, ty]
            else:
                dilate = False
                for i in xrange(mask_w):
                    if dilate: break
                    xx = tx - mask_w2 + i
                    for j in xrange(mask_h):
                        yy = ty - mask_h2 + j
                        if tmp[xx, yy] == mask[i, j] and mask[i, j] == fg:
                            dilate = True
                            break
                if dilate:
                    new_data[x, y] = fg

    return new_data

def opening(data, mask):
    return dilation(erosion(data, mask), mask)

def closing(data, mask):
    return erosion(dilation(data, mask), mask)

def boundary_extraction(data, mask):
    return data - erosion(data, mask)


def holes_filling(data, mask, fg=B, bg=0):

    def _holes_filling(data, mask, fg, bg):

        boundary = boundary_extraction(data, mask)
        complementary = np.abs(data - np.full(data.shape, fg, dtype=int))

        X0 = np.zeros(data.shape, dtype=int)

        mask_h2 = mask.shape[1] / 2
        border = np.full(mask_h2, fg)

        # Find a point inside the boundary (find a pattern [bg, <mask_h2> fg, bg])
        init = False
        for x in xrange(X0.shape[0]):
            if init: break
            for y in xrange(X0.shape[1]-mask_h2-1):
                if boundary[x, y] == bg and \
                   np.array_equal(boundary[x, y+1:y+1+mask_h2], border) and \
                   boundary[x, y+1+mask_h2] == bg:
                       X0[x, y+2] = fg
                       init = True
                       break;

        X1 = None
        while True:
            X1 = intersection(dilation(X0, mask), complementary)
            if np.array_equal(X0, X1): break
            X0 = X1

        return union(X0, boundary)

    prev_res = None
    while True:
        res = _holes_filling(data, mask, fg, bg)
        show(res)
        if np.array_equal(res, prev_res): break
        prev_res = res

    return res

def connected_extraction(data, mask, fg=B, bg=0):

    X0 = None
    X1 = None
    while True:
        X1 = intersection(dilation(X0, mask), data)
        if np.array_equal(X0, X1): break
        X0 = X1

    return X1


def morpho(op, data, mask, name, ref_op=None, debug=False):

    print "{1}### {0} ###{1}".format(name, os.linesep)

    res = op(data, mask)

    if debug:
        if ref_op is not None:
            ref = 255 * ref_op(data, structure=mask).astype('uint8')
            print "Ref:{2}{0}{2}{2}Res:{2}{1}{2}".format(ref, res, os.linesep)
            assert np.array_equal(ref, res), "{} is different from Scipy's".format(name)
        else:
            print "{0}{1}".format(res, os.linesep)

    return res


# Main
if __name__ == "__main__":

    # Available args
    parser = argparse.ArgumentParser(description=
        'Filtering in the frequency domain'
    )

    parser.add_argument('image_path', type=str, help='Image path')

    parser.add_argument('--debug', action='store_true', help='Enable debugging msg')
    parser.add_argument('--test', action='store_true', help='Use testing data')

    op_types = parser.add_argument_group('Morphological operator types', 'Type of morphological operator')
    op_types.add_argument('--erosion', action='store_true', help='Use Erosion')
    op_types.add_argument('--dilation', action='store_true', help='Use Dilation')
    op_types.add_argument('--opening', action='store_true', help='Use Opening')
    op_types.add_argument('--closing', action='store_true', help='Use Closing')

    app_types = parser.add_argument_group(
        'Morphological operator application types', 'Type of morphological operator application')
    app_types.add_argument('--boundary', action='store_true', help='Use Boundary Extraction')
    app_types.add_argument('--filling', action='store_true', help='Use Holes Filling')

    # Parse args
    args = parser.parse_args()

    # Image path
    image_path = args.image_path

    debug = args.debug

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

    print data

    mask = np.full((3, 3), B, dtype=int)

    if args.test:
        data = test
    if args.debug:
        print "{}{}".format(data, os.linesep)

    # Erosion
    if args.erosion:
        eroded = morpho(erosion, data, mask, "Erosion", ndimage.binary_erosion, debug=debug)
        if not debug: show(eroded)

    # Dilation
    if args.dilation:
        dilated = morpho(dilation, data, mask, "Dilation", ndimage.binary_dilation, debug=debug)
        if not debug: show(dilated)

    # Opening
    if args.opening:
        opened = morpho(opening, data, mask, "Opening", ndimage.binary_opening, debug=debug)
        if not debug: show(opened)

    # Closing
    if args.closing:
        closed = morpho(closing, data, mask, "Closing", ndimage.binary_closing, debug=debug)
        if not debug: show(closed)

    # Boundary extraction
    if args.boundary:
        if args.test: data = boundary_test
        boundary = morpho(boundary_extraction, data, mask, "Boundary extraction", debug=debug)
        if not debug: show(boundary)

    # Holes filling
    if args.filling:
        if args.test: data = filling_test
        mask = np.array([
            [0, B, 0],
            [B, B, B],
            [0, B, 0]
        ])
        show(255 * ndimage.binary_fill_holes(data, structure=mask))
        #filled = morpho(holes_filling, data, mask, "Holes filling", ndimage.binary_fill_holes, debug=debug)
        #if not debug: show(filled)
