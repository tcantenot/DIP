#!/usr/bin/python

import argparse, os, sys
from PIL import Image
import numpy as np
from scipy import ndimage

#TODO: binarize image and use only 0 and 1

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
    [0, 0, 0, 0, B, B, 0, B, B, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, B, B, 0, B, B, B, B, B, 0, 0, 0, 0],
    [0, 0, 0, 0, B, B, 0, B, B, B, B, B, 0, 0, 0, 0],
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


def intersection(lhs, rhs, fg=B, bg=0):
    assert lhs.shape == rhs.shape, "Shapes must be equal"
    return np.array(
        [lhs[i] if lhs[i] == rhs[i] else bg for i, _ in np.ndenumerate(lhs)]
    ).reshape(lhs.shape)

def union(lhs, rhs, fg=B):
    assert lhs.shape == rhs.shape, "Shapes must be equal"
    return np.array(
        [fg if lhs[i] != rhs[i] else lhs[i] for i, _ in np.ndenumerate(lhs)]
    ).reshape(lhs.shape)

def complementary(input, fg=B, bg=0):
    return np.abs(input - np.full(input.shape, fg, dtype=int))

def erosion(data, structure, fg=B, bg=0):

    new_data = np.full(data.shape, bg, dtype=int)

    w, h = data.shape
    struct_w, struct_h = structure.shape
    struct_w2, struct_h2 = struct_w / 2, struct_h / 2

    for x in xrange(struct_w2, w-struct_w2):
        for y in xrange(struct_h2, h-struct_h2):
            if data[x, y] == bg: continue
            equal = True
            for i in xrange(struct_w):
                if not equal: break
                xx = x - struct_w2 + i
                for j in xrange(struct_h):
                    yy = y - struct_h2 + j
                    if data[xx, yy] != structure[i, j] and structure[i, j] == fg:
                        equal = False
                        break

            new_data[x,y] = data[x,y] if equal else bg

    return new_data

    return np.array([data[i[0], i[1]] if np.array_equal(data[i[0]-(structure.shape[0]/2):i[0]+(structure.shape[0]/2)+1, i[1]-(structure.shape[1]/2):i[1]+(structure.shape[1]/2)+1] - structure, np.zeros(structure.shape)) else bg for i, d in np.ndenumerate(data) if i[0] >= (structure.shape[0]/2) and i[0] < data.shape[0]-(structure.shape[0]/2) and i[1] >= (structure.shape[1]/2) and i[1] < data.shape[1]-(structure.shape[1]/2)]).reshape((data.shape[0]-structure.shape[0]+1, data.shape[1]-structure.shape[1]+1))


def dilation(data, structure, fg=B, bg=0, border_value=0, mask=None):

    w, h = data.shape
    struct_w, struct_h = structure.shape
    struct_w2, struct_h2 = struct_w / 2, struct_h / 2

    new_data = np.copy(data)

    tmp = np.full((w+struct_w-1, h+struct_h-1), border_value, dtype=int)

    for x in xrange(w):
        for y in xrange(h):
            tmp[x+struct_w2, y+struct_h2] = data[x, y]

    for x in xrange(w):

        tx = x + struct_w2

        for y in xrange(h):

            if mask is not None and not mask[x, y]: continue

            if data[x, y] == fg: continue

            ty = y + struct_h2

            isZero = True
            for i in xrange(struct_w):
                if not isZero: break
                xx = tx - struct_w2 + i
                for j in xrange(struct_h):
                    yy = ty - struct_h2 + j

                    if structure[i, j] == fg and tmp[xx, yy] != bg:
                        isZero = False
                        break

            if isZero:
                new_data[x, y] = bg
            else:
                dilate = False
                for i in xrange(struct_w):
                    if dilate: break
                    xx = tx - struct_w2 + i
                    for j in xrange(struct_h):
                        yy = ty - struct_h2 + j
                        if tmp[xx, yy] == fg and structure[i, j] == fg:
                            dilate = True
                            break
                new_data[x, y] = fg if dilate else bg

    return new_data


def opening(data, structure):
    return dilation(erosion(data, structure), structure)

def closing(data, structure):
    return erosion(dilation(data, structure), structure)

def boundary_extraction(data, structure):
    return data - erosion(data, structure)


def holes_filling(data, structure, fg=B, bg=0):

    def scipy_filling(data, structure, fg, bg):
        #m = np.logical_not(data)
        #tmp = np.zeros(m.shape, dtype=int)
        #output = dilation(tmp, structure)=

        m = np.logical_not(data)
        tmp = np.zeros(m.shape, bool)
        output = ndimage.binary_dilation(tmp, structure, -1, m, None, 1)
        np.logical_not(output, output)

        return output * fg


    def _holes_filling(data, structure, fg, bg):

        boundary = boundary_extraction(data, structure)
        show(boundary)
        complementary = np.abs(boundary - np.full(data.shape, fg, dtype=int))
        show(complementary)

        X0 = np.zeros(data.shape, dtype=int)

        struct_h2 = structure.shape[1] / 2
        border = np.full(struct_h2, fg)

        # Find a point inside the boundary (find a pattern [bg, <struct_h2> fg, bg])
        init = False
        for x in xrange(X0.shape[0]):
            if init: break
            count = 0
            for y in xrange(X0.shape[1]-struct_h2-1):
                if boundary[x, y] == bg and \
                   np.array_equal(boundary[x, y+1:y+1+struct_h2], border) and \
                   boundary[x, y+1+struct_h2] == bg:
                       count += 1
                       if count == 2:
                           X0[x, y+2] = fg
                           init = True
                           break

        show(X0)

        X1 = None
        while True:
            dilated = dilation(X0, structure)
            X1 = intersection(dilated, complementary)
            if np.array_equal(X0, X1): break
            X0 = X1
            break

        return union(X0, boundary)


    def _holes_filling_2(data, structure):
        complementary = np.abs(boundary - np.full(data.shape, fg, dtype=int))
        show(complementary)


    #prev_res = None
    #while True:
        #res = _holes_filling(data, structure, fg, bg)
        #show(res)
        #if np.array_equal(res, prev_res): break
        #prev_res = res
        #break

    res = scipy_filling(data, structure, fg, bg)
    return res


def connected_extraction(data, structure, fg=B, bg=0):

    X0 = None
    X1 = None
    while True:
        X1 = intersection(dilation(X0, structure), data)
        if np.array_equal(X0, X1): break
        X0 = X1

    return X1


def morpho(op, data, structure, name, ref_op=None, debug=False):

    print "{1}### {0} ###{1}".format(name, os.linesep)

    res = op(data, structure)

    if debug:
        if ref_op is not None:
            ref = 255 * ref_op(data, structure).astype('uint8')
            print "Ref:{2}{0}{2}{2}Res:{2}{1}{2}".format(ref, res, os.linesep)
            print data
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

    structure = np.full((3, 3), B, dtype=int)

    structure = np.array([
        [0, B, 0],
        [B, B, B],
        [0, B, 0]
    ])

    blank = np.full((15, 15), 0, dtype=int)

    data = test

    mask = np.logical_not(data/255)
    print mask * 255
    tmp = np.zeros(data.shape, bool)

    #dilated = morpho(dilation, data, structure, "Dilation", ndimage.binary_dilation, debug=debug)
    prev_res = tmp * 1
    iter = 0
    while True:
        res = dilation(prev_res, structure, mask=mask, border_value=B)
        if np.array_equal(res, prev_res): break
        prev_res = res

    print complementary(prev_res)
    print np.logical_not(ndimage.binary_dilation(tmp, structure, -1, mask, None, 1)) * 255

    sys.exit(0)

    #structure = np.array([
        #[B, B, B],
        #[B, B, B],
        #[B, B, B]
    #])

    if args.test:
        data = test
    if args.debug:
        print "{}{}".format(data, os.linesep)

    # Erosion
    if args.erosion:
        #show(255 * ndimage.binary_erosion(data, structure=structure))
        eroded = morpho(erosion, data, structure, "Erosion", ndimage.binary_erosion, debug=debug)
        if not debug: show(eroded)

    # Dilation
    if args.dilation:
        #show(255 * ndimage.binary_dilation(data, structure=structure))
        dilated = morpho(dilation, data, structure, "Dilation", ndimage.binary_dilation, debug=debug)
        if not debug: show(dilated)

    # Opening
    if args.opening:
        opened = morpho(opening, data, structure, "Opening", ndimage.binary_opening, debug=debug)
        if not debug: show(opened)

    # Closing
    if args.closing:
        closed = morpho(closing, data, structure, "Closing", ndimage.binary_closing, debug=debug)
        if not debug: show(closed)

    # Boundary extraction
    if args.boundary:
        if args.test: data = boundary_test
        boundary = morpho(boundary_extraction, data, structure, "Boundary extraction", debug=debug)
        if not debug: show(boundary)

    # Holes filling
    if args.filling:
        if args.test: data = filling_test
        structure = np.array([
            [0, B, 0],
            [B, B, B],
            [0, B, 0]
        ])
        #show(255 * ndimage.binary_fill_holes(data, structure=structure))
        filled = morpho(holes_filling, data, structure, "Holes filling", ndimage.binary_fill_holes, debug=debug)
        show(filled)
        #if not debug: show(filled)
