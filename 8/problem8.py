#!/usr/bin/python

import argparse, os, sys
from PIL import Image
import numpy as np
from scipy import ndimage

from erosion import erosion, erosion3
from dilation import dilation, dilation2D
from morpho import opening, closing, holes_filling, connected_extraction
from image import new_image, show_binary_img


B = 1

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
    image = image.convert('1')
    image.show()

    M, N = image.size

    # Image's pixels
    data = np.array(image.getdata()).reshape((M, N))

    structure = np.full((3, 3), 1, dtype=int)

    print structure

    if args.test:
        data = test
    if args.debug:
        print "{}{}".format(data, os.linesep)

    # Erosion
    if args.erosion:
        show_binary_img(erosion3(data, structure) * 255)
        #show_binary_img(255 * ndimage.binary_erosion(data, structure=structure))
        #eroded = morpho(erosion2, data, structure, "Erosion", ndimage.binary_erosion, debug=debug)
        #if not debug: show_binary_img(eroded)

    # Dilation
    if args.dilation:
        #show_binary_img(255 * ndimage.binary_dilation(data, structure=structure))
        dilated = morpho(dilation, data, structure, "Dilation", ndimage.binary_dilation, debug=debug)
        if not debug: show_binary_img(dilated)

    # Opening
    if args.opening:
        opened = morpho(opening, data, structure, "Opening", ndimage.binary_opening, debug=debug)
        if not debug: show_binary_img(opened)

    # Closing
    if args.closing:
        closed = morpho(closing, data, structure, "Closing", ndimage.binary_closing, debug=debug)
        if not debug: show_binary_img(closed)

    # Boundary extraction
    if args.boundary:
        if args.test: data = boundary_test
        boundary = morpho(boundary_extraction, data, structure, "Boundary extraction", debug=debug)
        if not debug: show_binary_img(boundary)

    # Holes filling
    if args.filling:
        if args.test: data = filling_test
        structure = np.array([
            [0, B, 0],
            [B, B, B],
            [0, B, 0]
        ])
        #show_binary_img(255 * ndimage.binary_fill_holes(data, structure=structure))
        filled = morpho(holes_filling, data, structure, "Holes filling", ndimage.binary_fill_holes, debug=debug)
        show_binary_img(filled)
        #if not debug: show_binary_img(filled)
