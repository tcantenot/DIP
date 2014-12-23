#!/usr/bin/python

import sys
sys.path.append('..')

import argparse, os
import numpy as np
from scipy import ndimage

from common import Img

from erosion import erosion
from dilation import dilation, dilation2D
from morpho import opening, closing, boundary_extraction, holes_filling, connected_extraction


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


def morpho(op, img, structure, name, ref_op=None, debug=False):

    print "### {} ###".format(name)

    res = op(img, structure)

    if debug:
        if ref_op is not None:
            ref = ref_op(img, structure).astype('uint8')
            print "Ref:{2}{0}{2}{2}Res:{2}{1}{2}".format(ref, res, os.linesep)
            print img
            assert np.array_equal(ref, res), "{} is different from Scipy's".format(name)
        else:
            print "{0}{1}".format(res, os.linesep)

    return res


# Main
if __name__ == "__main__":

    # Available args
    parser = argparse.ArgumentParser(description=
        'Morphological processing'
    )

    parser.add_argument('image_path', type=str, help='Image path')

    parser.add_argument('--debug', action='store_true', help='Enable debugging msg')
    parser.add_argument('--test', action='store_true', help='Use testing img')

    op_types = parser.add_argument_group('Morphological operator types', 'Type of morphological operator')
    op_types.add_argument('--erosion', action='store_true', help='Use Erosion')
    op_types.add_argument('--dilation', action='store_true', help='Use Dilation')
    op_types.add_argument('--opening', action='store_true', help='Use Opening')
    op_types.add_argument('--closing', action='store_true', help='Use Closing')

    app_types = parser.add_argument_group(
        'Morphological operator application types', 'Type of morphological operator application')
    app_types.add_argument('--boundary', action='store_true', help='Use Boundary Extraction')
    app_types.add_argument('--filling', action='store_true', help='Use Holes Filling')
    app_types.add_argument('--connected', action='store_true', help='Use Connected Components Extraction')

    # Parse args
    args = parser.parse_args()

    debug = args.debug

    # Load and binarize the image
    img = Img.load(args.image_path)
    img[img < 255]  = 0
    img[img >= 255] = 1
    M, N = img.shape

    structure = np.full((3, 3), 1, dtype=int)

    if args.test:
        img = test
    if args.debug:
        print "{}{}".format(img, os.linesep)

    # Erosion
    if args.erosion:
        eroded = morpho(erosion, img, structure, "Erosion", ndimage.binary_erosion, debug=debug)
        ref = ndimage.binary_erosion(img, structure=structure)
        if not debug: Img.show_binary(eroded)

    # Dilation
    if args.dilation:
        dilated = morpho(dilation, img, structure, "Dilation", ndimage.binary_dilation, debug=debug)
        ref = ndimage.binary_dilation(img, structure=structure)
        if not debug: Img.show_binary(dilated)

    # Opening
    if args.opening:
        opened = morpho(opening, img, structure, "Opening", ndimage.binary_opening, debug=debug)
        ref = ndimage.binary_opening(img, structure=structure)
        if not debug: Img.show_binary(opened)

    # Closing
    if args.closing:
        closed = morpho(closing, img, structure, "Closing", ndimage.binary_closing, debug=debug)
        ref = ndimage.binary_closing(img, structure=structure)
        if not debug: Img.show_binary(closed)

    # Boundary extraction
    if args.boundary:
        if args.test: img = boundary_test
        boundary = morpho(boundary_extraction, img, structure, "Boundary extraction", debug=debug)
        if not debug: Img.show_binary(boundary)

    # Holes filling
    if args.filling:
        if args.test: img = filling_test
        structure = np.array([
            [0, B, 0],
            [B, B, B],
            [0, B, 0]
        ])
        ref = ndimage.binary_fill_holes(img, structure=structure)
        filled = morpho(holes_filling, img, structure, "Holes filling", debug=debug)
        Img.show_binary(filled)

    # Connected components extraction
    if args.connected:
        img = Img.load(args.image_path)

        # Thresholding
        threshold = 203
        img[img < threshold]  = 0
        img[img >= threshold] = 1
        Img.show_binary(img)

        # Erosion 5x5
        structure = np.full((5, 5), 1, dtype=int)
        img = erosion(img, structure)
        Img.show_binary(img)

        # Connected component extraction
        structure = np.full((3, 3), 1, dtype=int)
        connected, _ = morpho(connected_extraction, img, structure, "Connected components extraction", debug=debug)
        Img.show_binary(connected)
