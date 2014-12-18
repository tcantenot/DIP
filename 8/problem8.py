#!/usr/bin/python

import argparse, os, sys
from PIL import Image
import numpy as np
from scipy import ndimage

from erosion import erosion
from dilation import dilation, dilation2D
from morpho import opening, closing, boundary_extraction, holes_filling, connected_extraction
from image import load, show, show_binary_img, save


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

    print "{1}### {0} ###{1}".format(name, os.linesep)

    res = op(img, structure)

    if debug:
        if ref_op is not None:
            ref = 255 * ref_op(img, structure).astype('uint8')
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
        'Filtering in the frequency domain'
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

    img = load(args.image_path)
    M, N = img.shape

    structure = np.full((3, 3), 1, dtype=int)

    print structure

    if args.test:
        img = test
    if args.debug:
        print "{}{}".format(img, os.linesep)

    # Erosion
    if args.erosion:
        eroded = morpho(erosion, img, structure, "Erosion", ndimage.binary_erosion, debug=debug)
        ref = ndimage.binary_erosion(img, structure=structure)
        if not debug: show_binary_img(eroded)
        show_binary_img(np.abs(ref - eroded))

    # Dilation
    if args.dilation:
        dilated = morpho(dilation, img, structure, "Dilation", ndimage.binary_dilation, debug=debug)
        ref = ndimage.binary_dilation(img, structure=structure)
        if not debug: show_binary_img(dilated)
        show_binary_img(np.abs(ref - dilated))

    # Opening
    if args.opening:
        opened = morpho(opening, img, structure, "Opening", ndimage.binary_opening, debug=debug)
        ref = ndimage.binary_opening(img, structure=structure)
        if not debug: show_binary_img(opened)
        show_binary_img(np.abs(ref - opened))

    # Closing
    if args.closing:
        closed = morpho(closing, img, structure, "Closing", ndimage.binary_closing, debug=debug)
        ref = ndimage.binary_closing(img, structure=structure)
        if not debug: show_binary_img(closed)
        show_binary_img(np.abs(ref - closed))

    # Boundary extraction
    if args.boundary:
        if args.test: img = boundary_test
        boundary = morpho(boundary_extraction, img, structure, "Boundary extraction", debug=debug)
        if not debug: show_binary_img(boundary)

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
        show_binary_img(filled)
        show_binary_img(np.abs(ref - filled))

    # Connected components extraction
    if args.connected:
        threshold = 203
        img[img < threshold]  = 0
        img[img >= threshold] = 1
        show_binary_img(img)
        structure = np.full((3, 3), 1, dtype=int)
        connected = morpho(connected_extraction, img, structure, "Connected components extraction", debug=debug)
        show_binary_img(connected)
