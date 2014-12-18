#!/usr/bin/python

import sys
sys.path.append('..')

import argparse
import numpy as np

from common import Img
from dct import dct_compression, zonal_mask, threshold_mask


# Main
if __name__ == "__main__":

    # Available args
    parser = argparse.ArgumentParser(description='Transform image compression')

    parser.add_argument('image_path', type=str, help='Image path')

    dct_masks = parser.add_mutually_exclusive_group()
    dct_masks.add_argument('--zonal', action='store_true', help='Use a zonal mask')
    dct_masks.add_argument('--threshold', action='store_true', help='Use a threshold mask')

    parser.add_argument('-z', dest='z', type=int, default=4, help='Zonal mask size')

    # Parse args
    args = parser.parse_args()

    # Input image
    img = Img.load(args.image_path, dtype=np.float)
    Img.show(img)

    mask = None
    if args.zonal:
        mask = zonal_mask(args.z)
    elif args.threshold:
        mask = threshold_mask()

    if mask is not None:
        compressed = dct_compression(img, mask)
        Img.show(compressed)
        Img.show(img - compressed) # Difference between original and compressed
