import argparse, os, sys
from PIL import Image
import numpy as np

from boundary import boundary_following
from resampling import resample_boundary, show_resampled_boundary


# Show an image
def show(img_data):
    Image.fromarray(img_data).show()

# Compute the 8-chaincode of the given boundary
def chaincode_8(boundary):
    """
    boundary: Ordered list of points forming the boundary.
    """
    chaincode = []

    for i in xrange(1, boundary.shape[0]):
        code = -1
        dy, dx = boundary[i] - boundary[i-1]
        if dx > 0:
            if dy > 0:     code = 7
            elif dy == 0:  code = 0
            else:          code = 1
        elif dx == 0:
            if dy > 0:     code = 6
            else:          code = 2
        else:
            if dy > 0:     code = 5
            elif dy == 0:  code = 4
            else:          code = 3

        chaincode.append(code)

    return np.array(chaincode, np.int)

# Compute the first difference of a 8-chaincode
def first_difference_8(chaincode):
    """
    chaincode: 8-chaincode array
    """

    first_diff = []
    for i in xrange(1, chaincode.shape[0]):
        diff = chaincode[i] - chaincode[i-1]
        first_diff.append(diff if diff >= 0 else 8 + diff)

    return np.array(first_diff, np.int)


# Main
if __name__ == "__main__":

    # Available args
    parser = argparse.ArgumentParser(description='Resample grid and compute 8-chaincode')

    parser.add_argument('boundary_image', type=str, help='Black and white boundary image')

    parser.add_argument('-s', '--sampling', dest='sampling', nargs = '+',
        type=int, default=[10, 10],
        help='Sampling spacing in x and y direction'
    )

    # Parse args
    args = parser.parse_args()

    # Input boundary image
    image = np.array(Image.open(args.boundary_image).convert('L'), np.uint8)
    show(image)

    # Compute boundary
    boundaries = boundary_following(image)
    boundary = boundaries[0]

    # Resample boundary
    resampled = resample_boundary(boundary, args.sampling)
    show_resampled_boundary(resampled, image.shape)

    # Compute 8-chaincode
    chaincode = chaincode_8(resampled)
    chaincode_str = "".join(str(x) for x in chaincode)
    print "Chaincode (length = {}): {}".format(len(chaincode_str), chaincode_str)

    # Compute first difference of the 8-chaincode
    first_difference = first_difference_8(chaincode)
    first_difference_str = "".join(str(x) for x in first_difference)
    print "First diff (length = {}): {}".format(len(first_difference_str), first_difference_str)
