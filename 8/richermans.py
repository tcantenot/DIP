from image import show_binary_img
from scipy import misc
import numpy as np
import argparse
from PIL import Image

def erosion(inputimg, mask):
    maskoffset = len(mask) / 2
    returnimg = np.zeros(inputimg.shape, dtype=bool)
    #padimg(returnimg, 1)
    for i in range(maskoffset, len(inputimg) - maskoffset):
        for j in range(maskoffset, len(inputimg[0]) - maskoffset):
            curmask = inputimg[i - maskoffset:i + maskoffset + 1, j - maskoffset:j + maskoffset + 1] & mask
            if (curmask == mask).all() :
                returnimg[i, j] = 1
#    Slice up the resulting image since the borders do not fit into the mask
    return returnimg

parser = argparse.ArgumentParser()
parser.add_argument('inputimage', type=str)
args = parser.parse_args()

# Open image
image = Image.open(args.inputimage)
if image == None:
    print "Failed to open image '{}'".format(args.inputimage)
    sys.exit(-2)

M, N = image.size

# Image's pixels
data = np.array(image.getdata()).reshape((M, N))

mask = np.ones((3, 3), dtype=bool)
show_binary_img(erosion(data, mask))
