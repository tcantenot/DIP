import argparse
from scipy import misc
import numpy as np
from matplotlib.pyplot import imshow, show


def _masks(key):
    mask = np.array([
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]
                    ], dtype=bool
                    )

    largemask = np.ones((5, 5), dtype=bool)
    masks = {'mask':mask, 'largemask':largemask}
    return masks[key]


def main():
    args = parseArgs()
    mask = _masks('mask')
    if args.f:
        eros = erosion(args.inputimage, mask)
        dile = diletation(args.inputimage, mask)
        openimg = opening(args.inputimage, mask)
        closeimg = closing(args.inputimage, mask)
        if args.o:
            misc.imsave(args.o + '_erosion.tif', eros)
            misc.imsave(args.o + '_diletation.tif', dile)
            misc.imsave(args.o + '_opening.tif', openimg)
            misc.imsave(args.o + '_closing.tif', closeimg)
    if args.s:
        boundary_extract = boundaryextraction(args.inputimage,mask)
        hole_filled_img = holefilling(args.inputimage, mask)
        if args.o:
            misc.imsave(args.o + '_boundaryextraction.tif',boundary_extract)
            misc.imsave(args.o + '_holesfilled.tif',hole_filled_img)

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputimage', type=misc.imread)
    parser.add_argument('-f', help='does the first part of the homework, namely Erosion, \
    diletation, opening and closing, writes out the pictures to -o, if given',action='store_true')
    parser.add_argument('-s',help='Does the second part of the homework, namely',action='store_true')
    parser.add_argument('-o', type=str, help='Outputfile')
    return parser.parse_args()

def boundaryextraction(inputimage, mask):
    eros = erosion(inputimage, mask)
    return inputimage - eros


def holefilling(inputimg,mask):
    x_0 = np.zeros(inputimg.shape,dtype=bool)
#     maskoffset = len(mask) / 2
    complement_inputimg = np.invert(inputimg,dtype=bool)
    while True:
        lastx = x_0
        diletated = diletation(x_0, mask)
        x_0 = diletated & complement_inputimg
        if (lastx == x_0).all():
            break
#     for i in range(maskoffset, len(inputimg) - maskoffset):
#         for j in range(maskoffset, len(inputimg[0]) - maskoffset):

#             structureelement = x_0[i - maskoffset:i + maskoffset + 1, j - maskoffset:j + maskoffset + 1] | mask
#             result = structureelement & complement_inputimg [i - maskoffset:i + maskoffset + 1, j - maskoffset:j + maskoffset + 1]
# #             print "before" ,x_0[i - maskoffset:i + maskoffset + 1, j - maskoffset:j + maskoffset + 1]
#             x_0[i - maskoffset:i + maskoffset + 1, j - maskoffset:j + maskoffset + 1] = result
#             print "Resultmat : ",result
#             print "After" ,x_0[i - maskoffset:i + maskoffset + 1, j - maskoffset:j + maskoffset + 1]
    return x_0 |inputimg


def diletation(inputimg, mask):
    maskoffset = len(mask) / 2
    m, n = inputimg.shape
    returnimg = np.ones((m, n), dtype=bool)
    padimg(returnimg, 0)
    for i in range(maskoffset, len(inputimg) - maskoffset):
        for j in range(maskoffset, len(inputimg[0]) - maskoffset):
            structureelement = inputimg[i - maskoffset:i + maskoffset + 1, j - maskoffset:j + maskoffset + 1] & mask
            if (structureelement == mask).all():
                returnimg[i, j] = 0
    return returnimg

def opening(inputimg, mask):
    return erosion(diletation(inputimg, mask), mask)

def closing(inputimg, mask):
    return diletation(erosion(inputimg, mask), mask)

# Pads the inputimg, sets any border to val, does only work for 3x3
def padimg(inputimg,val):
    m,n = inputimg.shape
    inputimg[0:m,0]= val
    inputimg[0,0:n] = val
    inputimg[0:m,n-1] = val
    inputimg[m-1,0:n] = val

def erosion(inputimg, mask):
    maskoffset = len(mask) / 2
    returnimg = np.zeros(inputimg.shape, dtype=bool)
    padimg(returnimg, 1)
    for i in range(maskoffset, len(inputimg) - maskoffset):
        for j in range(maskoffset, len(inputimg[0]) - maskoffset):
            curmask = inputimg[i - maskoffset:i + maskoffset + 1, j - maskoffset:j + maskoffset + 1] & mask
            if (curmask == mask).all() :
                returnimg[i, j] = 1
#    Slice up the resulting image since the borders do not fit into the mask
    return returnimg

if __name__ == '__main__':
    main()
