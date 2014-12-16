import numpy as np

from deque import MinDeque, MaxDeque
from utils import union


# Perform a dilation with a 2D structuring element in two passes
# with two 1D-structuring elements
def dilation2D(input, struct_1, struct_2, border_value=0, mask=None):
    d1 = dilation1D(input, struct_1, border_value, mask)
    d2 = dilation1D(input, struct_2, border_value, mask)
    return union(d1, d2)


# Perform a dilation with a 1D-structuring element
def dilation1D(input, structure, border_value=0, mask=None):

    assert structure.shape[0] == 1 or structure.shape[1] == 1, "Struture must be a 1D array"

    w, h = input.shape

    output = np.empty(input.shape, dtype=int)

    md = MaxDeque()

    if structure.shape[0] == 1:

        struct_h2 = (structure.shape[1] - 1) / 2

        for x in xrange(w):

            # Initialize MaxDeque
            for y in xrange(struct_h2):
                md.push(border_value)
            for y in xrange(struct_h2+1):
                md.push(input[x, y])

            # Dilate
            for y in xrange(h-struct_h2-1):
                output[x, y] = md.max() if mask is None or mask[x, y] else input[x, y]
                md.pop()
                md.push(input[x, y+struct_h2+1])
            for y in xrange(h-struct_h2-1, h):
                output[x, y] = md.max() if mask is None or mask[x, y] else input[x, y]
                md.pop()
                md.push(border_value)

            md.empty()

    elif structure.shape[1] == 1:

        struct_w2 = (structure.shape[0] - 1) / 2

        for y in xrange(h):

            # Initialize MaxDeque
            for x in xrange(struct_w2):
                md.push(border_value)
            for x in xrange(struct_w2+1):
                md.push(input[x, y])

            # Dilate
            for x in xrange(w-struct_w2-1):
                output[x, y] = md.max() if mask is None or mask[x, y] == 1 else input[x, y]
                md.pop()
                md.push(input[x+struct_w2+1, y])
            for x in xrange(w-struct_w2-1, w):
                output[x, y] = md.max() if mask is None or mask[x, y] == 1 else input[x, y]
                md.pop()
                md.push(border_value)

            md.empty()

    return output

# Perform a dilation of the input with the given structuring element
def dilation(input, structure, border_value=0, mask=None):

    w, h = input.shape
    struct_w, struct_h = structure.shape
    struct_w2, struct_h2 = struct_w / 2, struct_h / 2

    output = np.empty(input.shape, dtype=int)

    def get(x, y):
        return border_value if x < 0 or x >= w or y < 0 or y >= h else input[x, y]

    for x in xrange(w):
        for y in xrange(h):
            if mask is not None and not mask[x, y]: continue

            l = -float('inf')

            for i in xrange(struct_w):
                for j in xrange(struct_h):
                    if structure[i, j] == 1:
                        l = max(get(x+i-struct_w2, y+j-struct_h2), l)
                        if l == 1: break

                output[x, y] = l

    return output

if __name__ == '__main__':

    from scipy import ndimage

    B = 1

    input = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, B, B, B, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, B, B, B, B, B, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, B, B, B, B, B, 0, 0, 0, 0, B, B, B, 0, 0],
        [0, 0, B, B, B, B, 0, 0, 0, 0, B, B, B, B, 0, 0],
        [0, 0, 0, B, B, 0, 0, 0, 0, B, B, B, B, B, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, B, B, B, B, B, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, B, B, 0, B, B, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, B, B, 0, B, B, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, B, B, B, B, B, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, B, B, 0, B, B, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, B, B, 0, B, B, B, B, B, 0, 0, 0, 0],
        [0, 0, 0, 0, B, B, 0, B, B, B, B, B, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, B, B, B, B, B, B, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    struct_1 = np.array([
        [1], [1], [1]
    ])

    struct_2 = np.array([
        [1, 1, 1]
    ])

    d1 = dilation1D(input, struct_1, border_value=0) * 255
    ref = ndimage.binary_dilation(input, struct_1, border_value=0) * 255
    print "d1"
    print d1
    print ""
    print ref
    assert np.array_equal(d1, ref), "d1"
    print ""
    print ""

    d2 = dilation1D(input, struct_2, border_value=0) * 255
    ref = ndimage.binary_dilation(input, struct_2, border_value=0) * 255
    print "d2"
    print d2
    print ""
    print ref
    assert np.array_equal(d2, ref), "d2"
    print ""
    print ""

    d = dilation2D(input, struct_1, struct_2, border_value=0) * 255

    structure = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ])

    ref = ndimage.binary_dilation(input, structure, border_value=0) * 255

    print "d"
    print d
    print ""
    print ref

    assert np.array_equal(d, ref), "Dilation failed"
