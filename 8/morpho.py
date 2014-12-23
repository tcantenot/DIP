import numpy as np

from scipy import ndimage

from erosion import erosion
from dilation import dilation, dilation2D
from utils import complementary, intersection, union


# Opening operator
def opening(input, structure):
    return dilation(erosion(input, structure), structure)

# Closing operator
def closing(input, structure):
    return erosion(dilation(input, structure), structure)

# Boundary extraction
def boundary_extraction(input, structure):
    return input - erosion(input, structure)

# Holes filling
def holes_filling(input, _):

    struct_1 = np.array([
        [1], [1], [1]
    ])

    struct_2 = np.array([
        [1, 1, 1]
    ])

    structure = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ])

    mask = complementary(input)

    #iter = 0
    X0 = np.zeros(input.shape)
    while True:
        d1 = ndimage.binary_dilation(X0, struct_1, border_value=1, mask=mask)
        d2 = ndimage.binary_dilation(X0, struct_2, border_value=1, mask=mask)
        X1 = union(d1, d2)

        if np.array_equal(X0, X1): break
        X0 = X1
        #iter += 1

    return complementary(X0)


# Connected component extraction
def connected_extraction(input, structure):

    tmp = np.copy(input)

    output = np.zeros(input.shape, input.dtype)

    connections = []

    iter = 0
    while np.count_nonzero(tmp) > 0:

        # Find the beginning of the next connected component
        X0 = np.zeros(input.shape, input.dtype)
        for (x, y), p in np.ndenumerate(tmp):
            if p == 1:
                X0[x, y] = 1
                break

        # Extract the connected component
        X1 = None
        while True:
            #X1 = intersection(dilation(X0, structure), input)
            X1 = intersection(ndimage.binary_dilation(X0, structure), input)
            if np.array_equal(X0, X1): break
            X0 = X1

        # Remove connected component from the copy of the input image
        tmp[X0 == 1] = 0

        # Reconstruct input image
        output[X0 == 1] = 1

        # Count pixels of connected component
        connections.append(np.count_nonzero(X0))

        print "Connected component {}: {} pixel{}".format(
            iter, connections[-1], 's' if connections[-1] > 1 else ''
        )

        iter += 1

    return output, connections
