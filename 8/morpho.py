import numpy as np

from erosion import erosion
from dilation import dilation, dilation2D
from utils import intersection, complementary

# FIXME
from image import show_binary_img
from scipy import ndimage
from utils import union


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
def holes_filling(input, structure):
    mask = complementary(input)

    #iter = 0
    X0 = np.zeros(input.shape)
    while True:
        X1 = dilation(X0, structure, mask=mask, border_value=1)
        if np.array_equal(X0, X1): break
        X0 = X1
        #iter += 1
        #if iter % 15 == 0: show_binary_img(X0)

    return complementary(X0)

def holes_filling(input):
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

    iter = 0
    X0 = np.zeros(input.shape)
    while True:
        #X1 = dilation(X0, structure, mask=mask, border_value=1)
        #X1 = dilation2D(X0, struct_1, struct_2, border_value=1, mask=mask)
        #X1 = ndimage.binary_dilation(X0, structure, border_value=1, mask=mask)
        d1 = ndimage.binary_dilation(X0, struct_1, border_value=1, mask=mask)
        d2 = ndimage.binary_dilation(X0, struct_2, border_value=1, mask=mask)
        X1 = union(d1, d2)


        #show_binary_img(X1)
        #break
        if np.array_equal(X0, X1): break
        X0 = X1
        iter += 1
        if iter % 15 == 0: show_binary_img(X0)

    return complementary(X0)



# Connected component extraction
def connected_extraction(input, structure):

    X0 = None
    X1 = None
    while True:
        X1 = intersection(dilation(X0, structure), input)
        if np.array_equal(X0, X1): break
        X0 = X1

    return X1
