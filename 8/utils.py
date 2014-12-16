import numpy as np

# Return the intersection of the two binary arrays
def intersection(lhs, rhs):
    return np.logical_and(lhs, rhs)

# Return the union of the two binary arrays
def union(lhs, rhs):
    return np.logical_or(lhs, rhs)

# Return the complementary of the binary array
def complementary(input):
    return np.logical_not(input)
