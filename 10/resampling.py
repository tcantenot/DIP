import sys
sys.path.append('..')

import argparse
import numpy as np

from common import Img

# Find the nearest point on a grid
def nearest_grid_point(p, grid_spacing, grid_min=[0, 0]):
    """
    p:            Input point to find the nearest grid neighbor of.
    grid_spacing: [sx, sy] grid spacing in x and y direction.
    grid_min:     [mx, my] mininum coordinate in x and y.
    """

    p_x, p_y = p
    grid_min_x, grid_min_y = grid_min
    grid_spacing_x, grid_spacing_y = grid_spacing

    bin_x = (p_x - grid_min_x) / grid_spacing_x
    bin_y = (p_y - grid_min_y) / grid_spacing_y

    min_x, max_x = np.floor(bin_x), np.ceil(bin_x)
    min_y, max_y = np.floor(bin_y), np.ceil(bin_y)

    a = np.array([min_x, min_y])
    b = np.array([min_x, max_y])
    c = np.array([max_x, min_y])
    d = np.array([max_x, max_y])

    nearest_grid_points = np.array([a, b, c, d])

    da = (p_x - a[0]) ** 2 + (p_y - a[1]) ** 2
    db = (p_x - b[0]) ** 2 + (p_y - b[1]) ** 2
    dc = (p_x - c[0]) ** 2 + (p_y - c[1]) ** 2
    dd = (p_x - d[0]) ** 2 + (p_y - d[1]) ** 2

    dist = np.array([da, db, dc, dd])

    x, y = nearest_grid_points[np.argmax(dist, axis=0)]

    x = x * grid_spacing_x + grid_min_x
    y = y * grid_spacing_y + grid_min_y

    return [x, y]

# Resample the image using the given grid spacing
def resample_image(image, grid_spacing):
    """
    image:        Input image.
    grid_spacing: [sx, sy] grid spacing in x and y direction.
    """

    grid_spacing = grid_spacing[::-1]
    resampled = np.zeros(image.shape)
    for (x, y), d in np.ndenumerate(image):
        if d != 255: continue
        xx, yy = nearest_grid_point([x, y], grid_spacing)
        resampled[xx, yy] = 255

    return resampled

# Resample the given boudary using the given grid spacing
def resample_boundary(boundary, grid_spacing):
    """
    boundary:     Ordered list of boundary point.
    grid_spacing: [sx, sy] grid spacing in x and y direction.
    """
    output = []
    grid_spacing = grid_spacing[::-1]
    resampled = (nearest_grid_point(b, grid_spacing) for b in boundary)
    unique = set()
    for p in resampled:
        p = tuple(p)
        if p not in unique:
            unique.add(p)
            x, y = p
            output.append([x, y])

    return np.array(output, np.int)

def show_resampled_boundary(boundary, shape):
    image = np.zeros(shape)
    for (x, y) in boundary: image[x, y] = 255
    Img.show(image)


# Main
if __name__ == "__main__":

    # Available args
    parser = argparse.ArgumentParser(description='Image representation and description - Resampling grid')

    parser.add_argument('boundary_image', type=str, help='Black and white boundary image')

    parser.add_argument('-s', '--sampling', dest='sampling', nargs = '+',
        type=int, default=[30, 30],
        help='Sampling spacing in x and y direction'
    )

    # Parse args
    args = parser.parse_args()

    # Input boundary image
    image = Img.load(args.boundary_image)
    Img.show(image)

    # Resample grid
    resampled = resample_image(image, args.sampling)
    Img.show(resampled)
