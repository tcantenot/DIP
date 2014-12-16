import argparse, os, sys
from PIL import Image
import numpy as np

# Show an image
def show(img_data):
    Image.fromarray(img_data).show()

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
    resampled = (nearest_grid_point(b, grid_spacing) for b in boundary)
    unique = set()
    for p in resampled:
        p = tuple(p)
        if p not in unique:
            unique.add(p)
            x, y = p
            output.append([x, y])

    return np.array(output, np.int)
    #return np.array(list(set(tuple(p) for p in

def show_resampled_boundary(boundary, shape):
    image = np.zeros(shape)
    for (x, y) in boundary: image[x, y] = 255
    show(image)


# Main
if __name__ == "__main__":

    # Available args
    parser = argparse.ArgumentParser(description='Resampling grid')

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

    # Resample grid
    resampled = resample_image(image, args.sampling)
    show(resampled)
