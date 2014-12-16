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

# Main
if __name__ == "__main__":

    # Available args
    parser = argparse.ArgumentParser(description='Resampling grid')

    parser.add_argument('image_path', type=str, help='Boundary image path')

    parser.add_argument('-s', '--sampling', dest='sampling', nargs = '+',
        type=int, default=[10, 10],
        help='Sampling spacing in x and y direction'
    )

    # Parse args
    args = parser.parse_args()

    # Input boundary image
    image = np.array(Image.open(args.image_path).convert('L'), np.uint8)
    show(image)

    # Sampling spacing
    sx, sy = args.sampling[0], args.sampling[1]

    # Resample grid
    resampled = np.zeros(image.shape)
    for (x, y), d in np.ndenumerate(image):
        if d != 255: continue
        xx, yy = nearest_grid_point([x, y], [sx, sy])
        resampled[xx, yy] = 255

    show(resampled)
