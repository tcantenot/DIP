import argparse, os, sys
from PIL import Image
import numpy as np
from scipy import signal, misc
from scipy.ndimage.filters import gaussian_filter

from boundary import boundary_following


# Show an image
def show(img_data):
    Image.fromarray(img_data).show()

# Gaussian smoothing
def smooth_gauss(img, sigma):
    return gaussian_filter(img, np.sqrt(sigma))

test = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0],
    [0, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
])

test1 = np.array([
  [0,0,0,0,0,0,0],
  [0,0,0,1,0,0,0],
  [0,0,1,0,1,0,0],
  [0,0,1,0,0,0,0],
  [0,1,0,1,0,0,0],
  [0,1,1,1,0,0,0],
  [0,0,0,0,0,0,0],
  ])

# Main
if __name__ == "__main__":

    # Available args
    parser = argparse.ArgumentParser(description='Image representation and description')

    parser.add_argument('image_path', type=str, help='Image path')

    # Parse args
    args = parser.parse_args()

    image = Image.open(args.image_path).convert('L')
    data = np.array(image, np.uint8)

    data = smooth_gauss(data, 10)

    data[data < 128] = 0
    data[data >= 128] = 255


    show(data)

    #data = test * 255
    #data = test1 * 255

    boundaries = boundary_following(data)

    result = np.zeros(data.shape)

    for boundary in boundaries:
        for b in boundary:
            x, y = b.b
            result[x, y] = 255

    show(result)
