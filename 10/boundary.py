import sys
sys.path.append('..')

import argparse
import numpy as np
from scipy import signal, misc
from scipy.ndimage.filters import gaussian_filter

from common import Img


class Point(object):

    def __init__(self, x, y):
        self.b = np.array([x, y])
        self.offsets = np.array([
            [+0, -1], [-1, -1], [-1, +0], [-1, +1],
            [+0, +1], [+1, +1], [+1, +0], [+1, -1]
        ])

    def __repr__(self):
        return str(self.b)

    def neighbors(self, c):
        offset = c - self.b
        index = 0
        for i in xrange(8):
            if np.array_equal(offset, self.offsets[i]):
                index = i
                break
        return iter(self.b + np.roll(self.offsets, -index, axis=0))

    def neighbor(self, idx):
        return self.b + self.offsets[idx]


def boundary_following_(img, b0, visited, v):

    boundary = []
    boundary.append(b0.b)

    valid = True

    # West neighbor of b0
    c0 = b0.neighbor(0)

    b1 = None
    c1 = None
    previous_c = np.array([b0.b[0], b0.b[1]])
    for (x, y) in b0.neighbors(c0):
        if visited[x, y] != 0 and visited[x, y] != v: valid = False
        visited[x, y] = v
        if img[x, y] == 255:
            b1 = Point(x, y)
            c1 = previous_c
            break
        previous_c = np.array([x, y])

    if b1 == None: return None

    b = b1
    c = c1

    b1_reached = False

    boundary.append(b.b)

    while True:

        for (x, y) in b.neighbors(c):
            if visited[x, y] != 0 and visited[x, y] != v: valid = False
            visited[x, y] = v
            if img[x, y] == 255:
                next_b = Point(x, y)
                c = previous_c
                break
            previous_c = np.array([x, y])

        for (x, y) in next_b.neighbors(c):
            if visited[x, y] != 0 and visited[x, y] != v: valid = False
            visited[x, y] = v
            if img[x, y] == 255:
                b_tmp = Point(x, y)
                b1_reached = np.array_equal(b_tmp.b, b1.b)
                c_tmp = previous_c
                break
            previous_c = np.array([x, y])

        b = next_b
        boundary.append(b.b)

        if b1_reached and np.array_equal(b.b, b0.b): break

    return np.array(boundary) if valid else None


def boundary_following(img):

    visited = np.zeros(img.shape)

    M, N = img.shape

    boundaries = []

    # Find starting point
    b0 = None
    iteration = 1
    for (x, y), p in np.ndenumerate(img):
        if x == 0 or x == (M-1) or y == 0 or y == (N-1): continue
        if p == 255 and visited[x, y] == 0:
            b0 = Point(x, y)

            try:
                boundary = boundary_following_(img, b0, visited, iteration)

                if boundary is not None:
                    boundaries.append(boundary)
                else:
                    pass
                    #print "ABORTED"
            except:
                pass

            iteration += 1

    return np.array(boundaries)


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
    parser = argparse.ArgumentParser(description='Image representation and description - Boundary following')

    parser.add_argument('--smooth', action='store_true', help='Smooth image to remove noise')

    parser.add_argument('image_path', type=str, help='Image path')

    # Parse args
    args = parser.parse_args()

    image = Img.load(args.image_path)

    # Smooth image
    if args.smooth:
        image = smooth_gauss(image, 10)

    # Binarize image
    image[image < 128] = 0
    image[image >= 128] = 255

    Img.show(image)

    boundaries = boundary_following(image)

    result = np.zeros(image.shape)
    for boundary in boundaries:
        for b in boundary:
            x, y = b
            result[x, y] = 255
    Img.show(result)
