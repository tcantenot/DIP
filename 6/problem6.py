import sys
sys.path.append('..')

import argparse
import numpy as np

from common import Img


class Transform(object):

    def __init__(self, interpolator):
        self.interpolator = interpolator

    # Translation
    def translate(self, src, ox, oy, truncate=False):
        """
        Translate the source image by the given amount.
        src:    Source Image.
        ox, oy: Translation along X and Y axis.
        """

        ox, oy = -ox, -oy

        sw, sh = src.shape
        dw, dh = int(sw+ox), int(sh+oy)

        # Generate a grid of pixel coordinates
        w1, w2 = int(min(0, ox)), int(sw + ox)
        h1, h2 = int(min(0, oy)), int(sh + oy)
        dx, dy = np.mgrid[w1:w2, h1:h2]

        sx, sy = dx, dy

        # Interpolate the pixels coordinates
        dest = self.interpolator.interpolate(src, sx, sy, dx, dy)

        if truncate:
            ox, oy = [s/2. for s in dest.shape]
            dest = dest[ox - sw/2.: ox + sw/2., oy - sh/2.: oy + sh/2.]

        return dest


    # Rotation
    def rotate(self, src, theta, truncate=False):
        """
        Rotate the source image by theta degree around its center.

        src:   Source image.
        theta: Rotation angle (counter-clockwise).
        ox:    X coordinate of the rotation center.
        oy:    Y coordinate of the rotation center.
        """

        theta *= np.pi / 180.0
        sw, sh = src.shape
        ox, oy = sw / 2., sh / 2.

        # Rotate the corners of the source image
        cx, cy = self._rotate([0, sw, sw, 0], [0, 0, sh, sh], theta, ox, oy)

        # Calculate the dimensions of the rotated image
        dw, dh = (int(np.ceil(c.max() - c.min())) for c in (cx, cy))

        # Coordinates of pixels in destination image
        dx, dy = np.mgrid[0:dw, 0:dh]

        # Rotate from destination to source (inverse rotation) to get
        # the corresponding coordinates of the source image in dest image
        sx, sy = self._rotate(dx + cx.min(), dy + cy.min(), -theta, ox, oy)

        # Nearest interpolation
        dest = self.interpolator.interpolate(src, sx, sy, dx, dy)

        if truncate:
            ox, oy = [s/2. for s in dest.shape]
            dest = dest[ox - sw/2.: ox + sw/2., oy - sh/2.: oy + sh/2.]

        return dest

    # Scaling
    def scale(self, src, s, truncate=False):
        sw, sh = src.shape
        dw, dh = np.int(sw * s), np.int(sh * s)

        # Coordinates of pixels in destination image
        dx, dy = np.mgrid[0:dw, 0:dh]

        x_ratio = sw / np.float(dw)
        y_ratio = sh / np.float(dh)

        sx = dx * x_ratio
        sy = dy * y_ratio

        # Interpolate the pixels coordinates
        dest = self.interpolator.interpolate(src, sx, sy, dx, dy)

        if s > 1.:
            if truncate:
                ox, oy = [s/2. for s in dest.shape]
                dest = dest[ox - sw/2.: ox + sw/2., oy - sh/2.: oy + sh/2.]
        else:
            ox, oy = [(l - r) / 2. for (l, r) in zip(src.shape, dest.shape)]
            dest = np.lib.pad(dest, (ox, oy), 'constant', constant_values=0)

        return dest

    # Rotate the x and y coordinates arrays around a point
    def _rotate(self, x, y, theta, ox, oy):
        """
        x, y:   X and Y-coordinates arrays to rotate.
        theta:  Rotation angle.
        ox, oy: Rotation center.
        """
        c, s = np.cos(theta), np.sin(theta)
        xx, yy = np.asarray(x) - ox, np.asarray(y) - oy
        return xx * c - yy * s + ox, xx * s + yy * c + oy


# Nearest neighbor
class NearestNeighbor(object):

    def interpolate(self, src, x, y, dx, dy):
        """
        x, y:   Source image pixels coordinates after destination-to-source transform.
        dx, dy: Destination image pixels coordinates (np.mgrid)
        """

        sw, sh = src.shape

        # Nearest neighbor interpolation
        f = lambda n: np.floor(n).astype(np.int)
        sx, sy = f(x), f(y)

        # Mask for valid pixel coordinates
        mask = (sx >= 0) & (sx < sw) & (sy >= 0) & (sy < sh)

        dest = np.empty(dx.shape, dtype=src.dtype)
        # Copy valid coordinates from source image.
        dest[dx[mask], dy[mask]] = src[sx[mask], sy[mask]]

        return dest


# Bilinear
class Bilinear(object):

    def interpolate(self, src, x, y, _, __):
        """
        x, y:   Source image pixels coordinates after destination-to-source transform.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        M, N = src.shape

        x1 = np.floor(x).astype(int)
        x2 = x1 + 1
        y1 = np.floor(y).astype(int)
        y2 = y1 + 1
        x1 = np.clip(x1, 0, M - 1);
        x2 = np.clip(x2, 0, M - 1);
        y1 = np.clip(y1, 0, N - 1);
        y2 = np.clip(y2, 0, N - 1);

        S11 = src[x1, y1]
        S12 = src[x1, y2]
        S21 = src[x2, y1]
        S22 = src[x2, y2]

        c = ((x2 - x1) * (y2 - y1))
        c = np.vectorize(lambda x: 1. / x if x != 0 else 1.)(c)

        I11 = (x2 - x) * (y2 - y)
        I12 = (x2 - x) * (y - y1)
        I21 = (x - x1) * (y2 - y)
        I22 = (x - x1) * (y - y1)

        return c * (S11 * I11 + S12 * I12 + S21 * I21 + S22 * I22)


# Main
if __name__ == "__main__":

    # Available args
    parser = argparse.ArgumentParser(description='Geometric transforms')

    # Input image
    parser.add_argument('-i', '--input', required=True, type=str, help='Input image path')

    # Transform types
    transforms = parser.add_mutually_exclusive_group(required=True);
    transforms.add_argument('-t', '--translate', nargs='+', type=float,
        default=None, help='Translate the image by the given amount'
    )
    transforms.add_argument('-r', '--rotate', type=float,
        default=None, help='Rotate the image around its center by the given angle'
    )
    transforms.add_argument('-s', '--scale', type=float,
        default=None, help='Scale the image by the given amount'
    )

    # Interpolation methods
    interpolations = parser.add_mutually_exclusive_group(required=True);
    interpolations.add_argument('--nearest', action='store_true',
        help='Use nearest-neighbor interpolation'
    )
    interpolations.add_argument('--bilinear', action='store_true',
        help='Use bilinear interpolation'
    )

    # Prevent truncating the transform image (allow different size)
    parser.add_argument('--ntruncate', action='store_true',
        help='Do not truncate the result of the transformation'
    )

    # Enable debug message
    parser.add_argument('--debug', action='store_true',
        help='Enable debug messages'
    )

    # Parse args
    args = parser.parse_args()

    # Input image
    img = Img.load(args.input)

    folder = 'output'

    # Choose interpolator
    interpolator = None
    interpolator_str = None
    if args.nearest:
        interpolator = NearestNeighbor()
        interpolator_str = 'nn'
    elif args.bilinear:
        interpolator = Bilinear()
        interpolator_str = 'bl'

    transform = Transform(interpolator)

    def _translate(src, ox, oy):
        f = lambda n: np.floor(n).astype(np.int)
        M, N = src.shape
        output = np.zeros(src.shape)
        ox, oy = -ox, -oy
        for (x, y), p in np.ndenumerate(src):
            xx, yy = x+ox, y+oy
            if xx >= 0 and xx < M and yy >= 0 and yy < N:
                output[x, y] = src[f(xx), f(yy)]
        return output

    # Perform transformation
    if args.translate is not None:
        if not args.nearest:
            dy, dx = args.translate[0], args.translate[1]
            translated = transform.translate(img, dx, dy, not args.ntruncate)
            Img.show(translated)
            Img.save("{}/translate_{}_{}_{}.png".format(folder, dy, dx, interpolator_str),
                translated, dtype=np.uint8
            )
        else:
            dy, dx = args.translate[0], args.translate[1]
            translated = _translate(img, dx, dy)
            Img.show(translated)
            Img.save("{}/translate_{}_{}_{}.png".format(folder, dy, dx, interpolator_str),
                translated, dtype=np.uint8
            )

    elif args.rotate is not None:
        theta = args.rotate
        rotated = transform.rotate(img, theta, not args.ntruncate)
        Img.show(rotated)
        Img.save("{}/rotate_{}_{}.png".format(folder, theta, interpolator_str), rotated, dtype=np.uint8)

    elif args.scale is not None:
        s = args.scale
        scaled = transform.scale(img, s, not args.ntruncate)
        Img.show(scaled)
        Img.save("{}/scale_{}_{}.png".format(folder, s, interpolator_str), scaled, dtype=np.uint8)
