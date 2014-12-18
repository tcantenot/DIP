import argparse
from PIL import Image
import numpy as np

# Scale the data between 0 and 255
def scale_data(data, zero=True, eps=np.finfo(np.float32).eps):
    min_value = np.min(data)
    scaled_data = data - min_value
    if zero: scaled_data[scaled_data <= eps] = 0.0
    max_value = np.max(scaled_data)
    if max_value != 0.0: scaled_data = scaled_data * (255./max_value)
    return scaled_data

# Load an grey-scale image
def load(path):
    return np.array(Image.open(path).convert('L'), np.uint8)

# Show an image
def show(img, scale=True, astype=None):
    image = img
    if scale: image = scale_data(img)
    if astype is not None: image = image.astype(astype)
    Image.fromarray(image).show()

# Save an image to disk
def save(path, img):
    Image.fromarray(scale_data(img).astype(np.uint8)).save(path)


class Transform(object):

    def __init__(self, interpolator):
        self.interpolator = interpolator

    # Translation
    def translate(self, src, ox, oy):
        """
        Translate the source image by the given amount.
        src:    Source Image.
        ox, oy: Translation along X and Y axis.
        """

        sw, sh = src.shape
        dw, dh = int(sw+ox), int(oy+sh)

        # Generate a grid of pixel coordinates
        dx, dy = np.mgrid[0:dw, 0:dh]

        sx, sy = dx, dy

        # Interpolate the pixels coordinates
        return self.interpolator.interpolate(src, sx, sy, dx, dy)

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
    def scale(self, src, s):
        sw, sh = src.shape
        dw, dh = np.int(sw * s), np.int(sh * s)

        # Coordinates of pixels in destination image
        dx, dy = np.mgrid[0:dw, 0:dh]

        x_ratio = sw / np.float(dw)
        y_ratio = sh / np.float(dh)

        sx = dx * x_ratio
        sy = dy * y_ratio

        # Interpolate the pixels coordinates
        return self.interpolator.interpolate(src, sx, sy, dx, dy)

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

        dest = np.empty((np.max(dx)+1, np.max(dy)+1), dtype=src.dtype)
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

    # Truncate the rotated image
    parser.add_argument('--truncate', action='store_true',
        help='Truncate the result of the rotation'
    )

    # Enable debug message
    parser.add_argument('--debug', action='store_true',
        help='Enable debug messages'
    )

    # Parse args
    args = parser.parse_args()

    # Input image
    img = load(args.input)

    # Choose interpolator
    interpolator = None
    if args.nearest:
        interpolator = NearestNeighbor()
    elif args.bilinear:
        interpolator = Bilinear()

    transform = Transform(interpolator)

    # Perform transformation
    if args.translate is not None:
        dx, dy = args.translate[0], args.translate[1]
        translated = transform.translate(img, dx, dy)
        show(translated)

    elif args.rotate is not None:
        theta = args.rotate
        rotated = transform.rotate(img, theta, args.truncate)
        show(rotated)

    elif args.scale is not None:
        s = args.scale
        scaled = transform.scale(img, s)
        show(scaled)
