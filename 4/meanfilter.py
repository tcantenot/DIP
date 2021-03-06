### Mean filters ###

# Arithmetic mean filter
class ArithmeticMeanFilter(object):

    def __init__(self, width, height):
        self.width  = width  # Witdh of the filter
        self.height = height # Height of the filter

    def apply(self, x, y, pixels):
        """
            x: x coordinate of the current pixel
            y: y coordinate of the current pixel
            pixels: 2D array of the image's pixels of shape (w, h)
        """

        w, h = pixels.shape

        def get_pixel(x, y):
            return float(0 if x<0 or x>(w-1) or y<0 or y>(h-1) else pixels[x, y])

        value = 0.

        for i in xrange(self.width):
            for j in xrange(self.height):
                img_x = (x - self.width / 2 + i)
                img_y = (y - self.height / 2 + j)
                value += get_pixel(img_x, img_y)

        value /= float(self.width * self.height)

        return value


# Geometric mean filter
class GeometricMeanFilter(object):

    def __init__(self, width, height):
        self.width  = width  # Witdh of the filter
        self.height = height # Height of the filter

    def apply(self, x, y, pixels):
        """
            x: x coordinate of the current pixel
            y: y coordinate of the current pixel
            pixels: 2D array of the image's pixels of shape (w, h)
        """

        w, h = pixels.shape

        def get_pixel(x, y):
            return float(1 if x<0 or x>(w-1) or y<0 or y>(h-1) else pixels[x, y])

        value = 1.

        for i in xrange(self.width):
            for j in xrange(self.height):
                img_x = (x - self.width / 2 + i)
                img_y = (y - self.height / 2 + j)
                value *= get_pixel(img_x, img_y)

        value = pow(value, 1./float(self.width * self.height))

        return value

# Harmonic mean filter
class HarmonicMeanFilter(object):

    """
        Salt Reduction (Pepper will increase) / Good for gaussian like
    """

    def __init__(self, width, height):
        self.width  = width  # Witdh of the filter
        self.height = height # Height of the filter

    def apply(self, x, y, pixels):
        """
            x: x coordinate of the current pixel
            y: y coordinate of the current pixel
            pixels: 2D array of the image's pixels of shape (w, h)
        """

        w, h = pixels.shape

        def get_pixel(x, y):
            return float(0 if x<0 or x>(w-1) or y<0 or y>(h-1) else pixels[x, y])

        value = 0.

        for i in xrange(self.width):
            for j in xrange(self.height):
                img_x = (x - self.width / 2 + i)
                img_y = (y - self.height / 2 + j)
                pixel = get_pixel(img_x, img_y)
                if pixel > 0: value += 1./ pixel

        value = float(self.width * self.height) / value

        return value

# Contra-Harmonic mean filter
class ContraHarmonicMeanFilter(object):

    def __init__(self, width, height, Q):
        self.width  = width  # Witdh of the filter
        self.height = height # Height of the filter
        self.Q = Q           # Q > 0: Pepper; Q < 0: Salt

    def apply(self, x, y, pixels):
        """
            x: x coordinate of the current pixel
            y: y coordinate of the current pixel
            pixels: 2D array of the image's pixels of shape (w, h)
        """

        w, h = pixels.shape

        def get_pixel(x, y):
            return float(0 if x<0 or x>(w-1) or y<0 or y>(h-1) else pixels[x, y])

        a = 0.
        b = 0.

        for i in xrange(self.width):
            for j in xrange(self.height):
                img_x = (x - self.width / 2 + i)
                img_y = (y - self.height / 2 + j)
                pixel = get_pixel(img_x, img_y)
                a += pixel**(self.Q+1) if pixel != 0 else 0
                b += pixel**(self.Q) if pixel != 0 else 0

        value = a / b

        return value
