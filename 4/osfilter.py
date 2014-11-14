### Order-Statistic Filters ###

# Median filter
class MedianFilter(object):

    def __init__(self, width, height):
        self.width  = width  # Witdh of the filter
        self.height = height # Height of the filter

    def apply(self, x, y, w, h, pixels):
        """
            x: x coordinate of the current pixel
            y: y coordinate of the current pixel
            w: width of the image
            h: height of the image
            pixels: linear array of the image's pixels of length w * h
        """

        def get_pixel(x, y):
            return None if x<0 or x>(w-1) or y<0 or y>(h-1) else float(pixels[y*w+x])

        value = 0.

        values = []

        for i in xrange(self.width):
            for j in xrange(self.height):
                img_x = (x - self.width / 2 + i)
                img_y = (y - self.height / 2 + j)
                pixel = get_pixel(img_x, img_y)
                if pixel is not None:
                    values.append(pixel)

        values.sort()
        n = len(values)

        if n % 2 == 0:
            value = float(values[n/2-1] + values[n/2]) / 2.
        else:
            value = values[n//2]

        return value

# Max filter
class MaxFilter(object):

    def __init__(self, width, height):
        self.width  = width  # Witdh of the filter
        self.height = height # Height of the filter

    def apply(self, x, y, w, h, pixels):
        """
            x: x coordinate of the current pixel
            y: y coordinate of the current pixel
            w: width of the image
            h: height of the image
            pixels: linear array of the image's pixels of length w * h
        """

        def get_pixel(x, y):
            return None if x<0 or x>(w-1) or y<0 or y>(h-1) else float(pixels[y*w+x])

        value = -float('inf')

        for i in xrange(self.width):
            for j in xrange(self.height):
                img_x = (x - self.width / 2 + i)
                img_y = (y - self.height / 2 + j)
                pixel = get_pixel(img_x, img_y)
                if pixel is not None:
                    value = max(value, pixel)

        return value

# Min filter
class MinFilter(object):

    def __init__(self, width, height):
        self.width  = width  # Witdh of the filter
        self.height = height # Height of the filter

    def apply(self, x, y, w, h, pixels):
        """
            x: x coordinate of the current pixel
            y: y coordinate of the current pixel
            w: width of the image
            h: height of the image
            pixels: linear array of the image's pixels of length w * h
        """

        def get_pixel(x, y):
            return None if x<0 or x>(w-1) or y<0 or y>(h-1) else float(pixels[y*w+x])

        value = float('inf')

        for i in xrange(self.width):
            for j in xrange(self.height):
                img_x = (x - self.width / 2 + i)
                img_y = (y - self.height / 2 + j)
                pixel = get_pixel(img_x, img_y)
                if pixel is not None:
                    value = min(value, pixel)

        return value

# Midpoint filter
class MidpointFilter(object):

    def __init__(self, width, height):
        self.width  = width  # Witdh of the filter
        self.height = height # Height of the filter

    def apply(self, x, y, w, h, pixels):
        """
            x: x coordinate of the current pixel
            y: y coordinate of the current pixel
            w: width of the image
            h: height of the image
            pixels: linear array of the image's pixels of length w * h
        """

        def get_pixel(x, y):
            return None if x<0 or x>(w-1) or y<0 or y>(h-1) else float(pixels[y*w+x])

        m = float('inf')
        M = -float('inf')

        for i in xrange(self.width):
            for j in xrange(self.height):
                img_x = (x - self.width / 2 + i)
                img_y = (y - self.height / 2 + j)
                pixel = get_pixel(img_x, img_y)
                if pixel is not None:
                    m = min(m, pixel)
                    M = max(M, pixel)

        return 0.5 * (m + M)

# Alpha-trimmed filter
class AlphaTrimmedFilter(object):

    def __init__(self, width, height, d):
        self.width  = width  # Witdh of the filter
        self.height = height # Height of the filter
        self.d      = d      # d in [0, width * height - 1]

        assert 0 <= d < (width * height), \
            "d (= {}) must be in [0, width*height-1] = [0, {}]".format(d, width*height-1)

    def apply(self, x, y, w, h, pixels):
        """
            x: x coordinate of the current pixel
            y: y coordinate of the current pixel
            w: width of the image
            h: height of the image
            pixels: linear array of the image's pixels of length w * h
        """

        def get_pixel(x, y):
            return None if x<0 or x>(w-1) or y<0 or y>(h-1) else float(pixels[y*w+x])

        value = 0.

        values = []

        for i in xrange(self.width):
            for j in xrange(self.height):
                img_x = (x - self.width / 2 + i)
                img_y = (y - self.height / 2 + j)
                pixel = get_pixel(img_x, img_y)
                if pixel is not None:
                    values.append(pixel)

        values.sort()
        values = values[(self.d//2):-(self.d//2)]
        n = len(values)

        value = 1.0 / (float(self.width * self.height) - self.d) *  sum(values)

        return value
