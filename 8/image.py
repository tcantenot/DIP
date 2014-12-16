import numpy as np
from PIL import Image

# Create and return a new image of size (w, h) from the given linear array of pixels
def new_image(pixels, w, h):
    image = Image.new('1', [w, h])
    data = image.load()
    for x in xrange(w):
        for y in xrange(h):
            data[x, y] = pixels[y*w+x]
    return image


# Display a binray image
def show_binary_img(img_data):
    w, h = img_data.shape
    new_image(img_data.ravel(), w, h).show()
