#import Image
from PIL import Image
import numpy as np

# Create and return a new image of size (w, h) from the given linear array of pixels
def new_image(pixels, w, h):
    image = Image.new('1', [w, h])
    data = image.load()
    for x in xrange(w):
        for y in xrange(h):
            data[x, y] = pixels[y*w+x]
    return image

image = Image.open('noisy_fingerprint.tif') # open colour image
image = image.convert('1') # convert image to black and white

M, N = image.size
print M, N

data = np.array(image.getdata()).reshape((M, N))

print data

new_img = new_image(data.ravel(), M, N)

new_img.save('result.png')
