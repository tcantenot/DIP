import argparse, os, sys
from PIL import Image
import numpy as np
from scipy import misc

# Translation
def translate(img, translation):
    x, y = translation[0], translation[1]
    M, N = img.shape

    print M, N

    print "Translate : {}".format(translation)

    if x >= M or y >= N:
        return np.zeros(img.shape, dtype=int)

    if x == 0 and y == 0:
        return np.copy(img)

    translated = np.full(img.shape, 0, dtype=int)

    def get(x, y):
        return 0 if x < 0 or x >= M or y < 0 or y >= N else img[x, y]

    for (u,v), value in np.ndenumerate(img):
        translated[u, v] = get(u-y, v-x)

    return translated


# Create and return a new image of size (w, h) from the given linear array of pixels
def new_image(pixels, w, h):
    image = Image.new("L", [w, h])
    data = image.load()
    for x in xrange(w):
        for y in xrange(h):
            data[x, y] = pixels[y*w+x]
    return image


def show(img_data):
    w, h = img_data.shape
    new_image(img_data.ravel(), w, h).show()

def shownp(img_data):
    print img_data.shape
    image = Image.new("L", img_data.shape)
    data = image.load()

    for (u,v), value in np.ndenumerate(img_data):
        data[u, v] = int(value)

    image.show()




# Main
if __name__ == "__main__":

    # Available args
    parser = argparse.ArgumentParser(description= 'Geometric transforms')

    parser.add_argument('image_path', type=str, help='Image path')

    parser.add_argument('-t', '--translate', dest='t', nargs='+', type=float,
        default=None, help='Translation vector'
    )

    parser.add_argument('-s', '--scale', dest='s', nargs='+', type=float,
        default=None, help='Scale vector'
    )

    args = parser.parse_args()

    image = Image.open(args.image_path).convert('L')
    #image.show()

    translation = args.t
    scale = args.s

    M, N = image.size

    # Image's pixels
    data = np.array(image.getdata()).reshape((M, N))
    #misc.imshow(data)
    show(data)


    #data = np.arange(25).reshape((5, 5))
    #M, N = data.shape
    #print data
    #print data[-5:, :]

    shape = data.shape

    if translation is not None:
        x, y = translation[0], translation[1]
        translated = translate(data, translation)
        #assert np.equal(data[:-y, :-x], translated[y:, x:]).all()
        print shape
        print translated.shape
        print M, N
        ni = new_image(translated.ravel(), M, N)
        ni_data = np.array(ni.getdata()).reshape((M, N))
        print ni_data.shape
        print translated.shape
        assert np.equal(ni_data, translated).all()

        show(translated)
