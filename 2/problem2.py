#!/usr/bin/python

import argparse, os, sys
from PIL import Image


# Smoothing mask A (Figure 3.34 a)
def smoothing_3x3_mask_a():
	r = float(1)/9
	return [[r, r, r], [r, r, r], [r, r, r]]

# Smoothing mask B (Figure 3.34 b)
def smoothing_3x3_mask_b():
	r = float(1)/16
	return [[r, 2*r, r], [2*r, 4*r, 2*r], [r, 2*r, r]]

# Laplacian mask A (Figure 3.42 a)
def laplacian_3x3_mask_a(A=0):
	return [[0, -1, 0], [-1, A+4, -1], [0, -1, 0]]

# Laplacian mask B (Figure 3.42 b)
def laplacian_3x3_mask_b(A=0):
	return [[-1, -1, -1], [-1, A+8, -1], [-1, -1, -1]]

# Sobel 3x3 mask X-axis
def sobel_3x3_mask_xaxis():
	return [[-1, 0, +1], [-2,  0, +2], [-1, 0, +1]]

# Sobel 3x3 mask Y-axis
def sobel_3x3_mask_yaxis():
	return [[+1, +2, +1], [0,  0, 0], [-1, -2, -1]]

# Transpose a 3x3 mask
def transpose_mask(mask):
    tr = mask
    tr[0][1] = mask[1][0]
    tr[0][2] = mask[2][0]
    tr[1][0] = mask[0][1]
    tr[1][2] = mask[2][1]
    tr[2][0] = mask[0][2]
    tr[2][1] = mask[1][2]

    return tr

# Create and return a new image of size (w, h) from the given linear array of pixels
def new_image(pixels, w, h):
    image = Image.new("L", [w, h])
    data = image.load()
    for x in xrange(w):
        for y in xrange(h):
            data[x, y] = pixels[y*w+x]
    return image


# Scale the pixels data resulting from the application of the mask to the range [0, 255]
def scale_filtered_data(pixels, w, h):

    min_value = min(pixels)
    for x in xrange(w):
        for y in xrange(h):
            pixels[y*w+x] -= min_value

    max_value = max(pixels)
    for x in xrange(w):
        for y in xrange(h):
            pixels[y*w+x] = pixels[y*w+x] * (float(255) / max_value)

    return pixels


# Add the filtered data to the original image and scale it to the range [0, 255]
def sharpen_image(image, filtered_data):

    w, h = image.size
    pixels = list(image.getdata())

    sharpened = [0] * w * h
    for i in xrange(len(sharpened)):
        sharpened[i] = pixels[i] + filtered_data[i]

    min_sharpened = min(sharpened)
    for i in xrange(len(sharpened)):
        sharpened[i] -= min_sharpened

    max_sharpened = max(sharpened)
    for i in xrange(len(sharpened)):
        sharpened[i] = sharpened[i] * (float(255) / max_sharpened)

    return sharpened


# Apply a 3x3 mask on an image and return the resulting image
def apply_mask(image, mask):
    w, h = image.size
    pixels = list(image.getdata())

	# Get the pixel value at the given bi-dimensional index (x, y)
	# Return 0 if the if the index falls outside the image
    def get_pixel(x, y):
      p = 0 if x<0 or x>(w-1) or y<0 or y>(h-1) else pixels[y*w+x]
      return float(p)

    data = [0] * w * h

    new_image = Image.new("L", image.size)
    new_pixels = new_image.load()

    mask_w = len(mask)
    mask_h = len(mask[0])

    print mask_w / 2
    print mask_h / 2

    if image.mode == "L": # Grey scale image
        for x in xrange(w):
            for y in xrange(h):
                # Apply the mask
                v = 0
                for i in xrange(mask_w):
                    for j in xrange(mask_h):
                        img_x = (x - mask_w / 2 + i)
                        img_y = (y - mask_h / 2 + j)
                        v += int(get_pixel(img_x, img_y) * float(mask[i][j]))

                data[y*w+x] = v

                # Clamp the pixel value to the range [0, 255]
                new_pixels[x, y] = max(0, min(255, v))

    return new_image, data


def contrast_stretching(image):
    w, h = image.size
    pixels = list(image.getdata())

    MP = 255
    a = min(pixels)
    b = max(pixels)
    R = b - a

    for x in xrange(w):
        for y in xrange(h):
            pixels[y*w+x] = round(MP * float(pixels[y*w+x]-a)/R)

    return new_image(pixels, w, h)


def process_image(image, mask):

    w, h = image.size

    filtered_image, filtered_data = apply_mask(image, mask)
    #filtered_image.show()

    scale_filtered_data(filtered_data, w, h)
    scaled_filtered_image = new_image(filtered_data, w, h)
    #scaled_filtered_image.show()

    sharpened = sharpen_image(image, filtered_data)
    sharpened_image = new_image(sharpened, w, h)
    sharpened_image.show()

    #contrast_stretching(scaled_filtered_image).show()


# Sum the pixels values of two images and return the resulting image
def add_images(img1, img2):
    if img1.size != img2.size: return None
    if img1.mode != "L" or img2.mode != "L": return None

    w, h = image.size
    pixels1 = list(img1.getdata())
    pixels2 = list(img2.getdata())

    new_image = Image.new("L", image.size)
    new_pixels = new_image.load()

    for x in xrange(w):
        for y in xrange(h):
            # Sum the two images' pixels
            new_pixels[x, y] = pixels1[y*w+x] + pixels2[y*w+x]
            # Clamp the pixel value to the range [0, 255]
            new_pixels[x, y] = max(0, min(255, new_pixels[x, y]))

    return new_image


# Main
if __name__ == "__main__":

    # Available args
    parser = argparse.ArgumentParser(description=
        'Combination of spatial enhancement methods'
    )

    parser.add_argument('path', type=str, help='Image path')
    parser.add_argument('-a', dest='a', type=float, default=0, help='"A" parameter of Laplacian mask')

    # Parse args
    args = parser.parse_args()

    # Image path
    image_path = args.path
    A = args.a

    # Check that the image exists
    if not os.path.isfile(image_path):
        print "Could not find image '{}'".format(image_path)
        sys.exit(-1)

    # Open image
    image = Image.open(image_path)
    if image == None:
        print "Failed to open image '{}'".format(image_path)
        sys.exit(-2)

    image.show()

    # Apply a laplacian B 3x3 mask on the image
    print "Applying laplacian B 3x3 mask (A = {}) on '{}'...".format(A, image_path)
    #process_image(image, laplacian_3x3_mask_b(A))
    process_image(image, laplacian_3x3_mask_b(0.0))
    process_image(image, laplacian_3x3_mask_b(1.0))
    process_image(image, laplacian_3x3_mask_b(1.7))
    #process_image(image, laplacian_3x3_mask_b(2.0))

    # Apply a Sobel Y-axis 3x3 mask on the image
    #print "Applying Sobel Y-axis 3x3 mask on '{}'...".format(image_path)
    #sobel_img, data = apply_mask(image, sobel_3x3_mask_yaxis())
    #sobel_img.show()

    # Apply a Sobel X-axis 3x3 mask on the image
    #print "Applying Sobel X-axis 3x3 mask on '{}'...".format(image_path)
    #sobel_img, data = apply_mask(sobel_img, sobel_3x3_mask_xaxis())
    #sobel_img.show()
