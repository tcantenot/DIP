#!/usr/bin/python

import argparse, os, sys
from math import sqrt
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
    return [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

# Sobel 3x3 mask Y-axis
def sobel_3x3_mask_yaxis():
    return [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

# 5x5 Smoothing mask
def smoothing_5x5_mask():
    r = float(1) / 25
    return [[r, r, r, r, r]] * 5


# Create and return a image of size (w, h) creted from a linear array of pixels
def new_image(pixels, w, h):
    image = Image.new("L", [w, h])
    data = image.load()
    for x in xrange(w):
        for y in xrange(h):
            data[x, y] = pixels[y*w+x]
    return image

# Scale the filtered pixels data to the range [0, 255]
def scale_filtered_data(pixels, w, h):

    min_value = min(pixels)
    for x in xrange(w):
        for y in xrange(h):
            pixels[y*w+x] -= min_value

    print "Min value: {}".format(min_value)

    max_value = max(pixels)

    print "Max value: {}".format(max_value)
    for x in xrange(w):
        for y in xrange(h):
            pixels[y*w+x] = pixels[y*w+x] * (float(255) / max_value)

    return pixels


# Add the filtered data to the original image and scale it to the range [0, 255]
# Return an linear array of pixels of the sharpened image
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

# Multiply two images
def multiply_images(img1, img2):
    w, h = img1.size
    pixels1 = img1.getdata()
    pixels2 = img2.getdata()

    data = [0] * w * h

    for x in xrange(w):
        for y in xrange(h):
            idx = y * w + x
            data[idx] = pixels1[idx] * pixels2[idx]

    # Scale the data
    data = scale_filtered_data(data, w, h)

    return new_image(data, w, h)

# Apply Power-Law transformation
def apply_power_law(image, gamma, c = float(1)):
    w, h = image.size
    pixels = image.getdata()

    data = [0] * w * h

    for x in xrange(w):
        for y in xrange(h):
            idx = y * w + x
            data[idx] = c * pixels[idx]**gamma

    # Scale the data
    data = scale_filtered_data(data, w, h)

    return new_image(data, w, h)


# Apply a mask on an image and return the resulting image
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


# Apply a mask to the image, then scale the filtered data and sharpen it
def process_image(image, mask):
    w, h = image.size

    filtered_image, filtered_data = apply_mask(image, mask)
    #filtered_image.show()

    scale_filtered_data(filtered_data, w, h)
    scaled_filtered_image = new_image(filtered_data, w, h)
    scaled_filtered_image.show()

    sharpened = sharpen_image(image, filtered_data)
    sharpened_image = new_image(sharpened, w, h)
    sharpened_image.show()

    return sharpened_image


# Main
if __name__ == "__main__":

    # Available args
    parser = argparse.ArgumentParser(
        description= 'Combination of spatial enhancement methods'
    )

    parser.add_argument('image_path', type=str, help='Image path')

    mask_types = parser.add_argument_group('Mask types', 'Type of the mask to apply')
    mask_types.add_argument('--laplacian', action='store_true', help='Use a 3x3 Laplacian filter \
        (the value of A can be specified with \'-a\')'
    )
    mask_types.add_argument('--sobel', action='store_true', help='Use a 3x3 Sobel filter')

    parser.add_argument('-a', dest='a', type=float, default=0,
        help='"A" parameter of Laplacian mask'
    )

    parser.add_argument('-g', dest='g', type=float, default=0.5,
        help='"Gamma" parameter of Power-Law transformation'
    )

    parser.add_argument('-c', dest='c', type=float, default=1.0,
        help='"c" parameter of Power-Law transformation'
    )

    # Parse args
    args = parser.parse_args()

    # Image path
    image_path = args.image_path

    # Mask parameters
    A = args.a

    # Gamma parameter of the Power-Law transformation
    gamma = args.g

    # C parameter of the Power-Law transformation
    c = args.c

    # Check that the image exists
    if not os.path.isfile(image_path):
        print "Could not find image '{}'".format(image_path)
        sys.exit(-1)

    # Open image
    image = Image.open(image_path)
    if image == None:
        print "Failed to open image '{}'".format(image_path)
        sys.exit(-2)

    # Make sure the image is a gray scale image
    image = image.convert("L")
    image.show()
    w, h = image.size

    # Apply a 3x3 Laplacian mask on the image
    if args.laplacian and not args.sobel:
        print "Applying laplacian B 3x3 mask (A = {}) on '{}'...".format(A, image_path)
        process_image(image, laplacian_3x3_mask_b(A))

    # Apply a 3x3 Sobel mask on the image
    if args.sobel and not args.laplacian:
        print "Applying Sobel X-axis 3x3 mask on '{}'...".format(image_path)
        sobel_img, data_x = apply_mask(image, sobel_3x3_mask_xaxis())
        #sobel_img.show()

        print "Applying Sobel Y-axis 3x3 mask on '{}'...".format(image_path)
        sobel_img, data_y = apply_mask(image, sobel_3x3_mask_yaxis())
        #sobel_img.show()

        print "Adding the two Sobel passes..."
        sobel_data = [sqrt(x**2 + y**2) for (x, y) in zip(data_x, data_y)]
        sobel_img = new_image(sobel_data, image.size[0], image.size[1])
        sobel_img.show()

        print "Smoothing Sobel..."
        sobel_img, data = apply_mask(sobel_img, smoothing_5x5_mask())
        sobel_img.show()

    if args.sobel and args.laplacian:
        print "Applying laplacian B 3x3 mask (A = {}) on '{}'...".format(A, image_path)
        sharpened_img = process_image(image, laplacian_3x3_mask_b(A))

        print "Applying Sobel X-axis 3x3 mask on '{}'...".format(image_path)
        sobel_img, data_x = apply_mask(image, sobel_3x3_mask_xaxis())
        #sobel_img.show()

        print "Applying Sobel Y-axis 3x3 mask on '{}'...".format(image_path)
        sobel_img, data_y = apply_mask(image, sobel_3x3_mask_yaxis())
        #sobel_img.show()

        print "Adding the two Sobel passes..."
        sobel_data = [sqrt(x**2 + y**2) for (x, y) in zip(data_x, data_y)]
        sobel_img = new_image(sobel_data, image.size[0], image.size[1])
        sobel_img.show()

        print "Smoothing Sobel..."
        smooth_sobel_img, data = apply_mask(sobel_img, smoothing_5x5_mask())
        smooth_sobel_img.show()

        print "Multiply Laplacian with smooth Sobel..."
        multiplied_img = multiply_images(sharpened_img, sobel_img)
        multiplied_img.show()

        print "Adding to original image..."
        image_data = image.getdata()
        mul_img_data = multiplied_img.getdata()
        enhanced_img = new_image([(x + y) for (x, y) in zip(image_data, mul_img_data)], w, h)
        enhanced_img.show()

        print "Applying Power-Law transformation..."
        final_img = apply_power_law(enhanced_img, gamma, c)
        final_img.show()
