#!/usr/bin/python

import sys
sys.path.append('..')

import argparse, os
import numpy as np

from common import Img


# Smoothing mask A (Figure 3.34 a)
def smoothing_3x3_mask_a():
    r = 1./9
    return np.array([[r, r, r], [r, r, r], [r, r, r]], dtype=np.float)

# Smoothing mask B (Figure 3.34 b)
def smoothing_3x3_mask_b():
    r = 1./16
    return np.array([[r, 2*r, r], [2*r, 4*r, 2*r], [r, 2*r, r]], dtype=np.float)

# Laplacian mask A (Figure 3.42 a)
def laplacian_3x3_mask_a(A=0):
    return np.array([[0, -1, 0], [-1, A+4, -1], [0, -1, 0]], dtype=np.float)

# Laplacian mask B (Figure 3.42 b)
def laplacian_3x3_mask_b(A=0):
    return np.array([[-1, -1, -1], [-1, A+8, -1], [-1, -1, -1]], dtype=np.float)

# Sobel 3x3 mask X-axis
def sobel_3x3_mask_xaxis():
    return np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float)

# Sobel 3x3 mask Y-axis
def sobel_3x3_mask_yaxis():
    return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float)

# 5x5 Smoothing mask
def smoothing_5x5_mask():
    r = 1./25
    return np.array([[r, r, r, r, r]] * 5, dtype=np.float)


# Add the filtered data to the original image and scale it to the range [0, 255]
# Return an linear array of pixels of the sharpened image
def sharpen_image(image, filtered_data):
    return Img.scale(image + filtered_data)

# Multiply two images
def multiply_images(img1, img2):
    return Img.scale(img1 * img2)

# Apply Power-Law transformation
def apply_power_law(image, gamma, c = 1.):
    return Img.scale(c * image**gamma)


# Apply a mask on an image and return the resulting image
def apply_mask(image, mask):

    w, h = image.shape

    # Get the pixel value at the given bi-dimensional index (x, y)
    # Return 0 if the if the index falls outside the image
    def get_pixel(x, y):
      p = 0 if x<0 or x>(w-1) or y<0 or y>(h-1) else image[x, y]
      return float(p)

    mask_w, mask_h = mask.shape

    output = np.empty(image.shape)
    for x in xrange(w):
        for y in xrange(h):
            # Apply the mask
            v = 0
            for i in xrange(mask_w):
                img_x = (x - mask_w / 2 + i)
                for j in xrange(mask_h):
                    img_y = (y - mask_h / 2 + j)
                    v += get_pixel(img_x, img_y) * float(mask[i][j])

            output[x, y] = v

    return output


# Apply a mask to the image, then scale the filtered data and sharpen it
def process_image(image, mask):

    filtered_data = apply_mask(image, mask)
    filtered_data = Img.scale(filtered_data)
    Img.show(filtered_data)

    sharpened = sharpen_image(image, filtered_data)
    Img.show(sharpened)

    return sharpened


# Main
if __name__ == "__main__":

    # Available args
    parser = argparse.ArgumentParser(
        description= 'Combination of spatial enhancement methods'
    )

    parser.add_argument('image_path', type=str, help='Image path')

    mask_types = parser.add_argument_group('Mask types', 'Type of the mask to apply')
    mask_types.add_argument('--laplacian', action='store_true',
        help='Use a 3x3 Laplacian filter (the value of A can be specified with \'-a\')'
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

    # Mask parameters
    A = args.a

    # Gamma parameter of the Power-Law transformation
    gamma = args.g

    # C parameter of the Power-Law transformation
    c = args.c

    # Load image
    image_path = args.image_path
    image = Img.load(image_path)
    w, h = image.shape

    # Apply a 3x3 Laplacian mask on the image
    if args.laplacian and not args.sobel:
        print "Applying laplacian B 3x3 mask (A = {}) on '{}'...".format(A, image_path)
        process_image(image, laplacian_3x3_mask_b(A))

    # Apply a 3x3 Sobel mask on the image
    if args.sobel and not args.laplacian:
        print "Applying Sobel X-axis 3x3 mask on '{}'...".format(image_path)
        sobel_x = apply_mask(image, sobel_3x3_mask_xaxis())
        Img.show(sobel_x)

        print "Applying Sobel Y-axis 3x3 mask on '{}'...".format(image_path)
        sobel_y = apply_mask(image, sobel_3x3_mask_yaxis())
        Img.show(sobel_y)

        print "Adding the two Sobel passes..."
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        Img.show(sobel)

        print "Smoothing Sobel..."
        data = apply_mask(sobel, smoothing_5x5_mask())
        Img.show(data)

    if args.sobel and args.laplacian:
        print "Applying laplacian B 3x3 mask (A = {}) on '{}'...".format(A, image_path)
        sharpened_img = process_image(image, laplacian_3x3_mask_b(A))

        print "Applying Sobel X-axis 3x3 mask on '{}'...".format(image_path)
        sobel_x = apply_mask(image, sobel_3x3_mask_xaxis())
        Img.show(sobel_x)

        print "Applying Sobel Y-axis 3x3 mask on '{}'...".format(image_path)
        sobel_y = apply_mask(image, sobel_3x3_mask_yaxis())
        Img.show(sobel_y)

        print "Adding the two Sobel passes..."
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        Img.show(sobel)

        print "Smoothing Sobel..."
        data = apply_mask(sobel, smoothing_5x5_mask())
        Img.show(data)

        print "Multiply Laplacian with smooth Sobel..."
        multiplied_img = multiply_images(sharpened_img, sobel)
        #Img.show(multiplied_img)

        print "Adding to original image..."
        enhanced_img = image + multiplied_img
        #Img.show(enhanced_img)

        print "Applying Power-Law transformation..."
        final_img = apply_power_law(enhanced_img, gamma, c)
        Img.show(final_img)
