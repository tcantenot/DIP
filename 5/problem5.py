#!/usr/bin/python

import argparse, os, sys
from math import exp, pow, sqrt
import math
import cmath
from PIL import Image
import numpy as np

from scipy import misc


def blurring_filter(a, b, T, shape):
    m, n = shape
    blurred = np.empty(shape, dtype=complex)

    for u in xrange(-m/2, m/2+1):
        for v in xrange(-n/2, n/2+1):
            x = u * a + v * b
            if x != 0:
                blurred[u, v] = T/(np.pi*(u*a+v*b)) * np.sin(np.pi*(u*a+v*b)) * np.exp(-1j*np.pi*(u*a+v*b))
            else:
                blurred[u, v] = T * np.exp(-1j*np.pi*(u*a+v*b))

    return blurred

def blurring_filter_(a, b, T, shape):
    m, n = shape
    u, v = np.ogrid[-m/2:m/2, -n/2:n/2]
    return T * np.sinc(u*a+v*b) * np.exp(-1j*np.pi*(u*a+v*b))

def inv_blurring_filter(a, b, T, shape):
    return 1./blurring_filter(a, b, T, shape)


def wiener_filter(H, K=0.1):
    H2 = np.abs(H) ** 2
    return (1./H) * (H2 / (H2 + K))


def wiener_filter_(H, N):

    #Sxx = np.abs(np.fft.fft2(H)) ** 2
    Sxx = H ** 2

    if 0:
        fft = np.fft.fft2(preprocess(H))
        show(fft)
        show(np.abs(fft) ** 2)

    #Snn = np.abs(np.fft.fft2(N)) ** 2
    Snn = N ** 2

    K = Snn / Sxx

    H2 = np.abs(H * H)
    return (1./H) * (H2 / (H2 + K))


def wiener_param(H, S, K=0.1):
    H2 = np.abs(H) ** 2
    return (1./H)**S * (H2 / (H2 + K)) ** (1 - S)


# Gaussian noise
def gaussian_noise(mu, sigma, shape):
    return np.random.normal(mu, sigma, shape)


# Preprocessing
def preprocess(data):
    x, y = np.ogrid[0:data.shape[0], 0:data.shape[1]]
    return data * ((-1)**(x+y))

# Postprocessing
def postprocess(data):
    return preprocess(data)


# Scale the data between 0 and 255
def scale_data(data):
    min_value = np.min(data)
    scaled_data = data - np.full(data.shape, min_value)
    max_value = np.max(scaled_data)
    scaled_data = scaled_data * np.full(data.shape, 255./max_value)
    return scaled_data


def extract_real(img):
    return np.array([x.real for _, x in np.ndenumerate(img)]).reshape(img.shape)


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

def show_fourier(fft_data):
    fft_img = fft_data.ravel()
    fft_img[0] = 0
    new_image(fft_img, M, N).show()


def process_image(img, filter, pp=True, scale=True, imshow_f=False):

    # Preprocess the data: multiply by (-1)^(x+y)
    if pp: img = preprocess(img)

    # 2D Fast Fourier Transform
    fft_data = np.fft.fft2(img)
    if imshow_f: show_fourier(fft_data)

    # Apply the filter
    filtered_data = fft_data * filter
    if imshow_f: show_fourier(filtered_data)

    # 2D Inverse Fast Fourier Transform
    filtered_img = np.fft.ifft2(filtered_data)

    # Retrieve the real part of the IFFT
    real_img = extract_real(filtered_img)

    # Postprocess the data: multiply by (-1)^(x+y)
    if pp: real_img = postprocess(real_img)

    # Scale the pixels values
    if scale: real_img = scale_data(real_img)

    return real_img

# Main
if __name__ == "__main__":

    # Available args
    parser = argparse.ArgumentParser(description=
        'Filtering in the frequency domain'
    )

    parser.add_argument('image_path', type=str, help='Image path')

    parser.add_argument('--blurred', dest='blurred', type=str, default=None,
        help='Precomputed blurred image'
    )

    parser.add_argument('--noise', action='store_true',
        help='Add Gaussian noise'
    )

    parser.add_argument('-s', dest='s', type=float, default=5,
        help='Sigma (variance) of the Gaussian noise'
    )

    parser.add_argument('--inv', action='store_true',
        help='Use inverse filter'
    )

    parser.add_argument('--wiener', action='store_true',
        help='Use Wiener deconvolution filter'
    )

    parser.add_argument('-a', dest='a', type=float, default=0.1,
        help='<a> parameter of the Wiener deconvolution filter'
    )

    parser.add_argument('-b', dest='b', type=float, default=0.1,
        help='<b> parameter of the Wiener deconvolution filter'
    )

    parser.add_argument('-T', dest='T', type=str, default=1.0,
        help='<T> parameter of the Wiener deconvolution filter'
    )

    parser.add_argument('-K', dest='K', type=float, default=0.1,
        help='Signal to noise ratio (Wiener filter)'
    )


    # Parse args
    args = parser.parse_args()

    a = args.a      # <a> parameter of the Wiener deconvolution filter
    b = args.b      # <b> parameter of the Wiener deconvolution filter
    T = args.T      # <T> parameter of the Wiener deconvolution filter
    K = args.K      # Signal-to-noise ratio
    sigma = args.s  # Sigma (variance) of the Gaussian noise

    shape = None
    bf = None
    blurred_image = None

    if not args.blurred:
        image = Image.open(args.image_path).convert('L')
        image.show()

        M, N = image.size

        # Image's pixels
        data = np.array(image.getdata()).reshape((M, N))

        shape = data.shape

        # Blurring filter
        bf = blurring_filter(a, b, T, data.shape)

        # Apply the blurring filter
        blurred_image = process_image(data, bf, pp=False, imshow_f=False)
        show(blurred_image)

    else:

        # Load precomputed blurred image
        blurred_image = Image.open(args.blurred).convert('L')
        blurred_image = np.array(blurred_image.getdata()).reshape(blurred_image.size)
        show(blurred_image)

        shape = blurred_image.shape


    # Blurring filter
    if bf is None:
        bf = blurring_filter(a, b, T, shape)

    # Add Gaussian noise
    blurred_image_w_noise = None
    if args.noise:
        blurred_image_w_noise = blurred_image + gaussian_noise(0, sigma, shape)
        blurred_image_w_noise = scale_data(blurred_image_w_noise)
        show(blurred_image_w_noise)


    # Inverse filter
    if args.inv:
        # Inverse blurring filter
        ibf = 1./bf

        # Apply the inverse blurring filter on the blurred image
        restored_image = process_image(blurred_image, ibf, pp=False, imshow_f=False)
        show(restored_image)

        # Apply the inverse blurring filter on the blurred and noisy image
        if args.noise:
            restored_image = process_image(blurred_image_w_noise, ibf, pp=False, imshow_f=False)
            show(restored_image)


    # Wiener deconvolution filter
    if args.wiener:

        # Wiener filter
        wf = wiener_filter(bf, K)
        #wf = wiener_filter_(bf, gaussian_noise(0, sigma, shape))
        #wf = wiener_param(bf, 0.1, K)

        # Apply the Wiener deconvolution filter on the blurred image
        restored_image = process_image(blurred_image, wf, pp=False, imshow_f=False)
        show(restored_image)

        # Apply the Wiener deconvolution filter on the blurred and noisy image
        if args.noise:
            restored_image = process_image(blurred_image_w_noise, wf, pp=False, imshow_f=False)
            show(restored_image)

    sys.exit(0)
