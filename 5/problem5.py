#!/usr/bin/python

import argparse, os, sys
from math import exp, pow, sqrt
import math
import cmath
from PIL import Image
import numpy as np


# Gaussian noise
def gaussian_noise(mu, sqsigma, shape):
    return np.random.normal(mu, np.sqrt(sqsigma), shape)


# Blurring filter
def blurring_filter(a, b, T, shape):
    m, n = shape
    u, v = np.ogrid[-m/2:m/2, -n/2:n/2]
    x = u * a + v * b
    return T * np.sinc(x) * np.exp(-1j*np.pi*x)


# Inverse filter
def inverse_filter(filter, d=0.01):
    """
        filter: Filter to invert
        d:      Threshold to constrain the division
    """
    return np.vectorize(lambda x: 1./x if np.abs(x) >= d else 1.0/d, otypes=[np.complex])(filter)


# Wiener filter
def wiener_filter(S, H, N=None):
    """
        S: Input image
        H: Degradation function
        N: Noise function
    """

    Sxx = np.abs(np.fft.fft2(preprocess(S))) ** 2
    #Sxx = S ** 2

    if 0:
        fft = np.fft.fft2(preprocess(H))
        show(fft)
        show(np.abs(fft) ** 2)

    #Snn = N ** 2 if N is not None else None
    Snn = np.abs(np.fft.fft2(N)) ** 2 if N is not None else None

    #return np.conj(H).T / (np.abs(H) ** 2 + Snn/Sxx) if Snn is not None else inverse_filter(H)

    if Snn is not None:
        m, n = H.shape

        d = 0.003

        result = np.empty(H.shape, dtype=complex)

        conjHT = np.conj(H).T
        denoms = (np.abs(H) ** 2 + Snn/Sxx)

        for u in xrange(m):
            for v in xrange(n):
                denom = denoms[u, v]
                result[u, v] = conjHT[u, v] / denom if np.abs(denom) >= d else 1.0

        return result

    else:
        return inverse_filter(H)



def wiener_param(H, S, NSR=0.1):
    H2 = np.abs(H) ** 2
    return (1./H)**S * (H2 / (H2 + NSR)) ** (1 - S)


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
    scaled_data = data - min_value
    max_value = np.max(scaled_data)
    scaled_data = scaled_data * (255./max_value)
    return scaled_data


# Show an image
def show(img_data):
    Image.fromarray(img_data).show()


def show_fourier(fft_data):
    fft_img = fft_data.ravel()
    fft_img[0] = 0
    fft_img.reshape(fft_data.shape)
    show(fft_img)


def apply_filter(img, filter, center=True, scale=True, imshow_f=False):

    # Preprocess the data: multiply by (-1)^(x+y)
    if center: img = preprocess(img)

    # 2D Fast Fourier Transform
    fft_data = np.fft.fft2(img)
    if imshow_f: show_fourier(fft_data)

    # Apply the filter
    filtered_data = fft_data * filter
    if imshow_f: show_fourier(filtered_data)

    # 2D Inverse Fast Fourier Transform
    filtered_img = np.fft.ifft2(filtered_data)

    # Retrieve the real part of the IFFT
    real_img = filtered_img.real

    # Postprocess the data: multiply by (-1)^(x+y)
    if center: real_img = postprocess(real_img)

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

    parser.add_argument('-s', dest='s', type=float, default=650,
        help='Variance (sigma square) of the Gaussian noise'
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


    image = Image.open(args.image_path).convert('L')
    #image.show()

    # Image's pixels
    data = np.array(image.getdata()).reshape(image.size)
    shape = data.shape

    # Blurring filter
    bf = blurring_filter(a, b, T, shape)

    blurred_image = None

    # Blur the image...
    if not args.blurred:
        blurred_image = apply_filter(data, bf)
        show(blurred_image)

   # ... or load a precomputed blurred image
    else:
        blurred_image = Image.open(args.blurred).convert('L')
        blurred_image = np.array(blurred_image.getdata()).reshape(blurred_image.size)
        show(blurred_image)


    # Add Gaussian noise
    blurred_image_w_noise = None
    noise = None
    if args.noise:
        noise = gaussian_noise(0, sigma, shape)
        blurred_image_w_noise = blurred_image + noise
        blurred_image_w_noise = scale_data(blurred_image_w_noise)
        show(blurred_image_w_noise)


    # Inverse filter
    if args.inv:

        # Inverse blurring filter
        ibf = inverse_filter(bf)

        # Apply the inverse blurring filter on the blurred image
        if not args.noise:
            restored_image = apply_filter(blurred_image, ibf)
            show(restored_image)

        # Apply the inverse blurring filter on the blurred and noisy image
        else:
            restored_image = apply_filter(blurred_image_w_noise, ibf)
            show(restored_image)


    # Wiener deconvolution filter
    if args.wiener:

        # Wiener filter
        wf = wiener_filter(data, bf, noise)

        # Apply the Wiener deconvolution filter on the blurred image
        if not args.noise:
            restored_image = apply_filter(blurred_image, wf)
            show(restored_image)

        # Apply the Wiener deconvolution filter on the blurred and noisy image
        else:
            restored_image = apply_filter(blurred_image_w_noise, wf)
            show(restored_image)

