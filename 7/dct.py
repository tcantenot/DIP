import numpy as np
from scipy import fftpack

# Textbook p598: Zonal mask
def zonal_mask(n, s=8):
    if n <= s:
        mask = np.zeros((s, s))
        for i in xrange(len(mask)):
            for j in reversed(xrange(n-i)):
                mask[i, j] = 1
    elif n <= s * 2:
        mask = np.zeros((n, n))
        for i in xrange(len(mask)):
            for j in reversed(xrange(n-i)):
                mask[i, j] = 1
    else:
        mask = np.ones((s, s))

    return mask[:s, :s]


# Textbook p598: threshold mask
def threshold_mask():
    return np.array([
    1, 1, 0, 1, 1, 0, 0, 0,
    1, 1, 1, 1, 0, 0, 0, 0,
    1, 1, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0 ]).reshape(8, 8)


# Apply a mask to an image
def apply_mask(subimg, mask):
    return subimg * mask


# Two-dimensional discrete cosine transform
def dct2(data):
    return fftpack.dct(fftpack.dct(data.T, norm='ortho').T, norm='ortho')

# Inverse two-dimensional discrete cosine transform
def idct2(data):
    return fftpack.idct(fftpack.idct(data.T, norm='ortho').T, norm='ortho')


# Generator yielding 8x8 subimages
def subimages(img, size):
    w, h = img.shape
    for x in xrange(0, w, size):
        for y in xrange(0, h, size):
            yield img[x:x+size, y:y+size]

# Compress the image
def compress_image(img, mask):
    """
    Divide the image into 8-by-8 subimages,
    compute the two-dimensional discrete cosine transform of each subimage,
    compress the test image to different qualities by discarding some
    DCT coefficients based on a mask and use the inverse discrete cosine
    transform with fewer transform coefficients.
    """

    MASK_SIZE = 8

    w, h = img.shape
    chunks = [[] for i in xrange(w/MASK_SIZE)]

    for i, subimg in enumerate(subimages(img, MASK_SIZE)):
        dct_data = dct2(subimg)
        dct_data = apply_mask(dct_data, mask)
        compress_img = idct2(dct_data)
        chunks[i/MASK_SIZE/MASK_SIZE].append(compress_img)

    return chunks

# Reconstruct the image from chunks
def reconstruct_image(img_chunks, w, h):
    """
    Reconstruct the image using an array of subimages
    """

    img = np.empty((w, h))
    sx, sy = img_chunks[0][1].shape

    for r, chunks in enumerate(img_chunks):
        for c, ch in enumerate(chunks):
            for i in xrange(sx):
                x = r * sx + i
                for j in xrange(sy):
                    y = c * sy + j
                    img[x, y] = ch[i, j]
    return img

# Perform DCT compression
def dct_compression(img, mask):
    """
    Perform image compression based on DCT using the given mask.
    img:  Input image.
    mask: Compression mask used to remove some DCT coefficients.
    """
    M, N = img.shape
    chunks = compress_image(img, mask)
    return reconstruct_image(chunks, M, N)
