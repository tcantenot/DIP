import scipy
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

im = ndimage.imread('noisy_fingerprint.tif')
o = ndimage.binary_erosion(im, np.ones((3,3)))
scipy.misc.imsave('erosion.png', o)
