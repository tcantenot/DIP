import numpy as np
from PIL import Image

# Load an grey-scale image
def load(path):
    return np.array(Image.open(path).convert('L'), np.uint8)

# Display an image
def show(img):
    Image.fromarray(img).show()

# Display a binary image
def show_binary_img(img):
    show(img.astype(np.uint8) * 255)

# Save an image to disk
def save(path, img):
    Image.fromarray(scale_data(img).astype(np.uint8)).save(path)
