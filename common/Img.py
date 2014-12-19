#!/usr/bin/python

from PIL import Image
import numpy as np

# Utility function related to images
class Img:

    # Load an image from disk
    @staticmethod
    def load(path, dtype=np.uint8):
        return np.array(Image.open(path).convert('L'), dtype=dtype)

    # Display the current image
    @staticmethod
    def show(img, dtype=None):
        if dtype is None:
            Image.fromarray(img).show()
        else:
            Image.fromarray(img.astype(dtype)).show()

    # Display a binary image
    @staticmethod
    def show_binary(img):
        Img.show(img * 255, dtype=np.uint8)

    # Show differences between two images
    @staticmethod
    def show_diff(lhs, rhs, dtype=None):
        Img.show(np.abs(lhs - rhs), dtype=dtype)

    # Save the given image to disk
    @staticmethod
    def save(path, img, dtype=None):
        if dtype is None:
            Image.fromarray(img).save(path)
        else:
            Image.fromarray(img.astype(dtype)).save(path)

    # Save the given image to disk
    @staticmethod
    def save_binary(path, img):
        Img.save(path, img * 255, dtype=np.uint8)

    # Scale the image intensity between 0 and 255
    @staticmethod
    def scale(img, zero=True, eps=np.finfo(np.float32).eps):
        min_value = np.min(img)
        scaled_data = img - min_value
        if zero: scaled_data[scaled_data <= eps] = 0.0
        max_value = np.max(scaled_data)
        if max_value != 0.0: scaled_data = scaled_data * (255./max_value)
        return scaled_data
