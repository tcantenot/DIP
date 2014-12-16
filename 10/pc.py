import argparse, os, sys
from PIL import Image
import numpy as np

# Show an image
def show(img_data):
    Image.fromarray(img_data).show()

class PC(object):

    def __init__(self, images):
        self.images = images

        self.means     = self._compute_means()
        print self.means
        self.covs = self._compute_covariances()
        print self.covs
        #self.eigen_values, self.eigen_vectors = self._compute_eigen()
        #print self.eigen_values.shape
        #print self.eigen_values
        #print self.eigen_vectors.shape
        #print self.eigen_values

        #self._compute_sorted_eigen_matrix()

    def _compute_means(self):
        return np.array([np.mean(m) for m in self.images])

    def _compute_covariances(self):
        #M, N = self.images[0].shape
        #K = M * N
        print K
        return np.array([np.cov(m) for m in self.images])

    def _compute_eigen(self):
        eigen_values = []
        eigen_vectors = []

        for m in self.images:
            w, v = np.linalg.eig(m)
            eigen_values.append(w)
            eigen_vectors.append(v)

        return np.array(eigen_values), np.array(eigen_vectors)

    def _compute_sorted_eigen_matrix(self):
        indices = np.argsort(self.eigen_values)
        print indices


# Main
if __name__ == "__main__":

    # Available args
    parser = argparse.ArgumentParser(description='Image description by Principal Components (PC)')

    #parser.add_argument('image_path', type=str, help='Input image')

    # Parse args
    args = parser.parse_args()

    # Input images
    #image = np.array(Image.open(args.image_path).convert('L'), np.uint8)
    #show(image)

    image_paths = [
        "WashingtonDC_Band1.tif",
        "WashingtonDC_Band2.tif",
        "WashingtonDC_Band3.tif",
        "WashingtonDC_Band4.tif",
        "WashingtonDC_Band5.tif",
        "WashingtonDC_Band6.tif"
    ]

    images = np.array([np.array(Image.open(p).convert('L'), np.uint8) for p in image_paths])

    images = np.array([
        [0, 1, 1, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    pc = PC(images)
