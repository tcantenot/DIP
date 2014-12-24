import sys
sys.path.append('..')

import argparse
import numpy as np

from common import Img

# Principal components
def principal_components(images, n=None, debug=False, showr=False, showdiff=False):
    """
    images:    Array of image components.
    n:         Number of eigen values/vector to keep (None means keep them all).
    debug:     Enable/Disable debug messages.
    showr:     Img.show reconstructed images.
    showdiff:  Img.show difference between original and reconstructed images.
    """

    np.set_printoptions(linewidth=150)

    def pprint(str, x):
        if debug: print "{} {}\n{}\n".format(str, x.shape, x)

    def remove_eigen_vectors(A, n):
        return A[:-n]

    M, N, O = images.shape
    x = images.reshape((M, N * O))
    pprint("x", x)

    # Mean vector
    means = np.mean(x, axis=1, keepdims=True)
    pprint("Mean vector", means)

    # Covariance matrix
    cov = np.cov(x)
    pprint("Covariance matrix", cov)

    # Eigen values and vectors
    eigen_values, eigen_vectors = np.linalg.eig(cov)
    pprint("Eigen values", eigen_values)
    pprint("Eigen vectors", eigen_vectors)

    # A matrix (ordered eigen vectors as rows)
    A = eigen_vectors
    pprint("A", A)

    # Remove some eigen values/vectors
    if n is not None:
        dn = M - n
        if dn > 0:
            if debug:
                print "Removing {0} eigen value{1}/vector{1}"\
                    .format(dn, "s" if dn > 1 else "")
            A = remove_eigen_vectors(A, dn)
            pprint("New A", A)
    else:
        n = M

    # Hotelling transform
    y = np.dot(A, (x - means))
    pprint("y = A(x - means)", y)

    # Reconstructing images
    xx = np.dot(A.T, y) + means
    pprint("Reconstructed images", xx)
    reconstructed = xx.reshape((M, N, O))

    # Compute the differences
    diffs = np.abs(reconstructed - images)

    # Display results
    for i, (image, diff) in enumerate(zip(reconstructed, diffs)):
        if showr:    Img.show(Img.scale(image))
        if showdiff: Img.show(Img.scale(diff))

    return reconstructed, diffs


# Main
if __name__ == "__main__":

    # Available args
    parser = argparse.ArgumentParser(description='Image description by Principal Components (PC)')

    parser.add_argument('-n', type=int, default=6,
        help='Number of eigen vectors to keep'
    )

    parser.add_argument('--debug', action='store_true',
        help='Enable debug messages'
    )

    parser.add_argument('--diff', action='store_true',
        help='Display the difference between original and reconstructed images'
    )

    parser.add_argument('--nshow', action='store_true',
        help='Don\'t display the reconstructed images'
    )


    # Parse args
    args = parser.parse_args()

    image_paths = [
        'WashingtonDC_Band1.tif',
        'WashingtonDC_Band2.tif',
        'WashingtonDC_Band3.tif',
        'WashingtonDC_Band4.tif',
        'WashingtonDC_Band5.tif',
        'WashingtonDC_Band6.tif'
    ]

    images = np.array([Img.load(p) for p in image_paths])

    n = np.clip(args.n, 0, 6)

    # Principal components
    pc_images, pc_diffs = principal_components(images, n,
        debug=args.debug, showr=not args.nshow, showdiff=args.diff
    )

    # Img.save results to disk
    results_dir = 'images/pc'
    for i, img in enumerate(pc_images):
        img = Img.scale(img, zero=True, eps=np.finfo(np.float32).eps)
        Img.save("{}/pc_eigen{}_band{}.png".format(results_dir, n, i+1), img, dtype=np.uint8)

    for i, img in enumerate(pc_diffs):
        Img.save("{}/pc_eigen{}_band{}_diff.png".format(results_dir, n, i+1), img, dtype=np.uint8)
