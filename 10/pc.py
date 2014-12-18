import argparse
from PIL import Image
import numpy as np

# Scale the data between 0 and 255
def scale_data(data, zero=True, eps=np.finfo(np.float32).eps):
    min_value = np.min(data)
    scaled_data = data - min_value
    if zero: scaled_data[scaled_data <= eps] = 0.0
    max_value = np.max(scaled_data)
    if max_value != 0.0: scaled_data = scaled_data * (255./max_value)
    return scaled_data

# Show an image
def show(img):
    Image.fromarray(scale_data(img)).show()

# Save an image to disk
def save(path, img):
    Image.fromarray(scale_data(img).astype(np.uint8)).save(path)

# Principal components
def principal_components(images, n=None, debug=False, showr=False, showdiff=False):
    """
    images:    Array of image components.
    n:         Number of eigen values/vector to keep (None means keep them all).
    debug:     Enable/Disable debug messages.
    showr:     Show reconstructed images.
    showdiff:  Show difference between original and reconstructed images.
    """

    np.set_printoptions(linewidth=150)

    def pprint(str, x):
        if debug: print "{} {}\n{}\n".format(str, x.shape, x)

    def remove_eigen_vectors(A, n):
        r = np.copy(A)
        r[-n:] = 0.0
        return r

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
        if showr:    show(image)
        if showdiff: show(diff)

    return reconstructed, diffs


# Main
if __name__ == "__main__":

    # Available args
    parser = argparse.ArgumentParser(description='Image description by Principal Components (PC)')

    parser.add_argument('-n', type=int, default=None,
        help='Number of eigen vectors to remove'
    )

    parser.add_argument('--debug', action='store_true',
        help='Enable debug messages'
    )

    parser.add_argument('--show', action='store_true',
        help='Display reconstructed images'
    )

    parser.add_argument('--diff', action='store_true',
        help='Display the difference between original and reconstructed images'
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

    images = np.array([np.array(Image.open(p).convert('L'), np.uint8) for p in image_paths])

    # Principal components
    pc_images, pc_diffs = principal_components(images, args.n, debug=args.debug, showr=args.show, showdiff=args.diff)

    # Save results to disk
    results_dir = 'images/pc'
    n = args.n if args.n is not None else len(image_paths)
    for i, img in enumerate(pc_images):
        save("{}/pc_eigen{}_band{}.tif".format(results_dir, n, i+1), img)

    for i, img in enumerate(pc_diffs):
        save("{}/pc_eigen{}_band{}_diff.tif".format(results_dir, n, i+1), img)
