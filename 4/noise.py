### Noise generators ###

import numpy as np

# Gaussian noise generator
class GaussianNoise(object):

    def __init__(self, mu=0, sigma=1):
        self.mu = mu        # Mean
        self.sigma = sigma  # Standard deviation

    def sample(self, x, y):
        return np.random.normal(self.mu, self.sigma)


# Uniform noise generator
class UniformNoise(object):

    def __init__(self, min=0, max=255):
        self.min = min # Lower bound
        self.max = max # Upper bound

    def sample(self, x , y):
        return np.random.uniform(self.min, self.max)
