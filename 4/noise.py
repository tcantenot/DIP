### Noise generators ###

import numpy as np

def GaussianNoise(mu, sigma, shape):
    return np.random.normal(mu, sigma, shape)

def UniformNoise(min, max):
    return np.random.uniform(min, max, shape)
