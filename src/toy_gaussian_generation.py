import numpy as np

n = 1000
mean = np.array([0.75, -0.25])
covariance = np.array([
    [1.0, 0.5],
    [0.5, 2.0] 
])

samples = np.random.multivariate_normal(mean, covariance, size=n)
np.save("gaussian_samples.npz", samples)