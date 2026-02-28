import numpy as np

n = 1000
mean = np.array([0.75, -0.25])
covariance = np.array([
    [1.0, 0.5],
    [0.5, 2.0] 
])

output_file = "gaussian_samples"
random_seed = 100

# Generate samples from 2D Gaussian
# Shape: (n, 2) where each row is [x, y]
samples = np.random.multivariate_normal(mean, covariance, size=n)
np.save(output_file, samples)

# Print summary statistics
print(f"\nSample statistics:")
print(f"  Shape: {samples.shape}")
print(f"  Mean: {np.mean(samples, axis=0)}")
print(f"  Covariance:\n{np.cov(samples.T)}")