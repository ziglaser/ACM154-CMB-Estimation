import numpy as np
import energy_score
import scipy.stats as st
import matplotlib.pyplot as plt
import time

seed = 0
#rng = np.random.default_rng(seed)
rng = np.random.default_rng()

#load the data
#data_array = np.zeros((1,10)) #update this with the thinned and reshaped array of shape (3, n)
f = np.load(f"unlensed_cmb_hmc_chains_seed{seed}_gaussianprior.npz")
num_params = 3
h0_chains = f["h0_chains"]
ombh2_chains = f["ombh2_chains"]
omch2_chains = f["omch2_chains"]
thin_factor = 3
thinned = h0_chains[:, ::thin_factor]  # let numpy determine the size
num_chains, num_thinned_samples = thinned.shape
final_matrix = np.empty((num_chains, num_params, num_thinned_samples))
final_matrix[:, 0, :] = h0_chains[:,::thin_factor]
final_matrix[:, 1, :] = ombh2_chains[:,::thin_factor]
final_matrix[:, 2, :] = omch2_chains[:,::thin_factor]
reshaped = final_matrix.transpose(1, 0, 2).reshape(num_params, -1)
print("reshaped shape", reshaped.shape)

#this samples from the MCMC distribution
def sample_p1(sample_size, key = None, data_array = reshaped, rng = rng):
    n = data_array.shape[1]
    indices = rng.choice(n, size=sample_size, replace=True)
    return (data_array[:, indices].transpose(), rng) #this samples jointly, as required for correctly using MCMC samples

#this samples from the VI distribution
gaussian_file = np.load(f"gaussian_vi_seed{seed}.npz")
mean = gaussian_file["means"]
cov = gaussian_file["cov"]
gaussian_file.close()

def sample_p2(sample_size, key = None, mean = mean, cov = cov):
    return (st.multivariate_normal.rvs(mean = mean, cov = cov, size = sample_size, random_state = rng), rng)

#get the r3 norm of a vector
def abs_fn(x):
    return np.linalg.norm(x, axis=-1)

es_old = energy_score.old_energy_square_distance(sample_p1, sample_p2, abs_fn, 1000)
es = energy_score.energy_square_distance(sample_p1, sample_p2, abs_fn, 4000)
print("es squared old", es_old)
print("es squared", es)
