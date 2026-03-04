import numpy as np
import matplotlib.pyplot as plt
from triangle_plotter_v2 import triangle_plot

true_h0 = 67.37
true_omch2 = 0.1198
true_ombh2 = 0.02233

seed = 0
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


param_names = [r'$H_0$', r'$\Omega_b h^2$', r'$\Omega_c h^2$']
true_values = [true_h0, true_ombh2, true_omch2]


f.close()

gaussian_file = np.load(f"gaussian_vi_seed{seed}.npz")
mean = gaussian_file["means"]
cov = gaussian_file["cov"]
gaussian_file.close()

gaussian_variables = (mean, cov)
fig_name = f'vi_mcmc_comparison_triangleplot_seed{seed}.png'

fig, ax = triangle_plot(final_matrix, param_names, thin_factor = thin_factor, true_values = true_values, gaussian = gaussian_variables, fig_name = fig_name)
plt.show()