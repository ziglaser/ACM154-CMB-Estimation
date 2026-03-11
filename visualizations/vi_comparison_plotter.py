import numpy as np
import matplotlib.pyplot as plt
from triangle_plotter_v2 import triangle_plot
from scipy.stats import gaussian_kde
from scipy.optimize import minimize


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
losses = gaussian_file['losses']
gaussian_file.close()

fig, ax = plt.subplots()
epochs = 5 * np.arange(losses.shape[0])
ax.plot(epochs, losses)
ax.set_xlabel("epoch")
ax.set_ylabel("ELBO loss")
ax.set_title("ELBO loss vs training epoch for Gaussian VI")
plt.savefig("gaussian_vi_losses.png")
plt.show()

gaussian_variables = (mean, cov)
fig_name = f'vi_mcmc_comparison_triangleplot_seed{seed}.png'

fig, ax = triangle_plot(final_matrix, param_names, thin_factor = thin_factor, true_values = true_values, gaussian = gaussian_variables, fig_name = fig_name)
plt.show()

#print("means", mean)
true_values = np.array([true_h0, true_ombh2, true_omch2])

squared_error = np.sum(((true_values - mean)/true_values)**2)
print("VI squared error", squared_error)


print("final matrix shape", final_matrix.shape)

samples_flat = final_matrix.transpose(1, 0, 2).reshape(num_params, -1)
kde = gaussian_kde(samples_flat)
x0 = samples_flat[:, np.argmax(kde.logpdf(samples_flat))]
result = minimize(lambda x: -kde.logpdf(x.reshape(-1, 1)), x0, method='Nelder-Mead')
map_estimate = result.x

print("kernel density estimation map estimate: ", map_estimate)


squared_error = np.sum(((true_values - map_estimate)/true_values)**2)
print("MCMC kde squared error", squared_error)