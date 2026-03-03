import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats

import blackjax

from datetime import date


rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))

#toy model: gaussian data with parameters corresponding to the means.
toy_data = np.load("gaussian_samples.npy")
covariance = jnp.array([
    [1.0, 0.5],   # variance in x = 1.0, covariance = 0.0
    [0.5, 2.0]    # covariance = 0.0, variance in y = 1.0
])
observed = toy_data[0,:]
print("observed data", observed)

#posteriors
def logdensity_fn(mean1, mean2, observed=observed):
    """Multivariate Normal"""
    likelihood = stats.multivariate_normal.logpdf(observed,jnp.array((mean1, mean2)), covariance)
    prior = stats.norm.logpdf(mean1) + stats.norm.logpdf(mean2)
    return likelihood + prior
logdensity = lambda x: logdensity_fn(**x)


#sampler parameters
inv_mass_matrix = np.array([0.5, 0.01])
num_integration_steps = 60
step_size = 1e-3

#hmc
hmc = blackjax.hmc(logdensity, step_size, inv_mass_matrix, num_integration_steps)

#set the initial state
#initial_position = {"loc": 1.0, "log_scale": 1.0}
initial_position = {"mean1": 0.0, "mean2": 0.0}
initial_state = hmc.init(initial_position)
initial_state

#HMCState(position={'loc': 1.0, 'log_scale': 1.0}, logdensity=Array(-35387.848, dtype=float32), logdensity_grad={'loc': Array(1354.3645, dtype=float32, weak_type=True), 'log_scale': Array(65940.82, dtype=float32, weak_type=True)})


#first demonstrate the problem with HMC
hmc_kernel = jax.jit(hmc.step)

def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states

rng_key, sample_key = jax.random.split(rng_key)
states = inference_loop(sample_key, hmc_kernel, initial_state, 10_000)

mcmc_samples = states.position
print("type of mcmc samples: ", type(mcmc_samples))

fig, (ax, ax1) = plt.subplots(ncols=2, figsize=(15, 6))
ax.plot(mcmc_samples["mean1"])
ax.set_xlabel("Samples")
ax.set_ylabel("mean1")

ax1.plot(mcmc_samples["mean2"])
ax1.set_xlabel("Samples")
ax1.set_ylabel("mean2")

plt.show()
plt.close()



#now do the problem with NUTS
warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
(state, parameters), _ = warmup.run(warmup_key, initial_position, num_steps=1000)

#print("keys of state", state.keys)
print("keys of parameters", parameters.keys())

inv_mass_matrix = parameters["inverse_mass_matrix"]
step_size = parameters["step_size"]
print("adapted inv mass matrix ", inv_mass_matrix)
print()
print("adapted step size ", step_size)
#inv_mass_matrix = np.array([0.5, 0.01])
#step_size = 1e-3

nuts = blackjax.nuts(logdensity, step_size, inv_mass_matrix)

#initial_position = {"loc": 1.0, "log_scale": 1.0}
initial_position = {"mean1": 0.0, "mean2": 0.0}
initial_state = nuts.init(initial_position)

rng_key, sample_key = jax.random.split(rng_key)
states = inference_loop(sample_key, nuts.step, initial_state, 10_000)

mcmc_samples = states.position

#get the analytic solution for the toy model
identity = jnp.array([
    [1.0, 0.0],
    [0.0, 1.0]
])
inv_cov = jnp.linalg.inv(covariance)
posterior_covariance = jnp.linalg.inv(identity + inv_cov)
posterior_mean = jnp.matmul(posterior_covariance, jnp.matmul(inv_cov, observed))
print("posterior mean ", posterior_mean)
print()
print("posterior covariance ", posterior_covariance)

#now plot the actual solution
fig, (ax, ax1) = plt.subplots(ncols=2, figsize=(15, 6))
mean1_samples = mcmc_samples["mean1"]
mean2_samples = mcmc_samples["mean2"]
ax.plot(mean1_samples)

ax.set_xlabel("Samples")
ax.set_ylabel("mean1")
ax.hlines(y = posterior_mean[0], xmin = 0, xmax = mean1_samples.shape[0], colors = 'k')
ax.hlines(y = posterior_mean[0] + jnp.sqrt(posterior_covariance[0,0]), xmin = 0, xmax = mean1_samples.shape[0], colors = 'r')
ax.hlines(y = posterior_mean[0] - jnp.sqrt(posterior_covariance[0,0]), xmin = 0, xmax = mean1_samples.shape[0], colors = 'r')

ax1.plot(mcmc_samples["mean2"])
ax1.hlines(y = posterior_mean[1], xmin = 0, xmax = mean1_samples.shape[0], colors = 'k')
ax1.hlines(y = posterior_mean[1] + jnp.sqrt(posterior_covariance[1,1]), xmin = 0, xmax = mean1_samples.shape[0], colors = 'r')
ax1.hlines(y = posterior_mean[1] - jnp.sqrt(posterior_covariance[1,1]), xmin = 0, xmax = mean1_samples.shape[0], colors = 'r')
ax1.set_xlabel("Samples")
ax1.set_ylabel("mean2")

plt.show()
plt.close()

#make a contour plot
# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# 1. Create 2D histogram of MCMC samples
hist = ax.hist2d(mean1_samples, mean2_samples, bins=50, cmap='Blues', alpha=0.7, density=True)
ax.set_xlabel("mean1")
ax.set_ylabel("mean2")
plt.colorbar(hist[3], ax=ax, label='MCMC Sample Density')
#plt.show()


def analytic_posterior(mean1, mean2):
    means = jnp.array((mean1, mean2))
    return stats.multivariate_normal.pdf(means, posterior_mean, posterior_covariance)

vectorized_analytic_posterior = np.vectorize(analytic_posterior)
# 2. Create grid for analytic posterior contours
x_min, x_max = -1, 4
y_min, y_max = -2, 2
x_grid = np.linspace(x_min, x_max, 50)
y_grid = np.linspace(y_min, y_max, 50)
X, Y = np.meshgrid(x_grid, y_grid)

# Evaluate analytic posterior on grid
Z = vectorized_analytic_posterior(X, Y)

# 3. Overlay contours of analytic posterior
contours = ax.contour(X, Y, Z, levels=8, colors='red', linewidths=2, alpha=0.6)
ax.clabel(contours, inline=True, fontsize=8)

plt.show()

#code structure:

#to get probability of a map given parameter values:
#1) convert map to the appropriate fourier transform
#2) use cosmopower to get the power spectrum given parameters
#3) use the power spectrum to get the correct multivariate gaussian in fourier space
#4) directly compute the logpdf

#to use MCMC:
#1) plug this into blackjax HMC algorithm to do parameter inference

