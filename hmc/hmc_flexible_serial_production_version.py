import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import blackjax
from datetime import date
import scipy.fft as fft
from scipy.interpolate import CubicSpline
from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ
from interpax import interp1d
import time
from scipy import integrate
from triangle_plotter import triangle_plot

import scipy.stats as st
import warnings

warnings.filterwarnings("ignore")

def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, (state, info.acceptance_rate)
    keys = jax.random.split(rng_key, num_samples)
    _, (states, acceptance_rates) = jax.lax.scan(one_step, initial_state, keys)
    return states, acceptance_rates

def run_nuts(logdensity, num_warmup_steps, num_steps, warmup_initial_position, initial_positions, rng_key, num_chains, num_params):
    #warmup the sampler
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity, is_mass_matrix_diagonal = False)
    rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
    (state, parameters), _ = warmup.run(warmup_key, warmup_initial_position, num_steps=num_warmup_steps)

    #printed the adapted mass matrix and step size
    inv_mass_matrix = parameters["inverse_mass_matrix"]
    step_size = parameters["step_size"]
    print("adapted inv mass matrix ", inv_mass_matrix)
    print()
    print("adapted step size ", step_size)

    #set up the nuts sampler
    nuts = blackjax.nuts(logdensity, step_size, inv_mass_matrix)

    #this requires the parameters to be a 1 dimensional array
    chain_samples = []
    acceptance_rates = []
    for i in range(num_chains):
        initial_state = nuts.init(initial_positions[i])
        rng_key, sample_key = jax.random.split(rng_key)
        #run the sampler
        states, acceptance_rate = inference_loop(sample_key, nuts.step, initial_state, num_steps)
        mcmc_samples = states.position
        chain_samples.append(mcmc_samples)
        acceptance_rates.append(acceptance_rate)
    return chain_samples, acceptance_rates

def run_hmc(logdensity, num_steps, inv_mass_matrix, num_integration_steps, step_size, initial_position, rng_key):
    #set up the hmc sampler
    hmc = blackjax.hmc(logdensity, step_size, inv_mass_matrix, num_integration_steps)
    initial_state = hmc.init(initial_position)
    initial_state
    hmc_kernel = jax.jit(hmc.step)
    rng_key, sample_key = jax.random.split(rng_key)

    #run the sampler
    states = inference_loop(sample_key, hmc_kernel, initial_state, num_steps)
    mcmc_samples = states.position
    return mcmc_samples

def load_toy_data(rng_key):
    #toy model: gaussian data with parameters corresponding to the means.
    toy_data = np.load("gaussian_samples.npy")
    observed = toy_data[0,:]
    return observed

def toy_model_main():
    #rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))
    rng_key = jax.random.key(0)

    observed = load_toy_data(rng_key)
    print("observed data: ", observed)
    covariance = jnp.array([
        [1.0, 0.5],   # variance in x = 1.0, covariance = 0.5
        [0.5, 2.0]    # covariance = 0.5, variance in y = 2.0
    ])

    def toy_logdensity_fn(mean1, mean2, observed=observed, covariance=covariance):
        likelihood = stats.multivariate_normal.logpdf(observed,jnp.array((mean1, mean2)), covariance)
        prior = stats.norm.logpdf(mean1) + stats.norm.logpdf(mean2)
        return likelihood + prior

    logdensity = lambda x: toy_logdensity_fn(**x)

    rng = np.random.default_rng()
    #samples = rng.uniform(low, high, size)

    num_chains = 3
    num_params = 2
    num_steps = 10000
    warmup_initial_position = {"mean1": 0.0, "mean2": 0.0}

    initial_positions = []
    for i in range(num_chains):
        initial_positions.append({"mean1": rng.uniform(-1, 1), "mean2": rng.uniform(-1, 1)})

    #initial_positions = {"mean1": rng.uniform(-1, 1, num_chains), "mean2": rng.uniform(-1, 1, num_chains)}
    chain_samples, acceptance_rates = run_nuts(logdensity, 1000, num_steps, warmup_initial_position, initial_positions, rng_key, num_chains, num_params)

    acceptance_rates_matrix = np.empty((num_chains, num_steps))
    for i in range(num_chains):
        acceptance_rates_matrix[i,:] = acceptance_rates[i][:]
        print(f"mean acceptance rate for chain {i}: ", np.mean(acceptance_rates[i][:]))


    #get chain plots for toy model
    fig, (ax, ax1) = plt.subplots(ncols=2, figsize=(15, 6))
    for i in range(num_chains):
        ax.plot(chain_samples[i]["mean1"])
        ax.set_xlabel("Samples")
        ax.set_ylabel("mean1")
        ax1.plot(chain_samples[i]["mean2"])
        ax1.set_xlabel("Samples")
        ax1.set_ylabel("mean2")
    #plt.show()
    plt.close()

    #get the analytic solution for the toy model
    identity = jnp.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    inv_cov = jnp.linalg.inv(covariance)
    posterior_covariance = jnp.linalg.inv(identity + inv_cov)
    posterior_mean = jnp.matmul(posterior_covariance, jnp.matmul(inv_cov, observed))
    print("true posterior mean ", posterior_mean)
    print()
    print("true posterior covariance ", posterior_covariance)

    #get chain plots with analytic posterior means and standard deviations overlaid
    fig, (ax, ax1) = plt.subplots(ncols=2, figsize=(15, 6))
    all_mean1_samples = []
    all_mean2_samples = []
    for i in range(num_chains):
        mean1_samples = chain_samples[i]["mean1"]
        mean2_samples = chain_samples[i]["mean2"]
        all_mean1_samples.append(mean1_samples)
        all_mean2_samples.append(mean2_samples)
        ax.plot(mean1_samples)
        ax.set_xlabel("Samples")
        ax.set_ylabel("mean1")
        ax.hlines(y = posterior_mean[0], xmin = 0, xmax = mean1_samples.shape[0], colors = 'k')
        ax.hlines(y = posterior_mean[0] + jnp.sqrt(posterior_covariance[0,0]), xmin = 0, xmax = mean1_samples.shape[0], colors = 'r')
        ax.hlines(y = posterior_mean[0] - jnp.sqrt(posterior_covariance[0,0]), xmin = 0, xmax = mean1_samples.shape[0], colors = 'r')
        ax1.plot(chain_samples[i]["mean2"])
        ax1.hlines(y = posterior_mean[1], xmin = 0, xmax = mean1_samples.shape[0], colors = 'k')
        ax1.hlines(y = posterior_mean[1] + jnp.sqrt(posterior_covariance[1,1]), xmin = 0, xmax = mean1_samples.shape[0], colors = 'r')
        ax1.hlines(y = posterior_mean[1] - jnp.sqrt(posterior_covariance[1,1]), xmin = 0, xmax = mean1_samples.shape[0], colors = 'r')
        ax1.set_xlabel("Samples")
        ax1.set_ylabel("mean2")
    #plt.show()
    plt.close()

    mean1_chains = np.empty((len(all_mean1_samples), all_mean1_samples[0].shape[0]))
    mean2_chains = np.empty((len(all_mean2_samples), all_mean2_samples[0].shape[0]))
    for i in range(num_chains):
        mean1_chains[i,:] = all_mean1_samples[i][:]
        mean2_chains[i,:] = all_mean2_samples[i][:]

    print("r value mean1 samples", blackjax.diagnostics.potential_scale_reduction(mean1_chains))
    print("r value mean2 samples", blackjax.diagnostics.potential_scale_reduction(mean2_chains))
    print("ess value mean1 samples", blackjax.diagnostics.effective_sample_size(mean1_chains))
    print("ess value mean2 samples", blackjax.diagnostics.effective_sample_size(mean2_chains))
    #make a contour plot
    fig, ax = plt.subplots(figsize=(10, 8))
    hist = ax.hist2d(mean1_chains.flatten(), mean2_chains.flatten(), bins=50, cmap='Blues', alpha=0.7, density=True)
    ax.set_xlabel("mean1")
    ax.set_ylabel("mean2")
    plt.colorbar(hist[3], ax=ax, label='MCMC Sample Density')

    def analytic_posterior(mean1, mean2):
        means = jnp.array((mean1, mean2))
        return stats.multivariate_normal.pdf(means, posterior_mean, posterior_covariance)

    vectorized_analytic_posterior = np.vectorize(analytic_posterior)
    # Create grid for analytic posterior contours
    x_min, x_max = -1, 4
    y_min, y_max = -2, 2
    x_grid = np.linspace(x_min, x_max, 50)
    y_grid = np.linspace(y_min, y_max, 50)
    X, Y = np.meshgrid(x_grid, y_grid)
    # Evaluate analytic posterior on grid
    Z = vectorized_analytic_posterior(X, Y)
    # Overlay contours of analytic posterior
    contours = ax.contour(X, Y, Z, levels=8, colors='red', linewidths=2, alpha=0.6)
    ax.clabel(contours, inline=True, fontsize=8)

    #plt.show()

def unlensed_cmb_main(seed, data, true_h0, true_ombh2, true_omch2, noise_level):
    rng_key = jax.random.key(0)
    npix = 64

    #get pixel size
    pixel_size = 8 #pixel size IN ARCMIN (ie, arcmin per pixel)
    radian_per_arcmin = np.pi/(60*180)
    pixel_size = pixel_size * radian_per_arcmin #pixel size IN RADIANS (ie, radians per pixel)
    
    #the julia map scales by 1/pixel size from the data generation process represented by the logpdf below.
    #This introduces a normalization constant from the Jacobian that has no dependence on data or model parameters, so we omit it.
    unlensed_cmb_map = jnp.array(data * pixel_size)
    unlensed_cmb_fouriers = fft.rfft2(unlensed_cmb_map, norm = "ortho") #convert to an rfft map

    #get fourier wave modes
    kx = 2 * np.pi * fft.fftfreq(npix, d=pixel_size) #d is the sample spacing in radians
    ky = 2 * np.pi * fft.rfftfreq(npix, d=pixel_size)
    ky, kx = np.meshgrid(ky, kx)
    k = np.sqrt(kx**2+ky**2)
    fourier_k = jnp.array(k) #compute the appropriate grid of fourier k
    self_inverse_indices = [0, npix//2]

    #get required masks
    self_inverse_mask = np.zeros_like(k)
    for i in self_inverse_indices:
        for j in self_inverse_indices:
            self_inverse_mask[i,j] = 1
    self_inverse_mask = jnp.array(self_inverse_mask)

    def make_upper_mask(npix, self_inverse_indices): #this is 1 for upper indices and 0 for self inverse indices
        mask = np.ones((npix, npix//2+1))
        self_inverse_indices = [0, npix//2]
        #set self inverse indices to zero
        for i in self_inverse_indices:
            for j in self_inverse_indices:
                mask[i,j] = 0
        #zero out all indices that are determined by conjugates elsewhere in the map
        mask[npix//2+1:, 0] = 0
        mask[npix//2+1:, -1] = 0
        return mask

    upper_mask = make_upper_mask(npix, self_inverse_indices)
    zero_mode_mask = np.ones_like(upper_mask)
    zero_mode_mask[0,0] = 0
    zero_mode_mask = jnp.array(zero_mode_mask)

    noise_variances = np.ones_like(fourier_k) * noise_level**2
    
    @jax.jit
    def get_logpdfs(sigmas):
        real_logpdfs = jnp.nan_to_num(stats.norm.logpdf(unlensed_cmb_fouriers.real, loc = 0, scale = sigmas/jnp.sqrt(2)))
        imag_logpdfs = jnp.nan_to_num(stats.norm.logpdf(unlensed_cmb_fouriers.imag, loc = 0, scale = sigmas/jnp.sqrt(2)))
        self_inverse_logpdfs = jnp.nan_to_num(stats.norm.logpdf(unlensed_cmb_fouriers.real, loc = 0, scale = sigmas))
        return jnp.sum(zero_mode_mask*self_inverse_mask*self_inverse_logpdfs + zero_mode_mask*upper_mask*real_logpdfs + zero_mode_mask*upper_mask*imag_logpdfs)
    
    @jax.jit
    def logdensity_fn(h0, ombh2, omch2):
        cosmo_params = jnp.array([ombh2, omch2, h0/100, .0540, .9652, np.log(10*2.08666022)])
        emulator = CPJ(probe='cmb_tt')
        spectrum = emulator.predict(cosmo_params)
        unitful_factor = 7428350250000.0
        spectrum = spectrum*unitful_factor
        emulator_ell = emulator.modes
        map_variances = interp1d(fourier_k.flatten(), emulator_ell, spectrum, method="cubic")
        map_variances = jnp.reshape(map_variances, (64, 33)) 
        sigmas = jnp.sqrt(map_variances + noise_variances)
        likelihood_logpdf = get_logpdfs(sigmas) #compute likelihood pdfs
        prior_logpdf = stats.norm.logpdf(h0, loc = 67.37, scale = 40) + stats.norm.logpdf(ombh2, loc = 0.02233, scale = 0.01) + stats.norm.logpdf(omch2, loc = 0.1198, scale = 0.06)
        return likelihood_logpdf + prior_logpdf

    logdensity = lambda x: logdensity_fn(**x)

    num_chains = 5
    num_params = 3
    warmup_initial_position = {"h0": 67.37, "ombh2": .02233, "omch2": 0.1198}
    rng = np.random.default_rng()
    initial_positions = []
    for i in range(num_chains):
        initial_positions.append({"h0": 67.37 + rng.uniform(-6, 6), "ombh2": .02233 + rng.uniform(-.002, .002), "omch2": 0.1198 + rng.uniform(-0.01, 0.01)})

    num_warmup_steps = 1500
    num_samples = 10000
    '''num_warmup_steps = 10
    num_samples = 20'''
    start = time.time()
    chain_samples, acceptance_rates = run_nuts(logdensity, num_warmup_steps, num_samples, warmup_initial_position, initial_positions, rng_key, num_chains, num_params)
    end = time.time()
    print("mcmc time: ", end - start)
    

    acceptance_rates_matrix = np.empty((num_chains, num_samples))
    for i in range(num_chains):
        acceptance_rates_matrix[i,:] = acceptance_rates[i][:]
        print(f"mean acceptance rate for chain {i}: ", np.mean(acceptance_rates[i][:]))
    np.save(f"acceptance_rates_matrix_seed{seed}.npy", acceptance_rates_matrix)

    #get chain plots
    all_h0_samples = []
    all_ombh2_samples = []
    all_omch2_samples = []
    for i in range(num_chains):
        h0_samples = chain_samples[i]["h0"]
        ombh2_samples = chain_samples[i]["ombh2"]
        omch2_samples = chain_samples[i]["omch2"]
        all_h0_samples.append(h0_samples)
        all_ombh2_samples.append(ombh2_samples)
        all_omch2_samples.append(omch2_samples)

    h0_chains = np.empty((len(all_h0_samples), all_h0_samples[0].shape[0]))
    ombh2_chains = np.empty((len(all_ombh2_samples), all_ombh2_samples[0].shape[0]))
    omch2_chains = np.empty((len(all_omch2_samples), all_omch2_samples[0].shape[0]))
    for i in range(num_chains):
        h0_chains[i,:] = all_h0_samples[i][:]
        ombh2_chains[i,:] = all_ombh2_samples[i][:]
        omch2_chains[i,:] = all_omch2_samples[i][:]

    np.savez(f"unlensed_cmb_hmc_chains_seed{seed}_gaussianprior.npz", h0_chains = h0_chains, ombh2_chains = ombh2_chains, omch2_chains = omch2_chains)

    thin_factor = 3
    thinned = h0_chains[:, ::thin_factor]  # let numpy determine the size
    num_chains, num_thinned_samples = thinned.shape
    final_matrix = np.empty((num_chains, num_params, num_thinned_samples))
    final_matrix[:, 0, :] = h0_chains[:,::thin_factor]
    final_matrix[:, 1, :] = ombh2_chains[:,::thin_factor]
    final_matrix[:, 2, :] = omch2_chains[:,::thin_factor]

    param_names = [r'$H_0$', r'$\Omega_b h^2$', r'$\Omega_c h^2$']
    true_values = [true_h0, true_ombh2, true_omch2]
    fig_name = f'mcmc_triangle_plot_seed{seed}_gaussianprior.png'
    fig, ax = triangle_plot(final_matrix, param_names, thin_factor = thin_factor, true_values = true_values, fig_name = fig_name)

    print("r stat h0:", blackjax.diagnostics.potential_scale_reduction(h0_chains[:,::thin_factor]))
    print("r stat ombh2:", blackjax.diagnostics.potential_scale_reduction(ombh2_chains[:,::thin_factor]))
    print("r stat omch2:", blackjax.diagnostics.potential_scale_reduction(omch2_chains[:,::thin_factor]))
    print("effective sample size h0:", blackjax.diagnostics.effective_sample_size(h0_chains[:,::thin_factor]))
    print("effective sample size ombh2:", blackjax.diagnostics.effective_sample_size(ombh2_chains[:,::thin_factor]))
    print("effective sample size omch2:", blackjax.diagnostics.effective_sample_size(omch2_chains[:,::thin_factor]))
    print("num thinned samples * num_chains", num_thinned_samples * num_chains)

def generate_cosmopower_map(seed, h0, omch2, ombh2, noise_level = 10**-8):
    npix = 64
    rng = np.random.default_rng(seed)
    #rng = np.random.default_rng()
    pixel_size = 8 #pixel size IN ARCMIN (ie, arcmin per pixel)
    radian_per_arcmin = np.pi/(60*180)
    pixel_size = pixel_size * radian_per_arcmin #pixel size IN RADIANS (ie, radians per pixel)
    kx = 2 * np.pi * fft.fftfreq(npix, d=pixel_size) #d is the sample spacing in radians
    ky = 2 * np.pi * fft.rfftfreq(npix, d=pixel_size)
    ky, kx = np.meshgrid(ky, kx)
    k = np.sqrt(kx**2+ky**2)
    fourier_k = k #compute the appropriate grid of fourier k
    self_inverse_indices = [0, npix//2]

    def make_upper_mask(npix, self_inverse_indices): #this is 1 for upper indices and 0 for self inverse indices
        mask = np.ones((npix, npix//2+1))
        self_inverse_indices = [0, npix//2]
        #set self inverse indices to zero
        for i in self_inverse_indices:
            for j in self_inverse_indices:
                mask[i,j] = 0
        #zero out all indices that are determined by conjugates elsewhere in the map
        mask[npix//2+1:, 0] = 0
        mask[npix//2+1:, -1] = 0
        return mask

    upper_mask = make_upper_mask(npix, self_inverse_indices)
    zero_mode_mask = np.ones_like(upper_mask)
    zero_mode_mask[0,0] = 0

    self_inverse_mask = np.zeros_like(k)
    for i in self_inverse_indices:
        for j in self_inverse_indices:
            self_inverse_mask[i,j] = 1

    cosmo_params = jnp.array([ombh2, omch2, h0/100, .0540, .9652, np.log(10*2.08666022)])
    emulator = CPJ(probe='cmb_tt')
    spectrum = emulator.predict(cosmo_params)
    emulator_ell = emulator.modes
    unitful_factor = 7428350250000.0 #this converts from a unitless map (in CAMB convention) to a map in Kelvin
    spectrum = spectrum*unitful_factor

    sigmas = interp1d(fourier_k.flatten(), emulator_ell, spectrum, method="cubic")
    sigmas = np.reshape(sigmas, (64, 33))
    sigmas = np.sqrt(sigmas)
    sigmas[0,0] = 0

    logdet_term1 = 2 * jnp.sum(zero_mode_mask * self_inverse_mask * np.nan_to_num(jnp.log(sigmas)))
    logdet_term2 = 4 * jnp.sum(zero_mode_mask * upper_mask * jnp.nan_to_num(jnp.log(sigmas/jnp.sqrt(2))))
    logdet = -1.0 * (logdet_term1 + logdet_term2)
    twopi_term = - npix**2 / 2 * jnp.log(2 * jnp.pi)
    const = - 2.0 * (twopi_term + logdet)

    #generate some white noise
    white_noise = st.norm.rvs(loc = 0, scale = 1/np.sqrt(2), size = (2, npix, npix//2 + 1), random_state = rng) + 1j * st.norm.rvs(loc = 0, scale = 1/np.sqrt(2), size = (2, npix, npix//2+1), random_state = rng)

    #make the necessary ones real
    for i in self_inverse_indices:
        for j in self_inverse_indices:
            white_noise[:, i, j] = st.norm.rvs(loc = 0, scale = 1, random_state = rng)

    #make the necessary ones conjugate
    for i in range(npix//2+1, npix):
        white_noise[:, i, 0] = np.conjugate(white_noise[:, npix-i, 0])
        white_noise[:, i, -1] = np.conjugate(white_noise[:, npix-i, -1])

    #scale the white noise to the correct spectrum to get the true map
    fourier_map_coeffs = white_noise[0,:,:] * sigmas
    fourier_map_coeffs[0,0] = 0.0

    #scale the white noise to get the correct observation noise
    fourier_noise_coeffs = white_noise[1,:,:] * noise_level
    fourier_noise_coeffs[0,0] = 0.0

    #add the noises to get the observed map (truth + observation noise)
    fourier_coeffs = fourier_map_coeffs + fourier_noise_coeffs

    #get the maps in pixel space
    map = fft.irfft2(fourier_coeffs, s = (npix,npix), norm = "ortho")
    map = map/pixel_size

    np.save(f"hmc_unlensed_map_seed{seed}.npy", map)
    print("map saved")

    return map, const

def looper():
    true_h0 = 67.37
    true_omch2 = 0.1198
    true_ombh2 = 0.02233
    noise_level = 0.08

    for i in range(5):
        print("doing seed i", i)
        map, const = generate_cosmopower_map(i, true_h0, true_omch2, true_ombh2, noise_level)
        unlensed_cmb_main(i, map, true_h0, true_ombh2, true_omch2, noise_level) #the map units should be in julia convention here
        print()

looper()
#toy_model_main()
