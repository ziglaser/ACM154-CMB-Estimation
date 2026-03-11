import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.scipy.stats import norm, multivariate_normal
import optax
import numpy as np
import matplotlib.pyplot as plt
from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ
from interpax import interp1d
import scipy.fft as fft
import time
from functools import partial

# --- Variational parameter initialisers ---
def init_diagonal_params(init_m: jnp.ndarray) -> dict:
    d = len(init_m)
    return {"mean": jnp.array(init_m), "log_sigma": jnp.zeros(d),}

def init_full_params(init_m: jnp.ndarray, init_lparams: None) -> dict:
    d = len(init_m)
    n_params = d * (d + 1) // 2      # number of lower-triangular entries
    if init_lparams is None:
        return {"mean": jnp.array(init_m), "l_params": jnp.zeros(n_params),}
    else:
        return {"mean": jnp.array(init_m), "l_params": jnp.array(init_lparams),}

# --- Build the Cholesky lower L matrix ---
def scale_tril_diagonal(params: dict) -> jnp.ndarray:
    return jnp.diag(jnp.exp(params["log_sigma"]))

@jax.jit
def scale_tril_full(params: dict) -> jnp.ndarray:
    d  = len(params["mean"])
    lp = params["l_params"]

    # Build mask for lower triangle (including diagonal)
    rows, cols = jnp.tril_indices(d)

    # Place all l_params into a flat lower triangle
    L_flat = jnp.zeros((d, d))
    L_flat = L_flat.at[rows, cols].set(lp)

    # Exponentiate only the diagonal entries
    diag_mask = jnp.eye(d, dtype=bool)
    L = jnp.where(diag_mask, jnp.exp(L_flat), L_flat)

    return L

'''def scale_tril_full(params: dict) -> jnp.ndarray:
    d  = len(params["mean"])
    lp = params["l_params"]
    L  = jnp.zeros((d, d))

    # Fill lower triangle: diagonal entries are exp-reparameterised,
    # off-diagonal entries are unconstrained
    idx = 0
    for i in range(d):
        for j in range(i + 1):
            if i == j:
                L = L.at[i, j].set(jnp.exp(lp[idx]))
            else:
                L = L.at[i, j].set(lp[idx])
            idx += 1
    return L'''

@partial(jit, static_argnums=(1, 3, 4))  # scale_tril_fn, batch_size, nparams
def sample_gaussian(params: dict, scale_tril_fn, key: jax.random.PRNGKey,
                    batch_size: int, nparams: int) -> jnp.ndarray:
    eps = jax.random.normal(key, (batch_size, nparams))
    L   = scale_tril_fn(params)
    return params["mean"] + eps @ L.T

@partial(jit, static_argnums=(1,))  # scale_tril_fn
def log_q(params: dict, scale_tril_fn, x: jnp.ndarray) -> jnp.ndarray:
    L   = scale_tril_fn(params)
    cov = L @ L.T
    return vmap(lambda xi: multivariate_normal.logpdf(xi, mean=params["mean"], cov=cov))(x)

@partial(jit, static_argnums=(1, 3, 4, 5))  # scale_tril_fn, batch_size, nparams, log_prob_fn
def elbo_loss(params: dict, scale_tril_fn, key: jax.random.PRNGKey,
              batch_size: int, nparams: int, log_prob_fn) -> jnp.ndarray:
    x   = sample_gaussian(params, scale_tril_fn, key, batch_size, nparams)
    lq  = log_q(params, scale_tril_fn, x)
    lpi = log_prob_fn(x)
    return jnp.mean(lq - lpi)

'''# --- Sampling (reparameterised) ---
@partial(jit, static_argnums=(1,))
def sample_gaussian(params: dict, scale_tril_fn, key: jax.random.PRNGKey,
                    batch_size: int, nparams: int) -> jnp.ndarray:
    eps = jax.random.normal(key, (batch_size, nparams))               # [N,2]
    L   = scale_tril_fn(params)
    return params["mean"] + eps @ L.T                           # [N,2]


# --- Log q(x) ---
@partial(jit, static_argnums=(1,))
def log_q(params: dict, scale_tril_fn, x: jnp.ndarray) -> jnp.ndarray:
    """Diagonal or full Gaussian log-density at x [N,d]."""
    L   = scale_tril_fn(params)
    cov = L @ L.T
    return vmap(lambda xi: multivariate_normal.logpdf(xi, mean=params["mean"], cov=cov))(x) # [N]


# --- ELBO loss  (mean of  log q - log π) ---
@partial(jit, static_argnums=(1,5))
def elbo_loss(params: dict, scale_tril_fn, key: jax.random.PRNGKey, batch_size: int, nparams: int, log_prob_fn) -> jnp.ndarray:
    x  = sample_gaussian(params, scale_tril_fn, key, batch_size, nparams)
    lq = log_q(params, scale_tril_fn, x) # [N]
    lpi  = log_prob_fn(x) # [N]
    return jnp.mean(lq - lpi) #scalar'''

def run_vi_jax(init_m, cov_type='full', lr=1e-1, num_epochs=100, batch_size=64, show_anim=False, log_prob_fn=None, master_key = jax.random.PRNGKey(0), init_lparams = None, savename = None):
    """
    init_m: jnp.ndarray of shape [d] — sets the problem dimension.
            e.g. jnp.zeros(2) for 2D, jnp.zeros(3) for 3D.
    """
    assert log_prob_fn is not None, "Provide a JAX log_prob_fn(x [N,d]) -> [N]"

    #get the dimension of the inverse problem
    nparams = len(init_m)

    #set up the stuff for the Gaussian approximation
    scale_tril_fn = scale_tril_full if cov_type == 'full' else scale_tril_diagonal #get the function to build the matrix L
    params = init_full_params(init_m, init_lparams) if cov_type == 'full' else init_diagonal_params(init_m, init_lparams) #initialize the gaussian mean and covariance

    #set up the optimizer and hyperparameters
    optimizer  = optax.adam(lr)
    opt_state  = optimizer.init(params)
    record_freq = 5

    #set up the loss function (which is the elbo)
    def step(params, opt_state, key):
        loss, grads = jax.value_and_grad(elbo_loss)(
            params, scale_tril_fn, key, batch_size, nparams, log_prob_fn)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    losses = []
    start = time.time()
    for epoch in range(num_epochs):
        master_key, subkey = jax.random.split(master_key)
        params, opt_state, loss = step(params, opt_state, subkey)
        if epoch%record_freq == 0:
            losses.append(loss)
    end = time.time()
    print("training time: ", end - start)
    
    plt.plot(losses)
    plt.show()
    #print the means of the Gaussian that the optimizer finds
    print("means: ", params["mean"])

    #print the covariance of the Gaussian that the optimizer finds
    L   = scale_tril_fn(params)
    cov = L @ L.T
    print("covariance: ", cov)

    if savename is not None:
        np.savez(savename, means = params["mean"], cov = cov, losses = losses)

def toy_main():
    def load_toy_data():
        #toy model: gaussian data with parameters corresponding to the means.
        toy_data = np.load("gaussian_samples.npy")
        observed = toy_data[0,:]
        return observed

    observed = load_toy_data()
    #init_m = jnp.zeros(2)
    init_m = np.array([1.41, 0.16])
    toy_covariance = jnp.array([[1.0, 0.5], [0.5, 2.0]])

    def toy_logdensity_fn(means, observed, covariance):
        mean1 = means[0]
        mean2 = means[1]
        likelihood = multivariate_normal.logpdf(observed,jnp.array((mean1, mean2)), covariance)
        prior = norm.logpdf(mean1) + norm.logpdf(mean2)
        return likelihood + prior

    bound_log_prob = lambda x: vmap(lambda means: toy_logdensity_fn(means, observed, toy_covariance))(x)
    init_lparams = np.load("toy_init_lparams.npy")

    run_vi_jax(init_m, lr = 1e-4, num_epochs=5000, init_lparams=init_lparams, log_prob_fn=bound_log_prob, savename='toy_Gaussian_VI.npz')

def unlensed_data_main():
    seed = 0
    noise_level = 0
    filename = f"hmc_unlensed_map_seed{seed}.npy"
    data = np.load(filename)
    npix = 64
    noise_level = 0.08

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
    
    @jit
    def get_logpdfs(sigmas):
        real_logpdfs = jnp.nan_to_num(norm.logpdf(unlensed_cmb_fouriers.real, loc = 0, scale = sigmas/jnp.sqrt(2)))
        imag_logpdfs = jnp.nan_to_num(norm.logpdf(unlensed_cmb_fouriers.imag, loc = 0, scale = sigmas/jnp.sqrt(2)))
        self_inverse_logpdfs = jnp.nan_to_num(norm.logpdf(unlensed_cmb_fouriers.real, loc = 0, scale = sigmas))
        return jnp.sum(zero_mode_mask*self_inverse_mask*self_inverse_logpdfs + zero_mode_mask*upper_mask*real_logpdfs + zero_mode_mask*upper_mask*imag_logpdfs)
    
    @jit
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
        prior_logpdf = norm.logpdf(h0, loc = 67.37, scale = 40) + norm.logpdf(ombh2, loc = 0.02233, scale = 0.01) + norm.logpdf(omch2, loc = 0.1198, scale = 0.06)
        return likelihood_logpdf + prior_logpdf

    logdensity = lambda x: logdensity_fn(x[0], x[1], x[2])
    bound_log_prob = lambda x: vmap(logdensity)(x)

    true_h0 = 67.37
    true_omch2 = 0.1198
    true_ombh2 = 0.02233

    init_m = [true_h0, true_ombh2, true_omch2]
    init_lparams = np.load("init_lparams.npy")
    savename = f"gaussian_vi_seed{seed}.npz"
    run_vi_jax(init_m, lr = 1e-5, num_epochs=1000, log_prob_fn=bound_log_prob, init_lparams=init_lparams, savename=savename)

toy_main()
#unlensed_data_main()
