import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax.scipy.stats as stats
import scipy.fft as fft
from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ
from interpax import interp1d
import scipy.stats as st

'''For reference, these are reasonable cosmological parameters to take as the ground truth. They are inferred
from the fourth data release + analysis of the Planck satellite.
h0 = 67.37
omch2 = 0.1198
ombh2 = .02233
'''
def make_upper_mask(npix, self_inverse_indices): 
    '''Makes a mask that is 1 for upper indices (ie, free complex modes) and 0 otherwise. 
    
    Arguments:
    npix (int): number of pixels per side of the real space map.
    self_inverse_indices (array or list of ints): indices which are required to have real valued fourier modes
    
    Returns:
    mask (np.array): a mask to be applied in the rfft space that is 1 for free complex modes, 0 otherwise.
    '''
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

def get_likelihood_logpdfs(sigmas, unlensed_cmb_fouriers, zero_mode_mask, self_inverse_mask, upper_mask):
    '''Returns the likelihood logpdf for a Gaussian random field.

    Arguments:
    sigmas (jnp.array of float): the sigmas defined by the power spectrum. (They scale the magnitude of the fourier variables.)
    unlensed_cmb_fouriers (jnp.array of float): the (rfft2 convention) fourier transform of the map
    zero_mode_mask (jnp.array of int): a mask in fourier space that is zero for the constant mode, 1 elsewhere
    self_inverse_mask (jnp.array of int): a mask in fourier space that is 1 for self inverse indices (ie, restricted to have real fourier coefficients) and 0 elsewhere
    upper_mask (jnp.array of int): a mask in fourier space that is 1 for upper indices (ie, free complex modes) and 0 elsewhere

    Returns:
    logpdf (float): the log likelihood of the map given the sigmas
    '''
    real_logpdfs = jnp.nan_to_num(stats.norm.logpdf(unlensed_cmb_fouriers.real, loc = 0, scale = sigmas/jnp.sqrt(2)))
    imag_logpdfs = jnp.nan_to_num(stats.norm.logpdf(unlensed_cmb_fouriers.imag, loc = 0, scale = sigmas/jnp.sqrt(2)))
    self_inverse_logpdfs = jnp.nan_to_num(stats.norm.logpdf(unlensed_cmb_fouriers.real, loc = 0, scale = sigmas))
    return jnp.sum(zero_mode_mask*self_inverse_mask*self_inverse_logpdfs + zero_mode_mask*upper_mask*real_logpdfs + zero_mode_mask*upper_mask*imag_logpdfs)

def logdensity_fn(h0, ombh2, omch2, fourier_k, unlensed_cmb_fouriers, zero_mode_mask, self_inverse_mask, upper_mask, noise_variances):
    '''Compute the posterior logdensity. Note that the prior must be manually set in this function in the current implementation.

    Arguments:
    h0 (float): the hubble constant h0 in units of km s^{-1} Mpc{^-1}
    ombh2 (float): the critical density of baryonic matter multiplied by (h0/100)**2
    omch2 (float): the critical density of cold dark matter multiplied by (h0/100)**2
    fourier_k (jnp.array of floats): the (rfft2 convention) grid of fourier wave numbers
    unlensed_cmb_fouriers (jnp.array of float): the (rfft2 convention) fourier transform of the map
    zero_mode_mask (jnp.array of int): a mask in fourier space that is zero for the constant mode, 1 elsewhere
    self_inverse_mask (jnp.array of int): a mask in fourier space that is 1 for self inverse indices (ie, restricted to have real fourier coefficients) and 0 elsewhere
    upper_mask (jnp.array of int): a mask in fourier space that is 1 for upper indices (ie, free complex modes) and 0 elsewhere
    noise_variances (jnp.array of float): a map in (rfft2 convention) fourier space that has the variances of the noise.

    Returns:
    logpdf (float): the log of the posterior of the map.
    '''
    cosmo_params = jnp.array([ombh2, omch2, h0/100, .0540, .9652, np.log(10*2.08666022)])
    emulator = CPJ(probe='cmb_tt')
    spectrum = emulator.predict(cosmo_params)
    unitful_factor = 7428350250000.0 #convert from dimensionless emulator spectrum to microK spectrum
    spectrum = unitful_factor*spectrum
    emulator_ell = emulator.modes
    map_variances = interp1d(fourier_k.flatten(), emulator_ell, spectrum, method="cubic")
    map_variances = jnp.reshape(map_variances, (64, 33)) 
    sigmas = jnp.sqrt(map_variances + noise_variances)
    likelihood_logpdf = get_likelihood_logpdfs(sigmas) #compute likelihood pdfs
    prior_logpdf = 0 #compute the prior pdfs
    return likelihood_logpdf + prior_logpdf

def generate_cosmopower_map(seed, noise_level = 0.08, h0 = 67.37, omch2 = 0.1198, ombh2 = 0.02233, save = True, show = False):
    '''Generates an unlensed CMB map given cosmological parameters of interest. Note that the generated map will be a
    Gaussian random field, but the power spectrum of this field will correspond to the LENSED temperature auto spectrum.
    For our project, this doesn't matter. It also saves the map in the file f"hmc_unlensed_map_seed{seed}.npy" if save = True.
    It will also save a plot of the emulator spectrum in the file "emulator_spectrum_fiducial.png" if save = True.

    Arguments:
    seed (int): random seed to pass to the generator
    noise_level (float): standard deviation of observation white noise (unitful in fourier space). At 0.08, the fiducial Planck spectrum is noise dominated around ell ~2000.
    h0 (float): the hubble constant h0 in units of km s^{-1} Mpc{^-1}
    ombh2 (float): the critical density of baryonic matter multiplied by (h0/100)**2
    omch2 (float): the critical density of cold dark matter multiplied by (h0/100)**2
    save (boolean): if True, the map will also be saved as an npy file called f"hmc_unlensed_map_seed{seed}.npy".
    show (boolean): if True, the map will be displayed as a pop up
    
    Returns:
    map (array of floats): the real valued pixel space realization of our random field
    '''
    npix = 64 #number of pixels per side in a square (pixel space) map
    rng = np.random.default_rng(seed)
    pixel_size = 8 #pixel size IN ARCMIN (ie, arcmin per pixel)
    radian_per_arcmin = np.pi/(60*180) #conversion factor
    pixel_size = pixel_size * radian_per_arcmin #pixel size IN RADIANS (ie, radians per pixel)
    kx = 2 * np.pi * fft.fftfreq(npix, d=pixel_size) #d is the sample spacing in radians
    ky = 2 * np.pi * fft.rfftfreq(npix, d=pixel_size)
    ky, kx = np.meshgrid(ky, kx)
    k = np.sqrt(kx**2+ky**2)
    fourier_k = k #compute the appropriate grid of fourier k
    self_inverse_indices = [0, npix//2] #these are the indices corresponding to fourier variables that are necessarily real

    #get necessary masks
    upper_mask = make_upper_mask(npix, self_inverse_indices) #1 for "upper" indices (ie, complex fourier modes that are random variables), 0 otherwise
    zero_mode_mask = np.ones_like(upper_mask) 
    zero_mode_mask[0,0] = 0 #0 for the constant (aka DC) mode, 1 elsewhere
    self_inverse_mask = np.zeros_like(k) #1 for the indices corresponding to real fourier variables, 0 elsewhere
    for i in self_inverse_indices:
        for j in self_inverse_indices:
            self_inverse_mask[i,j] = 1

    #construct the cosmopower emulator to get the lensed temperature power spectrum for the desired cosmological parameters.
    cosmo_params = jnp.array([ombh2, omch2, h0/100, .0540, .9652, np.log(10*2.08666022)]) #the other numbers are cosmological parameters required by cosmopower that we have fixed
    emulator = CPJ(probe='cmb_tt')
    spectrum = emulator.predict(cosmo_params)
    emulator_ell = emulator.modes
    unitful_factor = 7428350250000.0 #this converts from a unitless map (in CAMB convention) to a map in Kelvin
    spectrum = spectrum*unitful_factor

    #convert the cosmpower emulator spectrum to variances of fourier modes
    sigmas = interp1d(fourier_k.flatten(), emulator_ell, spectrum, method="cubic")
    sigmas = np.reshape(sigmas, (64, 33))
    sigmas = np.sqrt(sigmas)
    sigmas[0,0] = 0
    
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
    noiseless_map = fft.irfft2(fourier_map_coeffs, s = (npix, npix), norm = "ortho")
    map = map/pixel_size #to get to the units of the julia map, we need map/pixel size. The likelihood code takes maps with these units and multiplies by pixel size to compute logpdfs, ignoring the fixed normalization constant from this jacobian.

    #save the map 
    if save:
        #save the map
        np.save(f"hmc_unlensed_map_seed{seed}.npy", map)
        print("map saved")

    #plot the map and save the figures
    if show:
        plt.imshow(noiseless_map)
        plt.colorbar()
        plt.title(f"Noise free Unlensed CMB Temperature Map Seed {seed}")
        plt.savefig("noise_free_map.png")
        plt.show()

        plt.imshow(map)
        plt.colorbar()
        plt.title(f"Unlensed CMB Temperature Map Seed {seed}")
        plt.savefig("map_with_noise.png")
        plt.show()

        plt.imshow(map/pixel_size)
        plt.colorbar()
        plt.title(f"Unlensed CMB Temperature Map Seed {seed} with pixel size correction")
        plt.show()

    return map


#example function call to generate a map, save it, and plot it.
generate_cosmopower_map(0, 0.08, 67.37,0.1198, 0.02233, True, True)