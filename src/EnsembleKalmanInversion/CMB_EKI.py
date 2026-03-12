import numpy as np
import os
from time import time

from EnsembleKalmanInversion.ensemble_kalman_inversion import EKI
from generate_cosmopower_unlensed_maps import generate_cosmopower_map, compute_power_spectrum, generate_cosmopower_theory_spectrum

# File paths
HERE       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(HERE, "..", "..", "data")
FIGURE_DIR = os.path.join(HERE, "..", "..", "figures")

# Global 
N_OBS = 1
N_BINS = 50  

def cmb_initializer():
    theta_mean = np.array([67.37, 0.1198, 0.02233])
    theta_cov = np.array([[20.0**2, 0.,          0.      ],
                          [0.,     0.03 ** 2,  0.      ],
                          [0.,     0.,          0.005**2]])
    while True:
        sample = np.random.multivariate_normal(theta_mean, theta_cov)
        if np.all(sample > 0):
            return sample


def cmb_deterministic_forward_model(theta, n_obs):
    cl = generate_cosmopower_theory_spectrum(
        h0=theta[0], omch2=theta[1], ombh2=theta[2],
        noise_level=0.08, n_bins=N_BINS)
    return np.tile(cl, n_obs)


def cmb_forward_model(theta, n_obs):
    cmb_map = generate_cosmopower_map(seed=np.random.randint(0, 1000),
                                      h0=theta[0],
                                      omch2=theta[1],
                                      ombh2=theta[2],
                                      noise_level=0.08)
    cmb_spectrum = compute_power_spectrum(map_2d=cmb_map, 
                                          n_bins=N_BINS)
    
    return np.tile(cmb_spectrum, n_obs)


def analytic_gamma_ps(n_obs, n_bins=N_BINS, fiducial=(67.37, 0.1198, 0.02233), epsilon=1e-6):
    """Analytic diagonal Gamma for the binned power spectrum observable.

    For a zero-mean Gaussian random field, each Fourier mode's squared amplitude
    is exponentially distributed with variance equal to the square of its mean.
    The binned power spectrum estimator Ĉ_i = (1/N_i) Σ_{k∈bin i} |f̃_k|² therefore
    satisfies

        Var[Ĉ_i] = C_i² / N_i

    where C_i is the theoretical (signal + noise) power in bin i returned by
    generate_cosmopower_theory_spectrum, and N_i is the number of rfft2 Fourier
    modes in that bin.  This is the flat-sky analogue of the cosmic-variance formula.
    """
    npix = 64
    pixel_size = 8 * np.pi / (60 * 180)          # radians per pixel
    kx = 2 * np.pi * np.fft.fftfreq(npix, d=pixel_size)
    ky = 2 * np.pi * np.fft.rfftfreq(npix, d=pixel_size)
    ky_grid, kx_grid = np.meshgrid(ky, kx)
    k_grid = np.sqrt(kx_grid**2 + ky_grid**2).flatten()

    bin_edges = np.linspace(0, k_grid.max() * 1.001, n_bins + 1)
    n_modes = np.array([
        np.sum((k_grid >= bin_edges[i]) & (k_grid < bin_edges[i + 1]))
        for i in range(n_bins)
    ], dtype=float)
    n_modes = np.maximum(n_modes, 1)              # guard against empty bins

    h0, omch2, ombh2 = fiducial
    cl_theory = generate_cosmopower_theory_spectrum(
        h0=h0, omch2=omch2, ombh2=ombh2, noise_level=0.08, n_bins=n_bins)

    sigma2 = cl_theory ** 2 / n_modes + epsilon
    return np.diag(np.tile(sigma2, n_obs))


def compute_tau(y, Gamma, fiducial_params, n_obs, stochastic_n=1):
    """Compute the discrepancy stopping threshold τ at the fiducial parameters.

    Mirrors EKI.compute_tau but as a standalone function callable from Julia.
    Returns (mean_tau, std_tau).
    """
    y = np.asarray(y)
    Gamma = np.asarray(Gamma)
    fiducial_params = np.asarray(fiducial_params)
    discrepancies = []
    for _ in range(stochastic_n):
        g = cmb_forward_model(fiducial_params, n_obs)
        residual = g - y
        discrepancy = float(np.sqrt(residual @ np.linalg.solve(Gamma, residual)))
        discrepancies.append(discrepancy)
    return float(np.mean(discrepancies)), float(np.std(discrepancies))


def main():
    data = np.load("../data/cmb_fiducial_dataset.npz")
    y_obs = data["f"][:,:,:N_OBS]

    d = 3
    k = N_OBS * N_BINS

    y_obs_ps = np.stack([compute_power_spectrum(y_obs[:,:,i], n_bins=N_BINS)
                         for i in range(N_OBS)])
    y_flat = y_obs_ps.ravel()
    Gamma = analytic_gamma_ps(N_OBS)

    forward_model = lambda theta: cmb_forward_model(theta, N_OBS)
    theory_model  = lambda theta: cmb_deterministic_forward_model(theta, N_OBS)

    eki = EKI.load("../data/eki_ensemble.npz", initializer=cmb_initializer, forward_model=forward_model)
    param_bounds = np.array([[67.37 - 40, 67.37 + 40],
                              [0.001,  0.24],
                              [0.001, 0.04466]])
    # eki = EKI(y=y_flat,
    #           d=d,
    #           k=k,
    #           Gamma=Gamma,
    #           J=500,
    #           initializer=cmb_initializer,
    #           forward_model=forward_model,
    #           verbose=False,
    #           param_bounds=param_bounds)
    # tau_mean, tau_std = eki.compute_tau(fiducial_params=[67.37, 0.1198, 0.02233], 
    #                                     stochastic_n=500)
    # u = eki.invert(
    #     stopping_algo=lambda e: e.discrepancy_stopping(tau=tau_mean),
    #     max_iter=10,
    #     timed=True)
    
    # eki.save(os.path.join(DATA_DIR, "eki_ensemble.npz"))

    print("Estimated parameter:", eki.u)

    eki.animate_ensemble(
        param_names=[r"$H_0$", r"$\Omega_{b}h^2$", r"$\Omega_{c}h^2$"],
        true_params=[67.37, 0.02233, 0.1198],
        param_xlims=[(67.37 - 40, 67.37 + 40),
                     (0.001, 0.04466),
                     (0.001, 0.24)],
        col_order=[0, 2, 1],
        save_path=os.path.join(FIGURE_DIR, "zach_eki.gif"),
        y_obs_ps=y_obs_ps,
        frames_dir=os.path.join(FIGURE_DIR, "zach_eki")
    )


if __name__ == "__main__":
    main()