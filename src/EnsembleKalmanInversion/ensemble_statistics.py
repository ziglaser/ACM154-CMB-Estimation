import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import scipy.stats as st
import energy_score
import matplotlib.pyplot as plt
import time

from EnsembleKalmanInversion.ensemble_kalman_inversion import EKI

seed = 0
rng = np.random.default_rng()

FIDUCIAL = np.array([67.37, 0.1198, 0.02233])
TOY_MEAN = np.array([0.75, -0.25])
TOY_SIGMA = np.array([[1.0, 0.5],
                       [0.5, 2.0]])

DATA_DIR = os.path.normpath(os.path.join(os.getcwd(), "..", "data"))


def load_EKS_output():
    path = os.path.join(DATA_DIR, "eks_theta_history.npy")
    return np.load(path)


def load_EKI_output():
    data = np.load(os.path.join(DATA_DIR, "eki_ensemble.npz"))
    d = int(data["d"])
    history_len = int(data["history_len"])
    history = np.stack(
        [data[f"h{i}_z"][:, :d] for i in range(history_len)],
        axis=0
    )
    return history


def load_HMC_output():
    path = os.path.join(DATA_DIR, f"unlensed_cmb_hmc_chains_seed{seed}_gaussianprior.npz")
    f     = np.load(path)
    h0    = f["h0_chains"].flatten()
    omch2 = f["omch2_chains"].flatten()
    ombh2 = f["ombh2_chains"].flatten()
    return np.stack([h0, omch2, ombh2], axis=1)


def load_toy_EKI_output():
    y_obs = np.load(os.path.join(DATA_DIR, "gaussian_samples.npy"))
    n_obs, d = y_obs.shape
    y_flat = y_obs.flatten()

    sample_cov = np.cov(y_obs.T)
    Gamma = np.kron(np.eye(n_obs), sample_cov)

    def initializer():
        return np.random.multivariate_normal(np.zeros(d), np.eye(d))

    def forward_model(theta):
        return np.tile(np.eye(d) @ theta, n_obs)

    eki = EKI(y=y_flat, d=d, k=n_obs * d, Gamma=Gamma, J=5000,
              initializer=initializer, forward_model=forward_model)

    k = n_obs * d
    eki.invert(
        stopping_algo=lambda e: e.discrepancy_stopping(tau=np.sqrt(k)),
        max_iter=20,
    )

    history = np.stack(
        [np.array(entry["z"])[:, :d] for entry in eki.history],
        axis=0
    )
    return history


def analytic_toy_posterior(y_obs):
    d = TOY_SIGMA.shape[0]
    y_bar = y_obs.mean(axis=0)
    sigma_inv = np.linalg.inv(TOY_SIGMA)
    posterior_cov = np.linalg.inv(np.eye(d) + sigma_inv)
    posterior_mean = posterior_cov @ sigma_inv @ y_bar
    return posterior_mean, posterior_cov


def _make_ensemble_sampler(ensemble):
    def sampler(sample_size, **_):
        idx = rng.choice(len(ensemble), size=sample_size, replace=True)
        return ensemble[idx], rng
    return sampler


def _make_gaussian_sampler(mean, cov):
    dist = st.multivariate_normal(mean=mean, cov=cov)
    def sampler(sample_size, **_):
        return dist.rvs(size=sample_size, random_state=rng), rng
    return sampler


def compute_MSE(ensemble_history, true_params, scale=None):
    true_params = np.asarray(true_params)
    if scale is None:
        scale = np.abs(true_params)
        scale[scale == 0] = 1.0
    mse = []
    for ensemble in ensemble_history:
        mean = ensemble.mean(axis=0)
        mse.append(np.mean(((mean - true_params) / scale) ** 2))
    return np.array(mse)


def compute_energy_score(ensemble_history, ref_sampler, abs_fn, sample_size=4000):
    es = []
    for ensemble in ensemble_history:
        sample_ens = _make_ensemble_sampler(ensemble)
        es.append(energy_score.energy_square_distance(
            ref_sampler, sample_ens, abs_fn, sample_size))
    return np.array(es)


def _cmb_abs_fn(x):
    return np.linalg.norm(x / FIDUCIAL, axis=-1)


def _toy_abs_fn(x):
    scale = np.abs(TOY_MEAN)
    scale[scale == 0] = 1.0
    return np.linalg.norm(x / scale, axis=-1)


def main():
    print("CMB:")
    hmc = load_HMC_output()
    eki_history = load_EKI_output()
    eks_history = load_EKS_output()
    sample_hmc = _make_ensemble_sampler(hmc)

    mse_eki = compute_MSE(eki_history, FIDUCIAL)
    mse_eks = compute_MSE(eks_history, FIDUCIAL)
    print(" MSE:")
    print("     EKI by iteration:", mse_eki)
    print("     EKS by iteration:", mse_eks)

    es_eki = compute_energy_score(eki_history, sample_hmc, _cmb_abs_fn)
    es_eks = compute_energy_score(eks_history, sample_hmc, _cmb_abs_fn)
    print(" Squared energy score vs HMS")
    print("     EKI by iteration:", es_eki)
    print("     EKS by iteration:", es_eks)

    print("Toy:")
    y_obs_toy = np.load(os.path.join(DATA_DIR, "gaussian_samples.npy"))
    toy_history = load_toy_EKI_output()
    post_m, post_C = analytic_toy_posterior(y_obs_toy)
    print(f"Analytic posterior mean: {post_m}")
    print(f"Analytic posterior cov: {post_C}")

    sample_analytic = _make_gaussian_sampler(post_m, post_C)
    mse_toy = compute_MSE(toy_history, TOY_MEAN)
    print("MSE by iteration:", mse_toy)
    es_toy = compute_energy_score(toy_history, sample_analytic, _toy_abs_fn)
    print("Squared energy score by iteration:", es_toy)


if __name__ == "__main__":
    main()
