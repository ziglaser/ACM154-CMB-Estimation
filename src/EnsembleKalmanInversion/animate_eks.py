import os
import types
import numpy as np

from EnsembleKalmanInversion.ensemble_kalman_inversion import EKI
from generate_cosmopower_unlensed_maps import (
    generate_cosmopower_theory_spectrum,
    compute_power_spectrum,
)

HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "..", "data")

N_OBS  = 100
N_BINS = 50


def forward_model(theta):
    h0    = np.clip(theta[0], 67.37 - 40, 67.37 + 40)
    omch2 = np.clip(theta[1], 0.001,  0.24)
    ombh2 = np.clip(theta[2], 0.001, 0.04466)
    cl = generate_cosmopower_theory_spectrum(
        h0=h0, omch2=omch2, ombh2=ombh2,
        noise_level=0.08, n_bins=N_BINS)
    return np.tile(np.array(cl), N_OBS)


def main():
    theta_history = np.load(os.path.join(DATA_DIR, "eks_theta_history.npy"))
    n_snaps, N_ens, N_theta = theta_history.shape

    d = N_theta
    k = N_OBS * N_BINS

    data    = np.load(os.path.join(DATA_DIR, "cmb_fiducial_dataset.npz"))
    f       = data["f"][:, :, :N_OBS]
    y_obs_ps = np.stack([compute_power_spectrum(f[:, :, i], n_bins=N_BINS)
                         for i in range(N_OBS)])  
    history = []
    for i in range(n_snaps):
        theta_i = theta_history[i]
        G_i = np.stack([forward_model(theta_i[j]) for j in range(N_ens)])
        z_i = np.concatenate([theta_i, G_i], axis=1)
        history.append({
            "z":     z_i,
            "u":     theta_i.mean(axis=0),
            "z_hat": None,
            "C":     None,
            "K":     None,
        })

    eks = types.SimpleNamespace(history=history, d=d, k=k)
    EKI.animate_ensemble(
        eks,
        param_names=[r"$H_0$", r"$\Omega_{b}h^2$", r"$\Omega_{c}h^2$"],
        true_params=[67.37, 0.02233, 0.1198],
        param_xlims=[(67.37 - 40, 67.37 + 40), (0, 0.04466), (0, 0.24)],
        col_order=[0, 2, 1],
        save_path="../figures/bohan_eki.gif",
        y_obs_ps=y_obs_ps,
        frames_dir="../figures/bohan_eki"
    )


if __name__ == "__main__":
    main()
