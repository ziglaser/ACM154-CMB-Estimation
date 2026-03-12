import os
import numpy as np

from EnsembleKalmanInversion.ensemble_kalman_inversion import EKI, naive_convergence_stopping

HERE       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(HERE, "..", "..", "data")
FIGURE_DIR = os.path.join(HERE, "..", "..", "figures")

TRUE_PARAMS = [0.75, -0.25]

def toy_initializer(dim=2):
    theta_mean = np.zeros(dim)
    theta_cov = np.eye(dim)
    return np.random.multivariate_normal(theta_mean, theta_cov)

def toy_forward_model(theta, n_obs, dim=2):
    G_theta = np.eye(dim) @ theta
    return np.tile(G_theta, n_obs)

def main():
    y_obs = np.load(os.path.join(DATA_DIR, "gaussian_samples.npy"))
    y_flat = y_obs.flatten()

    n_obs = y_obs.shape[0]
    d = y_obs.shape[1]
    k = n_obs * d
    sample_cov = np.cov(y_obs.T)                  # (d, d) empirical covariance
    Gamma = np.kron(np.eye(n_obs), sample_cov)    # block-diagonal: each obs has cov = sample_cov

    forward_model = lambda theta: toy_forward_model(theta, n_obs, d)

    eki = EKI(y=y_flat,
              d=d,
              k=k,
              Gamma=Gamma,
              J=1000,
              initializer=toy_initializer,
              forward_model=forward_model)
    u = eki.invert(stopping_algo=lambda e: e.discrepancy_stopping(tau=1.0 * np.sqrt(k)))

    print("Observed means:", y_obs.mean(axis=0))
    print("Estimated parameter:", u)

    os.makedirs(FIGURE_DIR, exist_ok=True)
    eki.animate_ensemble(
        param_names=[r"$\theta_0$", r"$\theta_1$"],
        true_params=TRUE_PARAMS,
        param_xlims=[(-2.5, 2.5), (-2.5,2.5)],
        save_path=os.path.join(FIGURE_DIR, "toy_eki.gif"),
    )


if __name__ == "__main__":
    main()
