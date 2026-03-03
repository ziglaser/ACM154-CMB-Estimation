import numpy as np

from ensemble_kalman_inversion import EKI, naive_convergence_stopping

def toy_initializer(dim=2):
    theta_mean = np.zeros(dim)
    theta_cov = np.eye(dim)
    return np.random.multivariate_normal(theta_mean, theta_cov)

def toy_forward_model(theta, n_obs, dim=2):
    G_theta = np.eye(dim) @ theta
    return np.tile(G_theta, n_obs) 

def main():
    y_obs = np.load("gaussian_samples.npy")[:100]
    y_flat = y_obs.flatten()

    n_obs = y_obs.shape[0]
    d = y_obs.shape[1]
    k = n_obs * d
    Gamma = np.eye(k)

    forward_model = lambda theta: toy_forward_model(theta, n_obs, d)

    eki = EKI(y=y_flat,
              d=d,
              k=k,
              Gamma=Gamma,
              J=50,
              initializer=toy_initializer,
              forward_model=forward_model)
    u = eki.invert(stopping_algo=naive_convergence_stopping)

    print("Observed means:", y_obs.mean(axis=0))
    print("Estimated parameter:", u)


if __name__ == "__main__":
    main()