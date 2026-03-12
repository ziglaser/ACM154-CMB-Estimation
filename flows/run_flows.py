import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
import scipy.fft as fft
import matplotlib.pyplot as plt
from flows_triangle_plotter import triangle_plot

from normalizing_flows import (
    CosmoPosterior,
    run_sampling_test,
    run_transport_vi_for_cosmo,
)


# ─────────────────────────────────────────────────────────────────────────────
# Choose dataset: "toy" | "unlensed" | "lensed"
# ─────────────────────────────────────────────────────────────────────────────
DATASET = "toy"

# ─────────────────────────────────────────────────────────────────────────────
# Shared hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
LR         = 1e-3
NUM_EPOCHS = 2000
BATCH_SIZE = 64
NUM_LAYERS = 8
HIDDEN_DIM = 256
SHOW_ANIM  = False

# Likelihood temperature annealing.
# Beta is ramped linearly from BETA_START → BETA_END over NUM_EPOCHS.
# The log-likelihood is normalised by N_eff (≈2000 Fourier modes) so it is O(1),
# matching the scale of log_prior.  This means:
#   beta = 0   → pure prior
#   beta = 1   → prior and likelihood weighted equally
#   beta > 1   → likelihood dominates (data-driven, appropriate once surrogate is accurate)
# Start conservative (0→1) and increase BETA_END once results look stable.
BETA_START = 0.0
BETA_END   = 200.0


# ─────────────────────────────────────────────────────────────────────────────
# Shared posterior triangle-plot helper
# ─────────────────────────────────────────────────────────────────────────────

# Fiducial (true) cosmological parameters used for data generation
TRUE_VALUES = [67.37, 0.02233, 0.1198]   # h0, ombh2, omch2
PARAM_NAMES = ["$H_0$", r"$\Omega_b h^2$", r"$\Omega_c h^2$"]

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def make_snapshot_callback(phys_mean, phys_std, label, n_samples=1000):
    """
    Returns a callback(model, epoch) that:
    - Draws n_samples from the flow and un-normalises to physical units
    - Computes the posterior mean
    - Saves a triangle plot to plots/<label>_epoch<epoch>.png
    """
    def callback(model, epoch):
        model.eval()
        with torch.no_grad():
            samples_norm = model.sample(n_samples)
        samples_phys = (samples_norm * phys_std + phys_mean).numpy()  # (N, 3)
        model.train()

        posterior_mean = samples_phys.mean(axis=0).tolist()

        # triangle_plot expects (n_chains, n_params, n_samples)
        samples_tp = samples_phys.T[np.newaxis, :, :]   # (1, 3, N)

        fig_name = os.path.join(PLOT_DIR, f"{label}_epoch{epoch:04d}.png")
        fig, _ = triangle_plot(
            samples_tp,
            param_names=PARAM_NAMES,
            bins1d=40,
            bins2d=30,
            color="steelblue",
            hist2d_cmap="Blues",
            true_values=TRUE_VALUES,
            mean_values=posterior_mean,
            fig_name=fig_name,
            dpi=150,
        )
        fig.suptitle(f"{label}  —  epoch {epoch}", fontsize=13, y=1.01)
        plt.savefig(fig_name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fig_name}")

    return callback


# ─────────────────────────────────────────────────────────────────────────────
# CosmoPower surrogate definition (must match train_surrogate.py)
# ─────────────────────────────────────────────────────────────────────────────
class CosmoPowerSurrogate(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3,          512), nn.LeakyReLU(),
            nn.Linear(512,        512), nn.LeakyReLU(),
            nn.Linear(512,        512), nn.LeakyReLU(),
            nn.Linear(512,        256), nn.LeakyReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_surrogate(path="cosmopower_surrogate.pt"):
    ckpt       = torch.load(path, map_location="cpu", weights_only=False)
    surrogate  = CosmoPowerSurrogate(output_dim=ckpt["output_dim"])
    surrogate.load_state_dict(ckpt["model_state"])
    surrogate.eval()
    param_mean = torch.tensor(ckpt["param_mean"], dtype=torch.float32)
    param_std  = torch.tensor(ckpt["param_std"],  dtype=torch.float32)
    return surrogate, param_mean, param_std


# =============================================================================
# TOY GAUSSIAN
# =============================================================================
if DATASET == "toy":
    # ── data ──────────────────────────────────────────────────────────────────
    samples  = np.load("../data/gaussian_samples.npy")
    observed = torch.tensor(samples[0], dtype=torch.float32)   # one 2-D observation

    # Likelihood: log p(observed | mean) = log N(observed; mean, cov)
    # Posterior parameters are theta = [mean1, mean2].
    # Prior: standard normal on each mean  →  log p(theta) = sum_i log N(theta_i; 0, 1)
    data_cov = torch.tensor([[1.0, 0.5], [0.5, 2.0]])

    def log_likelihood_fn_toy(theta):
        """theta: [N, 2].  Returns log p(observed | theta) of shape [N]."""
        # log N(observed; theta_i, data_cov)  for each row theta_i
        diff = observed.unsqueeze(0) - theta        # [N, 2]
        lik_dist = dist.MultivariateNormal(
            torch.zeros(2), covariance_matrix=data_cov
        )
        return lik_dist.log_prob(diff)              # [N]

    prior_mean = torch.zeros(2)
    prior_cov  = torch.eye(2)

    pi  = CosmoPosterior(prior_mean, prior_cov, log_likelihood_fn_toy)
    pi.dim = 2  # already set by CosmoPosterior.__init__

    _t0 = time.time()
    flow = run_transport_vi_for_cosmo(
        pi,
        lr=LR,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        num_layers=NUM_LAYERS,
        hidden_dim=HIDDEN_DIM,
        show_anim=SHOW_ANIM,
    )
    _train_time = time.time() - _t0
    print(f"\nToy training time: {_train_time:.1f}s  ({_train_time/60:.2f} min)")

    # ── Analytical posterior (Gaussian conjugate) ────────────────────────────
    # Likelihood: x | θ ~ N(θ, Σ_data),  Prior: θ ~ N(0, I)
    # → Σ_post = inv(inv(Σ_data) + I),   μ_post = Σ_post @ inv(Σ_data) @ x
    _data_cov_np   = np.array([[1.0, 0.5], [0.5, 2.0]])
    _data_cov_inv  = np.linalg.inv(_data_cov_np)
    _post_cov_true = np.linalg.inv(_data_cov_inv + np.eye(2))
    _obs_np        = observed.numpy()
    _post_mean_true = _post_cov_true @ _data_cov_inv @ _obs_np   # (2,)

    # Flow samples
    flow.eval()
    with torch.no_grad():
        _flow_samples = flow.sample(2000).numpy()   # (2000, 2)
    _flow_mean = _flow_samples.mean(axis=0)

    # MSE of posterior mean vs analytical posterior mean
    _mse_toy = np.mean((_flow_mean - _post_mean_true) ** 2)
    print(f"\n── Toy summary ───────────────────────────────────")
    print(f"  Analytical posterior mean : {_post_mean_true}")
    print(f"  Flow posterior mean       : {_flow_mean}")
    print(f"  MSE (mean vs true mean)   : {_mse_toy:.6f}")

    # Energy score: flow posterior vs analytical posterior
    from energy_score import energy_square_distance as _es
    _rng = np.random.default_rng(0)

    def _sample_true(n, key=None):
        s = _rng.multivariate_normal(_post_mean_true, _post_cov_true, size=n)
        return s, None

    flow.eval()  # ensure eval mode for sampling
    def _sample_flow_fresh(n, key=None):
        with torch.no_grad():
            s = flow.sample(n).numpy()
        return s, None

    _es_val = _es(_sample_true, _sample_flow_fresh, lambda d: np.sqrt((d ** 2).sum(-1)), 1000)
    print(f"  Squared energy score (Flow || True posterior): {_es_val:.6f}")
    print(f"  Training time             : {_train_time:.1f}s")

    # Visualise: target space vs latent space, true posterior contour in orange
    run_sampling_test(
        flow,
        pi=pi,
        n_samples=400,
        u1_range=[-2, 3],
        u2_range=[-2, 2],
        fig_name=os.path.join(PLOT_DIR, "toy_final.png"),
    )


# =============================================================================
# UNLENSED CMB  (single 64×64 map; parameters: h0, ombh2, omch2)
# =============================================================================
elif DATASET == "unlensed":
    # ── data ──────────────────────────────────────────────────────────────────
    unlensed_map_np = np.load("../data/hmc_unlensed_map_seed0.npy").astype(np.float32)

    npix           = 64
    pixel_size_am  = 8.0                          # arcmin / pixel
    pixel_size_rad = pixel_size_am * np.pi / (60 * 180)

    # Pre-compute the Fourier transform of the observed map (fixed, not differentiated)
    unlensed_map_scaled = unlensed_map_np * pixel_size_rad   # undo the 1/pix scaling from generation
    fourier_obs = fft.rfft2(unlensed_map_scaled, norm="ortho")

    kx = 2 * np.pi * fft.fftfreq(npix, d=pixel_size_rad)
    ky = 2 * np.pi * fft.rfftfreq(npix, d=pixel_size_rad)
    ky_grid, kx_grid = np.meshgrid(ky, kx)
    fourier_k  = np.sqrt(kx_grid ** 2 + ky_grid ** 2)        # (64, 33)

    noise_level    = 0.08   # matches hmc_flexible_serial_production_version.py
    noise_variances = np.ones_like(fourier_k) * noise_level ** 2

    # Masks (same as hmc_flexible_serial_production_version.py)
    self_inverse_indices = [0, npix // 2]
    self_inverse_mask    = np.zeros_like(fourier_k)
    for i in self_inverse_indices:
        for j in self_inverse_indices:
            self_inverse_mask[i, j] = 1

    upper_mask = np.ones((npix, npix // 2 + 1))
    for i in self_inverse_indices:
        for j in self_inverse_indices:
            upper_mask[i, j] = 0
    upper_mask[npix // 2 + 1:, 0]  = 0
    upper_mask[npix // 2 + 1:, -1] = 0

    zero_mode_mask           = np.ones_like(upper_mask)
    zero_mode_mask[0, 0]     = 0

    # ── load surrogate (run train_surrogate.py first) ─────────────────────────
    surrogate, param_mean_t, param_std_t = load_surrogate("cosmopower_surrogate.pt")

    # Pre-convert fixed numpy arrays to torch tensors once
    noise_var_t         = torch.tensor(noise_variances,    dtype=torch.float32)  # (64, 33)
    fourier_real_t      = torch.tensor(fourier_obs.real,   dtype=torch.float32)  # (64, 33)
    fourier_imag_t      = torch.tensor(fourier_obs.imag,   dtype=torch.float32)  # (64, 33)
    self_inverse_mask_t = torch.tensor(self_inverse_mask,  dtype=torch.float32)  # (64, 33)
    upper_mask_t        = torch.tensor(upper_mask,         dtype=torch.float32)  # (64, 33)
    zero_mode_mask_t    = torch.tensor(zero_mode_mask,     dtype=torch.float32)  # (64, 33)

    # Number of independent scalar terms summed in log_lik.
    # Self-inverse modes contribute 1 real term each; upper-triangle modes
    # contribute 2 terms (real + imag).  Dividing by N_eff makes log_lik O(1),
    # the same scale as log_prior (a 3D Gaussian ≈ -5 to -10), so that beta
    # annealing actually controls the prior/likelihood balance.
    # Without this, log_lik ≈ -10,000 and even beta=0.01 overwhelms the prior.
    _zm = zero_mode_mask_t
    _si = self_inverse_mask_t
    _um = upper_mask_t
    N_EFF_UNLENSED = float((_zm * _si).sum() + 2.0 * (_zm * _um).sum())

    def log_likelihood_fn_unlensed(theta):
        """
        theta: [N, 3] normalised parameter tensor (z-scored by param_mean_t / param_std_t).
        Returns normalised log p(map | theta) / N_eff of shape [N].

        Dividing by N_eff (≈2000 independent Fourier terms) brings the
        log-likelihood to O(1), matching the scale of the 3D log-prior so that
        beta annealing actually controls the prior/likelihood balance.
        """
        # 1. theta is already normalised → feed directly to surrogate
        log_map_var = surrogate(theta)                              # [N, 64*33]
        map_var     = torch.exp(log_map_var).reshape(-1, 64, 33)   # [N, 64, 33]

        # 2. sigmas = sqrt(signal_variance + noise_variance)
        sigmas = torch.sqrt(map_var + noise_var_t.unsqueeze(0))    # [N, 64, 33]

        # 3. Log-pdf of observed Fourier coefficients
        sqrt2 = 2 ** 0.5
        self_inv_lp = dist.Normal(0., sigmas).log_prob(
            fourier_real_t.unsqueeze(0))                           # [N, 64, 33]
        real_lp = dist.Normal(0., sigmas / sqrt2).log_prob(
            fourier_real_t.unsqueeze(0))
        imag_lp = dist.Normal(0., sigmas / sqrt2).log_prob(
            fourier_imag_t.unsqueeze(0))

        zm = zero_mode_mask_t.unsqueeze(0)
        si = self_inverse_mask_t.unsqueeze(0)
        um = upper_mask_t.unsqueeze(0)

        log_lik = (zm * si * self_inv_lp
                   + zm * um * real_lp
                   + zm * um * imag_lp).sum(dim=(1, 2))            # [N]

        return log_lik / N_EFF_UNLENSED

    # Prior in NORMALISED space → N(0, I_3).
    # The flow reference is also N(0, I_3), so they are aligned from step 1.
    # The surrogate was trained on z-scored inputs, so theta ~ N(0,I) is within
    # its training distribution throughout training.
    prior_mean = torch.zeros(3)
    prior_cov  = torch.eye(3)

    phys_mean_t = param_mean_t
    phys_std_t  = param_std_t

    snapshot_epochs = [
        0,
        NUM_EPOCHS // 4,
        NUM_EPOCHS // 2,
        3 * NUM_EPOCHS // 4,
        NUM_EPOCHS - 1,
    ]
    callback = make_snapshot_callback(phys_mean_t, phys_std_t, label="unlensed")

    pi = CosmoPosterior(prior_mean, prior_cov, log_likelihood_fn_unlensed)

    _t0 = time.time()
    flow = run_transport_vi_for_cosmo(
        pi,
        lr=LR,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        num_layers=NUM_LAYERS,
        hidden_dim=HIDDEN_DIM,
        show_anim=SHOW_ANIM,
        snapshot_epochs=snapshot_epochs,
        snapshot_callback=callback,
        beta_start=BETA_START,
        beta_end=BETA_END,
    )
    _train_time = time.time() - _t0
    print(f"\nUnlensed training time: {_train_time:.1f}s  ({_train_time/60:.2f} min)")

    # Save 5000 un-normalised flow samples for energy score evaluation
    flow.eval()
    with torch.no_grad():
        _samples_norm = flow.sample(5000)
    _samples_phys = (_samples_norm * phys_std_t + phys_mean_t).numpy()  # (5000, 3)
    np.save("unlensed_flow_samples.npy", _samples_phys)
    print(f"Saved unlensed_flow_samples.npy  shape={_samples_phys.shape}")

    # MSE of posterior mean vs true values
    _true = np.array(TRUE_VALUES)                        # [h0, ombh2, omch2]
    _fiducial = np.array([67.37, 0.02233, 0.1198])
    _post_mean = _samples_phys.mean(axis=0)              # (3,)
    _mse_raw   = np.mean((_post_mean - _true) ** 2)
    _mse_scaled = np.mean(((_post_mean - _true) / _fiducial) ** 2)
    print(f"\n── Unlensed summary ──────────────────────────────")
    print(f"  True values  : h0={_true[0]:.4f}  ombh2={_true[1]:.5f}  omch2={_true[2]:.4f}")
    print(f"  Posterior mean: h0={_post_mean[0]:.4f}  ombh2={_post_mean[1]:.5f}  omch2={_post_mean[2]:.4f}")
    print(f"  MSE (raw)    : {_mse_raw:.6f}")
    print(f"  MSE (scaled) : {_mse_scaled:.6f}  (normalised by fiducial²)")
    print(f"  Training time: {_train_time:.1f}s")


# =============================================================================
# LENSED CMB  (10 maps from map_generation.jl; same parameter space)
# =============================================================================
elif DATASET == "lensed":
    # f_tilde shape from Julia: (64, 64, nsims) after cat(..., dims=3)
    lensed_npz  = np.load("../data/all_CMB_simulations.npz")
    f_tilde     = lensed_npz["f_tilde"].astype(np.float32)  # (64, 64, nsims)
    # Use the first map as the "observation" for posterior inference
    observed_map_np = f_tilde[:, :, 0]

    npix           = 64
    pixel_size_am  = 8.0
    pixel_size_rad = pixel_size_am * np.pi / (60 * 180)

    # Fourier transform of the observed lensed map (same convention as unlensed)
    lensed_map_scaled = observed_map_np * pixel_size_rad
    fourier_obs_l = fft.rfft2(lensed_map_scaled, norm="ortho")

    kx = 2 * np.pi * fft.fftfreq(npix, d=pixel_size_rad)
    ky = 2 * np.pi * fft.rfftfreq(npix, d=pixel_size_rad)
    ky_grid, kx_grid = np.meshgrid(ky, kx)
    fourier_k_l = np.sqrt(kx_grid ** 2 + ky_grid ** 2)       # (64, 33)

    noise_level_l      = 0.08   # matches hmc_flexible_serial_production_version.py
    noise_variances_l  = np.ones_like(fourier_k_l) * noise_level_l ** 2

    # Mode masks (identical logic to unlensed)
    self_inverse_indices = [0, npix // 2]
    self_inverse_mask_l  = np.zeros_like(fourier_k_l)
    for i in self_inverse_indices:
        for j in self_inverse_indices:
            self_inverse_mask_l[i, j] = 1

    upper_mask_l = np.ones((npix, npix // 2 + 1))
    for i in self_inverse_indices:
        for j in self_inverse_indices:
            upper_mask_l[i, j] = 0
    upper_mask_l[npix // 2 + 1:, 0]  = 0
    upper_mask_l[npix // 2 + 1:, -1] = 0

    zero_mode_mask_l        = np.ones_like(upper_mask_l)
    zero_mode_mask_l[0, 0]  = 0

    # Load surrogate (trained on unlensed C_ell — an approximation for lensed)
    surrogate_l, param_mean_tl, param_std_tl = load_surrogate("cosmopower_surrogate.pt")

    noise_var_tl         = torch.tensor(noise_variances_l,   dtype=torch.float32)
    fourier_real_tl      = torch.tensor(fourier_obs_l.real,  dtype=torch.float32)
    fourier_imag_tl      = torch.tensor(fourier_obs_l.imag,  dtype=torch.float32)
    self_inverse_mask_tl = torch.tensor(self_inverse_mask_l, dtype=torch.float32)
    upper_mask_tl        = torch.tensor(upper_mask_l,        dtype=torch.float32)
    zero_mode_mask_tl    = torch.tensor(zero_mode_mask_l,    dtype=torch.float32)

    N_EFF_LENSED = float(
        (zero_mode_mask_tl * self_inverse_mask_tl).sum()
        + 2.0 * (zero_mode_mask_tl * upper_mask_tl).sum()
    )

    def log_likelihood_fn_lensed(theta):
        """
        theta: [N, 3] normalised parameter tensor (z-scored).
        Returns normalised log p(lensed_map | theta) / N_eff of shape [N].
        Uses the unlensed surrogate as an approximation; divides by N_eff so
        the likelihood is O(1) and beta annealing controls prior/likelihood balance.
        """
        log_map_var = surrogate_l(theta)
        map_var     = torch.exp(log_map_var).reshape(-1, 64, 33)
        sigmas      = torch.sqrt(map_var + noise_var_tl.unsqueeze(0))
        sqrt2       = 2 ** 0.5
        self_inv_lp = dist.Normal(0., sigmas).log_prob(fourier_real_tl.unsqueeze(0))
        real_lp     = dist.Normal(0., sigmas / sqrt2).log_prob(fourier_real_tl.unsqueeze(0))
        imag_lp     = dist.Normal(0., sigmas / sqrt2).log_prob(fourier_imag_tl.unsqueeze(0))
        zm = zero_mode_mask_tl.unsqueeze(0)
        si = self_inverse_mask_tl.unsqueeze(0)
        um = upper_mask_tl.unsqueeze(0)
        log_lik = (zm * si * self_inv_lp
                   + zm * um * real_lp
                   + zm * um * imag_lp).sum(dim=(1, 2))
        return log_lik / N_EFF_LENSED

    # Prior in normalised space (same reasoning as unlensed)
    prior_mean = torch.zeros(3)
    prior_cov  = torch.eye(3)

    # Physical-space scales from the surrogate checkpoint (HMC prior widths)
    phys_mean = param_mean_tl
    phys_std  = param_std_tl

    snapshot_epochs = [
        0,
        NUM_EPOCHS // 4,
        NUM_EPOCHS // 2,
        3 * NUM_EPOCHS // 4,
        NUM_EPOCHS - 1,
    ]
    callback = make_snapshot_callback(phys_mean, phys_std, label="lensed", n_samples=1000)

    pi = CosmoPosterior(prior_mean, prior_cov, log_likelihood_fn_lensed)

    _t0 = time.time()
    flow = run_transport_vi_for_cosmo(
        pi,
        lr=LR,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        num_layers=NUM_LAYERS,
        hidden_dim=HIDDEN_DIM,
        show_anim=SHOW_ANIM,
        snapshot_epochs=snapshot_epochs,
        snapshot_callback=callback,
        beta_start=BETA_START,
        beta_end=BETA_END,
    )
    _train_time = time.time() - _t0
    print(f"\nLensed training time: {_train_time:.1f}s  ({_train_time/60:.2f} min)")

    # MSE of posterior mean vs true values
    flow.eval()
    with torch.no_grad():
        _samples_norm = flow.sample(5000)
    _samples_phys = (_samples_norm * phys_std + phys_mean).numpy()
    _true     = np.array(TRUE_VALUES)
    _fiducial = np.array([67.37, 0.02233, 0.1198])
    _post_mean   = _samples_phys.mean(axis=0)
    _mse_raw     = np.mean((_post_mean - _true) ** 2)
    _mse_scaled  = np.mean(((_post_mean - _true) / _fiducial) ** 2)
    print(f"\n── Lensed summary ────────────────────────────────")
    print(f"  True values  : h0={_true[0]:.4f}  ombh2={_true[1]:.5f}  omch2={_true[2]:.4f}")
    print(f"  Posterior mean: h0={_post_mean[0]:.4f}  ombh2={_post_mean[1]:.5f}  omch2={_post_mean[2]:.4f}")
    print(f"  MSE (raw)    : {_mse_raw:.6f}")
    print(f"  MSE (scaled) : {_mse_scaled:.6f}  (normalised by fiducial²)")
    print(f"  Training time: {_train_time:.1f}s")


else:
    raise ValueError(f"Unknown DATASET: '{DATASET}'. Choose 'toy', 'unlensed', or 'lensed'.")
