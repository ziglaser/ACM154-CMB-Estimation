import os
import sys
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
NUM_EPOCHS = 200
BATCH_SIZE = 64
NUM_LAYERS = 6
HIDDEN_DIM = 256
SHOW_ANIM  = False

# Likelihood temperature annealing.
# Beta is ramped linearly from BETA_START → BETA_END over NUM_EPOCHS.
#   BETA_START = 0.0  → pure prior at epoch 0 (flow learns a stable shape first)
#   BETA_END   = 1.0  → full posterior at the final epoch
# Lower BETA_END if the surrogate is still inaccurate (e.g. 0.1 until retrained).
BETA_START = 0.0
BETA_END   = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Shared posterior triangle-plot helper
# ─────────────────────────────────────────────────────────────────────────────

# Fiducial (true) cosmological parameters used for data generation
TRUE_VALUES = [67.37, 0.02233, 0.1198]   # h0, ombh2, omch2
PARAM_NAMES = ["$H_0$", r"$\omega_b$", r"$\omega_c$"]

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

    flow = run_transport_vi_for_cosmo(
        pi,
        lr=LR,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        num_layers=NUM_LAYERS,
        hidden_dim=HIDDEN_DIM,
        show_anim=SHOW_ANIM,
    )

    # Visualise: target space vs latent space, true posterior contour in orange
    run_sampling_test(
        flow,
        pi=pi,
        n_samples=400,
        u1_range=[-2, 3],
        u2_range=[-2, 2],
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

    noise_level    = 10 ** -8
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
    noise_var_t        = torch.tensor(noise_variances,    dtype=torch.float32)  # (64, 33)
    fourier_real_t     = torch.tensor(fourier_obs.real,   dtype=torch.float32)  # (64, 33)
    fourier_imag_t     = torch.tensor(fourier_obs.imag,   dtype=torch.float32)  # (64, 33)
    self_inverse_mask_t = torch.tensor(self_inverse_mask, dtype=torch.float32)  # (64, 33)
    upper_mask_t        = torch.tensor(upper_mask,        dtype=torch.float32)  # (64, 33)
    zero_mode_mask_t    = torch.tensor(zero_mode_mask,    dtype=torch.float32)  # (64, 33)

    def log_likelihood_fn_unlensed(theta):
        """
        theta: [N, 3] normalised parameter tensor (z-scored by param_mean_t / param_std_t).
        Returns log p(map | theta) of shape [N].

        Mirrors get_logpdfs_v2 from hmc_flexible_serial_production_version.py,
        with CosmoPower replaced by the differentiable PyTorch surrogate.

        Working in normalised space keeps the flow reference N(0,I) aligned with
        the prior N(0,I), avoiding the NaN explosion caused by scale mismatch.
        """
        # 1. theta is already normalised → feed directly to surrogate
        log_map_var = surrogate(theta)                              # [N, 64*33]
        map_var     = torch.exp(log_map_var).reshape(-1, 64, 33)   # [N, 64, 33]

        # 2. sigmas = sqrt(signal_variance + noise_variance)
        sigmas = torch.sqrt(map_var + noise_var_t.unsqueeze(0))    # [N, 64, 33]

        # 3. Log-pdf of observed Fourier coefficients (get_logpdfs_v2 in PyTorch)
        sqrt2 = 2 ** 0.5
        # Real-valued self-inverse modes: log N(x_real; 0, sigma)
        self_inv_lp = dist.Normal(0., sigmas).log_prob(
            fourier_real_t.unsqueeze(0))                           # [N, 64, 33]
        # Complex upper-triangle modes: each split into N(re;0,σ/√2)+N(im;0,σ/√2)
        real_lp = dist.Normal(0., sigmas / sqrt2).log_prob(
            fourier_real_t.unsqueeze(0))
        imag_lp = dist.Normal(0., sigmas / sqrt2).log_prob(
            fourier_imag_t.unsqueeze(0))

        zm = zero_mode_mask_t.unsqueeze(0)    # [1, 64, 33]
        si = self_inverse_mask_t.unsqueeze(0)
        um = upper_mask_t.unsqueeze(0)

        log_lik = (zm * si * self_inv_lp
                   + zm * um * real_lp
                   + zm * um * imag_lp).sum(dim=(1, 2))            # [N]
        return log_lik

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


# =============================================================================
# LENSED CMB  (10 maps from map_generation.jl; same parameter space)
# =============================================================================
elif DATASET == "lensed":
    # f_tilde shape from Julia: (64, 64, nsims) after cat(..., dims=3)
    lensed_npz  = np.load("../data/all_CMB_simulations.npz")
    f_tilde     = lensed_npz["f_tilde"].astype(np.float32)  # (64, 64, nsims)
    # Use the first map as the "observation" for posterior inference
    observed_map_np = f_tilde[:, :, 0]

    def log_likelihood_fn_lensed(theta):
        """
        theta: [N, 3] tensor of (h0, ombh2, omch2).
        Returns log p(lensed_map | theta) of shape [N].

        The lensed likelihood is more complex than the unlensed one because the
        lensing operation mixes Fourier modes non-linearly.  Typical approaches:
          - Simulation-based inference (SBI / neural likelihood estimation)
          - Marginalising over the unlensed map with HMC (as in map_generation.jl)
          - A learned summary statistic + Gaussian likelihood approximation

        Placeholder: returns zeros (prior-only training until likelihood is implemented).
        """
        return torch.zeros(theta.shape[0])

    # Prior in normalised space (same reasoning as unlensed)
    prior_mean = torch.zeros(3)
    prior_cov  = torch.eye(3)

    # Physical-space scales for un-normalising displayed samples
    phys_mean = torch.tensor([67.37,   0.02233, 0.1198])
    phys_std  = torch.tensor([6.0,     0.003,   0.015])

    snapshot_epochs = [
        0,
        NUM_EPOCHS // 4,
        NUM_EPOCHS // 2,
        3 * NUM_EPOCHS // 4,
        NUM_EPOCHS - 1,
    ]
    callback = make_snapshot_callback(phys_mean, phys_std, label="lensed", n_samples=1000)

    pi = CosmoPosterior(prior_mean, prior_cov, log_likelihood_fn_lensed)

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


else:
    raise ValueError(f"Unknown DATASET: '{DATASET}'. Choose 'toy', 'unlensed', or 'lensed'.")
