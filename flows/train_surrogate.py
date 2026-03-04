"""
train_surrogate.py

Step 1: Generates training data by sampling random (h0, ombh2, omch2)
        parameters and evaluating the CosmoPowerJAX emulator at each.

Step 2: Trains a small PyTorch MLP that maps (h0, ombh2, omch2) directly to
        the interpolated, scaled power spectrum at the 64×33 rfft2 Fourier-k
        grid used by the unlensed CMB likelihood.

        Output of the surrogate: log(map_variances) at each grid point,
        where map_variances = C_ell(theta) * unitful_factor, interpolated
        from ell space to the 2D Fourier-k grid.

Run once from the flows/ directory:
    python train_surrogate.py

Saves: cosmopower_surrogate.pt
"""
import numpy as np
import scipy.fft as fft
from scipy.interpolate import CubicSpline
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ


# ─────────────────────────────────────────────────────────────────────────────
# Fixed parameters (identical to hmc_flexible_serial_production_version.py)
# ─────────────────────────────────────────────────────────────────────────────
TAU           = 0.0540
NS            = 0.9652
LOG_AS        = np.log(10 * 2.08666022)
UNITFUL_FACTOR = 7428350250000.0         # converts dimensionless → µK²

# Map geometry (same as HMC code: npix=64, pixel_size=8 arcmin)
NPIX           = 64
PIXEL_SIZE_RAD = 8.0 * np.pi / (60 * 180)

# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute the fixed 64×33 rfft2 Fourier-k grid
# ─────────────────────────────────────────────────────────────────────────────
kx           = 2 * np.pi * fft.fftfreq(NPIX, d=PIXEL_SIZE_RAD)
ky           = 2 * np.pi * fft.rfftfreq(NPIX, d=PIXEL_SIZE_RAD)
ky_g, kx_g  = np.meshgrid(ky, kx)
FOURIER_K    = np.sqrt(kx_g ** 2 + ky_g ** 2)          # (64, 33)
FOURIER_K_FL = FOURIER_K.flatten()                       # (2112,)

# ─────────────────────────────────────────────────────────────────────────────
# Prior ranges (broad enough to cover HMC posterior)
# ─────────────────────────────────────────────────────────────────────────────
PRIOR = {
    "h0":    {"mean": 67.37,   "std": 6.0,   "lo": 55.0,   "hi": 80.0},
    "ombh2": {"mean": 0.02233, "std": 0.003,  "lo": 0.015,  "hi": 0.030},
    "omch2": {"mean": 0.1198,  "std": 0.015,  "lo": 0.08,   "hi": 0.16},
}
PARAM_MEAN = np.array([PRIOR["h0"]["mean"], PRIOR["ombh2"]["mean"], PRIOR["omch2"]["mean"]])
PARAM_STD  = np.array([PRIOR["h0"]["std"],  PRIOR["ombh2"]["std"],  PRIOR["omch2"]["std"]])

N_TRAIN = 20000
N_VAL   = 1000


# =============================================================================
# 1. Generate training data with CosmoPowerJAX
# =============================================================================
print("Initialising CosmoPowerJAX emulator …")
emulator     = CPJ(probe="cmb_tt")
emulator_ell = np.asarray(emulator.modes, dtype=np.float64)  # 1D array of ell values

print(f"Generating {N_TRAIN + N_VAL} random cosmologies …")
rng = np.random.default_rng(42)
h0_samp    = rng.normal(PRIOR["h0"]["mean"],    PRIOR["h0"]["std"],    N_TRAIN + N_VAL)
ombh2_samp = rng.normal(PRIOR["ombh2"]["mean"], PRIOR["ombh2"]["std"], N_TRAIN + N_VAL)
omch2_samp = rng.normal(PRIOR["omch2"]["mean"], PRIOR["omch2"]["std"], N_TRAIN + N_VAL)

h0_samp    = np.clip(h0_samp,    PRIOR["h0"]["lo"],    PRIOR["h0"]["hi"])
ombh2_samp = np.clip(ombh2_samp, PRIOR["ombh2"]["lo"], PRIOR["ombh2"]["hi"])
omch2_samp = np.clip(omch2_samp, PRIOR["omch2"]["lo"], PRIOR["omch2"]["hi"])

params = np.stack([h0_samp, ombh2_samp, omch2_samp], axis=1)   # (N, 3)

# Target: log(map_variances) at each Fourier-k grid point
log_map_var_all = np.zeros((len(params), len(FOURIER_K_FL)), dtype=np.float32)

for i in tqdm(range(len(params)), desc="Evaluating CosmoPower"):
    h0, ombh2, omch2 = params[i]
    cosmo_params = np.array([ombh2, omch2, h0 / 100, TAU, NS, LOG_AS])
    spectrum     = np.asarray(emulator.predict(cosmo_params)) * UNITFUL_FACTOR  # (n_ell,)

    # Cubic-spline interpolation from ell → fourier_k
    cs = CubicSpline(emulator_ell, spectrum, extrapolate=True)
    mv = cs(FOURIER_K_FL)
    mv = np.clip(mv, 1e-30, None)   # variances must be positive

    log_map_var_all[i] = np.log(mv).astype(np.float32)

print("Data generation complete.")


# =============================================================================
# 2. Train PyTorch surrogate MLP
# =============================================================================
X = torch.tensor((params - PARAM_MEAN) / PARAM_STD, dtype=torch.float32)   # normalised input
Y = torch.tensor(log_map_var_all, dtype=torch.float32)                       # log map variances

X_train, X_val = X[:N_TRAIN], X[N_TRAIN:]
Y_train, Y_val = Y[:N_TRAIN], Y[N_TRAIN:]

OUTPUT_DIM = len(FOURIER_K_FL)   # 64 * 33 = 2112


class CosmoPowerSurrogate(nn.Module):
    """
    MLP emulator: normalised (h0, ombh2, omch2) → log(map_variances) at
    the 64×33 rfft2 Fourier-k grid.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3,          512), nn.LeakyReLU(),
            nn.Linear(512,        512), nn.LeakyReLU(),
            nn.Linear(512,        512), nn.LeakyReLU(),
            nn.Linear(512,        256), nn.LeakyReLU(),
            nn.Linear(256, OUTPUT_DIM),
        )

    def forward(self, x):
        return self.net(x)


surrogate = CosmoPowerSurrogate()
optimizer = optim.Adam(surrogate.parameters(), lr=1e-3)
# Cosine annealing gives a smoother LR decay over 3000 epochs
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3000, eta_min=1e-5)

print(f"\nTraining surrogate (output dim = {OUTPUT_DIM}) …")
EPOCHS = 3000
best_val = float("inf")
best_state = None

for epoch in tqdm(range(EPOCHS), desc="Training surrogate"):
    surrogate.train()
    pred  = surrogate(X_train)
    loss  = nn.MSELoss()(pred, Y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if (epoch + 1) % 200 == 0:
        surrogate.eval()
        with torch.no_grad():
            val_loss = nn.MSELoss()(surrogate(X_val), Y_val).item()
        print(f"  Epoch {epoch+1:4d}: train={loss.item():.5f}  val={val_loss:.5f}")
        if val_loss < best_val:
            best_val  = val_loss
            best_state = {k: v.clone() for k, v in surrogate.state_dict().items()}

# Restore best weights
surrogate.load_state_dict(best_state)

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
torch.save(
    {
        "model_state": surrogate.state_dict(),
        "param_mean":  PARAM_MEAN,
        "param_std":   PARAM_STD,
        "fourier_k":   FOURIER_K,    # (64, 33) – saved for verification
        "output_dim":  OUTPUT_DIM,
    },
    "cosmopower_surrogate.pt",
)
print(f"\nSaved cosmopower_surrogate.pt  (best val MSE = {best_val:.5f})")
