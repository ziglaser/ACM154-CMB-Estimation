"""
flows_energy_score.py

Computes the energy score (squared energy distance) between:
  P1 = HMC posterior  (ground truth, loaded from saved chains)
  P2 = Normalizing flow posterior  (loaded from saved samples)

for the unlensed CMB inverse problem.

Usage:
  1. Run run_flows.py with DATASET = "unlensed" to produce:
       unlensed_flow_samples.npy          (5000 flow samples in physical space)
  2. Run hmc_flexible_serial_production_version.py to produce:
       unlensed_cmb_hmc_chains_seed0_gaussianprior.npz
  3. Run this script from the flows/ directory:
       python flows_energy_score.py

Interpretation:
  - ES² = 0 means P1 and P2 are identical distributions.
  - Compare against the self-energy scores (P1 vs P1, P2 vs P2) to get
    a noise floor: any ES² smaller than that is statistically indistinguishable
    from zero given the sample sizes used.
"""

import numpy as np
import energy_score

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
SEED          = 0
THIN_FACTOR   = 3       # thinning applied to HMC chains (match HMC script)
SAMPLE_SIZE   = 4000    # samples used per energy score evaluation
rng = np.random.default_rng(SEED)

# Fiducial values used to scale each parameter so no single dimension dominates
# the norm.  Same approach as vi_energy_score.py.
FIDUCIAL = np.array([67.37, 0.02233, 0.1198])   # h0, ombh2, omch2

# ─────────────────────────────────────────────────────────────────────────────
# Load HMC chains  (P1 — ground truth)
# ─────────────────────────────────────────────────────────────────────────────
hmc_file  = np.load(f"../data/unlensed_cmb_hmc_chains_seed{SEED}_gaussianprior.npz")
h0_chains    = hmc_file["h0_chains"]
ombh2_chains = hmc_file["ombh2_chains"]
omch2_chains = hmc_file["omch2_chains"]
hmc_file.close()

num_chains, _ = h0_chains.shape
num_params = 3

# Thin chains and reshape to (3, total_samples)
thinned       = h0_chains[:, ::THIN_FACTOR]
_, n_thinned  = thinned.shape
hmc_matrix    = np.empty((num_chains, num_params, n_thinned))
hmc_matrix[:, 0, :] = h0_chains[:, ::THIN_FACTOR]
hmc_matrix[:, 1, :] = ombh2_chains[:, ::THIN_FACTOR]
hmc_matrix[:, 2, :] = omch2_chains[:, ::THIN_FACTOR]
# (num_params, total_samples)
hmc_flat = hmc_matrix.transpose(1, 0, 2).reshape(num_params, -1)
print(f"HMC samples: {hmc_flat.shape[1]} (after thinning by {THIN_FACTOR})")

def sample_hmc(sample_size, key=None, data=hmc_flat, rng=rng):
    """Bootstrap from HMC chain samples."""
    n = data.shape[1]
    idx = rng.choice(n, size=sample_size, replace=True)
    return data[:, idx].T, rng   # (sample_size, 3)

# ─────────────────────────────────────────────────────────────────────────────
# Load flow samples  (P2 — normalizing flow posterior)
# ─────────────────────────────────────────────────────────────────────────────
flow_samples = np.load("../data/unlensed_flow_samples.npy")   # (N, 3) physical space
print(f"Flow samples: {flow_samples.shape[0]}")

def sample_flow(sample_size, key=None, data=flow_samples, rng=rng):
    """Bootstrap from saved flow samples."""
    n = data.shape[0]
    idx = rng.choice(n, size=sample_size, replace=True)
    return data[idx], rng   # (sample_size, 3)

# ─────────────────────────────────────────────────────────────────────────────
# Weighted norm: normalise each parameter by its fiducial value so h0
# (~67) doesn't dominate over ombh2 (~0.02) in the distance calculation.
# ─────────────────────────────────────────────────────────────────────────────
def abs_fn(x):
    return np.linalg.norm(x / FIDUCIAL, axis=-1)

# ─────────────────────────────────────────────────────────────────────────────
# Compute energy scores
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nComputing energy scores (sample_size={SAMPLE_SIZE}) …")

es = energy_score.energy_square_distance(
    sample_hmc, sample_flow, abs_fn, SAMPLE_SIZE)

# Self-scores give the noise floor: variance due to finite sample size.
# ES²(P, P) should be ~0 in theory; any non-zero value is sampling noise.
es_hmc_self  = energy_score.energy_square_distance(
    sample_hmc, sample_hmc, abs_fn, SAMPLE_SIZE)
es_flow_self = energy_score.energy_square_distance(
    sample_flow, sample_flow, abs_fn, SAMPLE_SIZE)

print(f"\nES²(HMC || Flow)   = {es:.6f}")
print(f"ES²(HMC || HMC)    = {es_hmc_self:.6f}  ← HMC noise floor")
print(f"ES²(Flow || Flow)  = {es_flow_self:.6f}  ← Flow noise floor")
print()
if es <= max(es_hmc_self, es_flow_self) * 3:
    print("Result: Flow and HMC posteriors are statistically similar "
          "(ES² within ~3× noise floor).")
else:
    ratio = es / max(es_hmc_self, es_flow_self)
    print(f"Result: ES² is {ratio:.1f}× the noise floor — "
          "the flow posterior differs meaningfully from HMC.")
