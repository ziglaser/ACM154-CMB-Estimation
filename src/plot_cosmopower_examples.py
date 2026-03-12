import numpy as np
import matplotlib.pyplot as plt

from generate_cosmopower_unlensed_maps import (
    generate_cosmopower_map,
    compute_power_spectrum,
    generate_cosmopower_theory_spectrum,
)

FIDUCIAL = dict(h0=67.37, omch2=0.1198, ombh2=0.02233)
NOISE    = 0.08
N_BINS   = 50
SEEDS    = [0, 1]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for row, seed in enumerate(SEEDS):
    map_2d = generate_cosmopower_map(seed=seed, noise_level=NOISE,
                                     save=False, show=False, **FIDUCIAL)
    ps_data   = compute_power_spectrum(map_2d, n_bins=N_BINS)
    ps_theory = generate_cosmopower_theory_spectrum(noise_level=NOISE,
                                                    n_bins=N_BINS, **FIDUCIAL)

    # Power spectrum
    ax_ps = axes[row, 0]
    ell_bins = np.arange(N_BINS)
    ax_ps.plot(ell_bins, ps_data,   label="Transform", alpha=0.8)
    ax_ps.plot(ell_bins, ps_theory, label="Theory",      linestyle="--")
    ax_ps.set_yscale("log")
    ax_ps.set_ylim(10**2.5, 10**6.5)
    ax_ps.set_xlabel("bin index")
    ax_ps.set_ylabel("power")
    ax_ps.set_title(f"Power Spectrum  (seed={seed}, noise={NOISE})")
    ax_ps.legend()

    # CM map
    ax_map = axes[row, 1]
    im = ax_map.imshow(map_2d, cmap="viridis", origin="lower")
    plt.colorbar(im, ax=ax_map, label=r"")
    ax_map.set_title(f"CMB Map  (seed={seed}, noise={NOISE})")

plt.tight_layout()
plt.savefig("../figures/cosmopower_examples.png", dpi=150)
print("Saved figures/cosmopower_examples.png")
plt.show()
