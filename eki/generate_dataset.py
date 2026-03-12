import numpy as np
from tqdm import tqdm
from generate_cosmopower_unlensed_maps import generate_cosmopower_map

N_MAPS = 100
NOISE_LEVEL = 0.08
FIDUCIAL = dict(h0=67.37, omch2=0.1198, ombh2=0.02233)
OUT_PATH = "cmb_fiducial_dataset.npz"

maps = np.stack([
    generate_cosmopower_map(seed=i, noise_level=NOISE_LEVEL,
                            save=False, show=False, **FIDUCIAL)
    for i in tqdm(range(N_MAPS), desc="Generating maps")
], axis=-1)

