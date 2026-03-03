from normalizing_flows import run_flow

# Choose which dataset to use by setting DATASET to one of:
# "toy"      -> ../data/gaussian_samples.npy
# "unlensed" -> ../data/hmc_unlensed_map_seed1000.npy
# "lensed"   -> ../data/all_CMB_simulations.npz
DATASET = "lensed"

if DATASET == "toy":
    data_path = "../data/gaussian_samples.npy"
elif DATASET == "unlensed":
    data_path = "../data/hmc_unlensed_map_seed1000.npy"
elif DATASET == "lensed":
    data_path = "../data/all_CMB_simulations.npz"
else:
    raise ValueError(f"Unknown DATASET: {DATASET}")

# Train the flow on the chosen dataset
flow = run_flow(
    lr=1e-3,
    num_epochs=200,
    batch_size=64,
    num_layers=6,
    hidden_dim=256,
    data_path=data_path,
    show_anim=False,
)

# # Sample from the flow
# samples = flow.sample(1000)
# print(samples.shape)

# # Plot the samples
# plt.scatter(samples[:, 0], samples[:, 1])
# plt.show()

# # Evaluate log-density of the flow
# log_density = flow.approximate_log_prob(samples)
# print(log_density.mean())
