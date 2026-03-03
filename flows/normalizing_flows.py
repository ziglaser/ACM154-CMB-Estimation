import torch
import torch.nn as nn
import torch.distributions as dist
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML, display
from tqdm.auto import tqdm
from matplotlib.animation import ArtistAnimation


class RealNVP(nn.Module):
    """
    Implementation of RealNVP (Real-valued Non-Volume Preserving) flow.
    It learns a bijective mapping T: Z -> X between a simple reference
    distribution (Z) and a complex target posterior (X).
    """
    def __init__(self, nets, nett, masks, ref_dist, pi_obj=None):
        super(RealNVP, self).__init__()
        # Reference distribution in latent space (usually Standard Gaussian)
        self.ref = ref_dist
        # Optional target log-posterior (for transport VI over parameters)
        self.pi = pi_obj
        self.mask = nn.Parameter(masks, requires_grad=False)

        # Scaling (s) and Translation (t) networks for each coupling layer
        self.s = nn.ModuleList([nets() for _ in range(len(masks))])
        self.t = nn.ModuleList([nett() for _ in range(len(masks))])

    def T(self, z):
        """
        Forward transform: Reference Z -> Target X.
        Used for sampling: x = T(z).
        """
        log_det_J, x = z.new_zeros(z.shape[0]), z
        for i in range(len(self.s)):
            x_ = x * self.mask[i]
            # s and t only depend on masked dimensions
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            # Affine transformation: x = x_gate * exp(s) + t
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
            log_det_J += s.sum(dim=1)
        return x, log_det_J

    def Tinv(self, x):
        """
        Inverse transform: Target X -> Reference Z.
        Used for density estimation and KL divergence: z = T_inv(x).
        """
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.s))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            # Inverse affine: z = (z_gate - t) * exp(-s)
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

    def approximate_log_prob(self, x):
        """
        Calculates the density of the push-forward distribution q(x).
        log q(x) = log ref(T_inv(x)) + log|det J_T_inv(x)|.
        """
        z, logdet = self.Tinv(x)
        return self.ref.log_prob(z) + logdet

    def sample(self, batch_size):
        """Generates samples from the flow q(x)."""
        z = self.ref.sample((batch_size,))
        x, _ = self.T(z)
        return x

    def log_prob_transport(self, z):
        """
        Transport-VI objective:
        log pi(T(z)) + log|det J_T(z)|.
        """
        if self.pi is None:
            raise RuntimeError("RealNVP.log_prob_transport called but self.pi is None.")
        x, logdet = self.T(z)
        return self.pi.log_prob(x) + logdet


def run_flow(
    lr=1e-3,
    num_epochs=200,
    batch_size=64,
    num_layers=6,
    hidden_dim=256,
    data_path="gaussian_samples.npy",
    show_anim=False,
):
    """
    Train a RealNVP normalizing flow directly on samples stored in ``data_path``
    via maximum likelihood.

    The loader is flexible enough to handle:
    - 2D point clouds, e.g. ``gaussian_samples.npy`` with shape (N, 2)
    - Single 2D maps, e.g. an unlensed CMB map with shape (H, W)
    - Stacks of maps, e.g. lensed simulations with shape (H, W, Nmaps)
      stored in an .npz file (it will use the ``f_tilde`` field if present).

    In all cases, the data are reshaped to a matrix of shape (N_samples, D),
    and the flow is defined in D dimensions.
    """
    # -------------------------------
    # Load and reshape data
    # -------------------------------
    raw = np.load(data_path, allow_pickle=False)

    # Handle .npz containers (e.g. lensed CMB sims from Julia)
    if isinstance(raw, np.lib.npyio.NpzFile):
        if "f_tilde" in raw.files:
            arr = raw["f_tilde"]
        elif "f" in raw.files:
            arr = raw["f"]
        else:
            raise ValueError(
                f"Unsupported .npz structure in {data_path}. "
                "Expected an array under key 'f_tilde' or 'f'."
            )
    else:
        arr = raw

    arr = np.asarray(arr, dtype=np.float32)

    # Cases:
    # - (N, D) point cloud (e.g. gaussian_samples): keep as is
    # - (H, W) single map: treat as one D=H*W-dimensional sample
    # - (H, W, Nmaps): treat each map as a sample, flatten to D=H*W
    if arr.ndim == 1:
        data_array = arr.reshape(-1, 1)
    elif arr.ndim == 2:
        # Heuristic: if it's "tall and skinny", assume (N, D) already.
        if arr.shape[0] > arr.shape[1] and arr.shape[1] <= 8:
            data_array = arr
        else:
            # Likely a single map (H, W)
            data_array = arr.reshape(1, -1)
    elif arr.ndim == 3:
        # Assume (H, W, Nmaps) and move maps to leading axis
        arr = np.moveaxis(arr, -1, 0)  # (Nmaps, H, W)
        data_array = arr.reshape(arr.shape[0], -1)
    else:
        raise ValueError(f"Unsupported data shape {arr.shape} loaded from {data_path}")

    data = torch.from_numpy(data_array)
    input_dim = data.shape[1]

    # Reference distribution in latent space (D-dimensional standard normal)
    ref = dist.MultivariateNormal(torch.zeros(input_dim), torch.eye(input_dim))

    # Network architectures for the coupling layers (dimension-agnostic)
    nets = lambda: nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LeakyReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LeakyReLU(),
        nn.Linear(hidden_dim, input_dim),
        nn.Tanh(),
    )
    nett = lambda: nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LeakyReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LeakyReLU(),
        nn.Linear(hidden_dim, input_dim),
    )

    # Alternating binary masks for D dimensions
    d = input_dim
    half = d // 2
    base_mask = torch.cat(
        [torch.zeros(half), torch.ones(d - half)]
    )  # first half masked, second active
    mask_list = []
    for layer in range(num_layers):
        if layer % 2 == 0:
            mask_list.append(base_mask)
        else:
            mask_list.append(1.0 - base_mask)
    masks = torch.stack(mask_list).float()

    model = RealNVP(nets, nett, masks, ref_dist=ref)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Simple DataLoader for minibatch training
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # -------------------------------
    # Visualization setup
    # -------------------------------
    is_2d = input_dim == 2

    if is_2d:
        # 2D point-cloud visualization (toy Gaussian case)
        x1_min, x1_max = data[:, 0].min().item() - 1.0, data[:, 0].max().item() + 1.0
        x2_min, x2_max = data[:, 1].min().item() - 1.0, data[:, 1].max().item() + 1.0

        x1_line = torch.linspace(x1_min, x1_max, 100)
        x2_line = torch.linspace(x2_min, x2_max, 100)
        xg = torch.meshgrid(x1_line, x2_line, indexing="ij")
        xx = torch.stack([xg[0].flatten(), xg[1].flatten()], dim=1)

        milestones = [0, int(num_epochs * 0.25), int(num_epochs * 0.5), num_epochs - 1]
        fig_static, axes = plt.subplots(1, 4, figsize=(20, 5))

        frames = []
        if show_anim:
            fig_anim, ax_anim = plt.subplots(figsize=(6, 5))
            # Plot the target samples as a background reference
            arr_2d = data_array if data_array.shape[1] == 2 else data.numpy()
            ax_anim.plot(arr_2d[:, 0], arr_2d[:, 1], ".r", markersize=2, alpha=0.3)
            plt.close(fig_anim)
    else:
        milestones = []
        frames = []
        xg = xx = None
        fig_static = axes = None
        fig_anim = ax_anim = None

    # For high-dimensional maps, optionally visualize example maps at the end
    is_map_like = arr.ndim in (2, 3)

    pbar = tqdm(range(num_epochs), desc=f"Training NF on {data_path}")
    plot_idx = 0

    for epoch in pbar:
        for (x_batch,) in loader:
            loss = -model.approximate_log_prob(x_batch).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # 2D visualization at milestones
        if is_2d and epoch in milestones:
            ax = axes[plot_idx]
            ax.clear()
            # Show the target data
            arr_2d = data_array if data_array.shape[1] == 2 else data.numpy()
            ax.plot(arr_2d[:, 0], arr_2d[:, 1], ".r", markersize=2, alpha=0.3, label="data")
            with torch.no_grad():
                xs = model.sample(batch_size).cpu().numpy()
                ax.plot(xs[:, 0], xs[:, 1], ".b", markersize=4, alpha=0.5, label="flow")
                log_qx = model.approximate_log_prob(xx)
                qx = torch.exp(log_qx).reshape(100, 100).numpy()
                ax.contour(
                    xg[0].numpy(),
                    xg[1].numpy(),
                    qx,
                    colors="black",
                    levels=7,
                )
            ax.set_title(f"Epoch {epoch}")
            if plot_idx == 0:
                ax.legend()
            plot_idx += 1

        # Optional 2D animation frames
        if is_2d and show_anim and (epoch % 5 == 0 or epoch == num_epochs - 1):
            with torch.no_grad():
                xs = model.sample(batch_size).cpu().numpy()
                qx = torch.exp(model.approximate_log_prob(xx)).reshape(100, 100).numpy()
                pts, = ax_anim.plot(xs[:, 0], xs[:, 1], ".b", markersize=4, alpha=0.5)
                cnt = ax_anim.contour(
                    xg[0].numpy(),
                    xg[1].numpy(),
                    qx,
                    colors="black",
                    levels=7,
                )
                txt = ax_anim.text(
                    0.05,
                    0.92,
                    f"Epoch: {epoch}",
                    transform=ax_anim.transAxes,
                    fontweight="bold",
                )
                frames.append(
                    [pts, txt] + list(cnt.collections if hasattr(cnt, "collections") else [cnt])
                )

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # Show final visualizations
    if is_2d and milestones:
        plt.figure(fig_static.number)
        plt.tight_layout()
        plt.show()

        if show_anim and frames:
            ani = ArtistAnimation(fig_anim, frames, interval=100, blit=False)
            display(HTML(ani.to_jshtml()))

    # Simple map visualization: original vs sample (if applicable)
    if is_map_like:
        if arr.ndim == 2:
            H, W = arr.shape
            original_map = arr
            with torch.no_grad():
                sample_vec = model.sample(1).cpu().numpy().reshape(-1)
            sample_map = sample_vec.reshape(H, W)
        else:  # arr.ndim == 3
            # Assume (H, W, Nmaps)
            H, W = arr.shape[0], arr.shape[1]
            original_map = arr[:, :, 0]
            with torch.no_grad():
                sample_vec = model.sample(1).cpu().numpy().reshape(-1)
            sample_map = sample_vec.reshape(H, W)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        im1 = ax1.imshow(original_map, cmap="viridis")
        ax1.set_title("Example data map")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        im2 = ax2.imshow(sample_map, cmap="viridis")
        ax2.set_title("Sample from flow")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

    return model


class CosmoPosterior:
    """
    Posterior over cosmological parameters given a map.

    Combines a Gaussian prior over theta with a user-supplied log-likelihood.
    """

    def __init__(self, prior_mean, prior_cov, log_likelihood_fn):
        """
        prior_mean: 1D torch.Tensor of shape (d,)
        prior_cov:  2D torch.Tensor of shape (d, d)
        log_likelihood_fn: callable taking theta [N, d] -> log p(data | theta) [N]
        """
        self.prior = dist.MultivariateNormal(prior_mean, prior_cov)
        self.log_likelihood_fn = log_likelihood_fn
        self.dim = prior_mean.shape[0]

    def log_prob(self, theta):
        """
        theta: [N, d] tensor of parameters.
        Returns log posterior up to a constant: log p(data | theta) + log p(theta).
        """
        log_prior = self.prior.log_prob(theta)
        log_lik = self.log_likelihood_fn(theta)
        return log_lik + log_prior


def run_transport_vi_for_cosmo(
    pi,
    lr=1e-3,
    num_epochs=200,
    batch_size=64,
    num_layers=6,
    hidden_dim=256,
):
    """
    Transport VI over cosmological parameters using RealNVP.

    pi: CosmoPosterior instance (or any object with .dim and .log_prob(x)).
    Returns a RealNVP flow that approximates the posterior over theta.
    """
    dim = pi.dim

    # Reference distribution over latent z (dim-dimensional standard normal)
    ref = dist.MultivariateNormal(torch.zeros(dim), torch.eye(dim))

    # Networks for coupling layers
    nets = lambda: nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.LeakyReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LeakyReLU(),
        nn.Linear(hidden_dim, dim),
        nn.Tanh(),
    )
    nett = lambda: nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.LeakyReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LeakyReLU(),
        nn.Linear(hidden_dim, dim),
    )

    # Alternating masks in parameter space
    half = dim // 2
    base_mask = torch.cat(
        [torch.zeros(half), torch.ones(dim - half)]
    )  # first half masked, second active
    mask_list = []
    for layer in range(num_layers):
        if layer % 2 == 0:
            mask_list.append(base_mask)
        else:
            mask_list.append(1.0 - base_mask)
    masks = torch.stack(mask_list).float()

    model = RealNVP(nets, nett, masks, ref_dist=ref, pi_obj=pi)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    pbar = tqdm(range(num_epochs), desc="Training transport VI for cosmology")

    for epoch in pbar:
        z_samples = model.ref.sample((batch_size,))
        loss = -model.log_prob_transport(z_samples).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return model
