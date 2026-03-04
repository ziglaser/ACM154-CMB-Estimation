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

    def log_prob_transport(self, z, beta=1.0):
        """
        Transport-VI objective:
        log pi_beta(T(z)) + log|det J_T(z)|,
        where pi_beta is the tempered posterior with likelihood weight beta.
        """
        if self.pi is None:
            raise RuntimeError("RealNVP.log_prob_transport called but self.pi is None.")
        x, logdet = self.T(z)
        return self.pi.log_prob(x, beta=beta) + logdet


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

    def log_prob(self, theta, beta=1.0):
        """
        theta: [N, d] tensor of parameters.
        beta:  likelihood temperature in (0, 1].  The tempered posterior is
               proportional to  p(data|theta)^beta * p(theta).
               beta=1 → full posterior; beta<1 → prior-dominant (useful during
               early training when the surrogate may not yet be trustworthy).
        Returns log tempered-posterior up to a constant.
        """
        log_prior = self.prior.log_prob(theta)
        log_lik = self.log_likelihood_fn(theta)
        return beta * log_lik + log_prior


def run_transport_vi_for_cosmo(
    pi,
    lr=1e-3,
    num_epochs=200,
    batch_size=64,
    num_layers=6,
    hidden_dim=256,
    show_anim=False,
    snapshot_epochs=None,
    snapshot_callback=None,
    beta_start=0.01,
    beta_end=1.0,
):
    """
    Transport VI over cosmological parameters using RealNVP.

    pi: CosmoPosterior instance (or any object with .dim, .log_prob(x, beta), and
        optionally .plot_posterior(ax)).

    beta_start / beta_end: likelihood temperature schedule.
        Beta is annealed linearly from beta_start (prior-dominant, epoch 0) to
        beta_end (full posterior, final epoch).  This lets the flow first learn a
        good shape from the prior and only gradually commit to the (potentially
        noisy) likelihood signal.

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

    # Alternating masks: [[0,1],[1,0]] pattern for 2D, generalised half-masking otherwise
    if dim == 2:
        mask_list = [[0, 1], [1, 0]] * (num_layers // 2)
        masks = torch.tensor(mask_list, dtype=torch.float32)
    else:
        half = dim // 2
        base_mask = torch.cat([torch.zeros(half), torch.ones(dim - half)])
        mask_list = [base_mask if layer % 2 == 0 else 1.0 - base_mask
                     for layer in range(num_layers)]
        masks = torch.stack(mask_list).float()

    model = RealNVP(nets, nett, masks, ref_dist=ref, pi_obj=pi)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Milestone visualisation (only for 2D parameter spaces)
    is_2d = dim == 2
    milestones = [0, int(num_epochs * 0.25), int(num_epochs * 0.5), num_epochs - 1]
    if is_2d:
        fig_static, axes = plt.subplots(1, 4, figsize=(20, 5))

        frames = []
        if show_anim:
            fig_anim, ax_anim = plt.subplots(figsize=(6, 5))
            if hasattr(pi, "plot_posterior"):
                pi.plot_posterior(ax=ax_anim)
            plt.close(fig_anim)

    pbar = tqdm(range(num_epochs), desc="Training transport VI for cosmology")
    plot_idx = 0

    for epoch in pbar:
        # Linear beta annealing: ramps from beta_start → beta_end over training
        frac = epoch / max(num_epochs - 1, 1)
        beta = beta_start + frac * (beta_end - beta_start)

        z_samples = model.ref.sample((batch_size,))
        loss = -model.log_prob_transport(z_samples, beta=beta).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Milestone static plots
        if is_2d and epoch in milestones:
            ax = axes[plot_idx]
            if hasattr(pi, "plot_posterior"):
                pi.plot_posterior(ax=ax)
            with torch.no_grad():
                xs = model.sample(batch_size).cpu().numpy()
                ax.plot(xs[:, 0], xs[:, 1], '.b', markersize=4, alpha=0.5)
            ax.set_title(f"Epoch {epoch}")
            plot_idx += 1

        # Animation frames
        if is_2d and show_anim and (epoch % 5 == 0 or epoch == num_epochs - 1):
            with torch.no_grad():
                xs = model.sample(batch_size).cpu().numpy()
                pts, = ax_anim.plot(xs[:, 0], xs[:, 1], '.b', markersize=4, alpha=0.5)
                txt = ax_anim.text(0.05, 0.92, f'Epoch: {epoch}',
                                   transform=ax_anim.transAxes, fontweight='bold')
                frames.append([pts, txt])

        # External snapshot callback (e.g. to save triangle plots at chosen epochs)
        if snapshot_callback is not None and snapshot_epochs is not None:
            if epoch in snapshot_epochs:
                snapshot_callback(model, epoch)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "beta": f"{beta:.4f}"})

    if is_2d:
        plt.figure(fig_static.number)
        plt.tight_layout()
        plt.show()

        if show_anim and frames:
            ani = ArtistAnimation(fig_anim, frames, interval=100, blit=False)
            display(HTML(ani.to_jshtml()))

    return model


def run_sampling_test(model, pi=None, n_samples=300,
                      u1_range=None, u2_range=None, z_range=None):
    """
    Visualise a trained 2D RealNVP flow in both target and latent space.

    Left panel  — target space x = T(z):
        - Orange filled background: true posterior pi(x) (if pi is provided).
        - Black contours: learned flow density q(x).
        - Blue dots: samples x ~ q(x).

    Right panel — latent space z = T^{-1}(x):
        - Green filled background: reference N(0, I).
        - Blue dots: same samples pushed back via T^{-1}.

    model:    trained RealNVP instance (2D).
    pi:       optional posterior object with .log_prob(x [N,2]) -> [N].
    n_samples: number of samples to draw.
    u1_range, u2_range: [min, max] for target-space axes (auto-detected if None).
    z_range:  [min, max] (symmetric) for latent-space axes.
    """
    u_res, z_res = 200, 200
    if z_range is None:
        z_range = [-4, 4]

    model.eval()
    with torch.no_grad():
        xs = model.sample(n_samples)
        zs, _ = model.Tinv(xs)

    xs_np = xs.cpu().numpy()
    zs_np = zs.cpu().numpy()

    # Auto-detect target-space range from samples if not given
    if u1_range is None:
        u1_range = [xs_np[:, 0].min() - 1.0, xs_np[:, 0].max() + 1.0]
    if u2_range is None:
        u2_range = [xs_np[:, 1].min() - 1.0, xs_np[:, 1].max() + 1.0]

    u1 = torch.linspace(u1_range[0], u1_range[1], u_res)
    u2 = torch.linspace(u2_range[0], u2_range[1], u_res)
    ug1, ug2 = torch.meshgrid(u1, u2, indexing='ij')
    uu = torch.stack([ug1.flatten(), ug2.flatten()], dim=1)

    z1 = torch.linspace(z_range[0], z_range[1], z_res)
    z2 = torch.linspace(z_range[0], z_range[1], z_res)
    zg1, zg2 = torch.meshgrid(z1, z2, indexing='ij')
    zz = torch.stack([zg1.flatten(), zg2.flatten()], dim=1)

    with torch.no_grad():
        log_qx = model.approximate_log_prob(uu)
        log_pz = model.ref.log_prob(zz)
        if pi is not None:
            log_pi = pi.log_prob(uu)

    plt.figure(figsize=(14, 6))

    # --- Left: target space ---
    plt.subplot(1, 2, 1)
    if pi is not None:
        pi_density = torch.exp(log_pi).reshape(u_res, u_res).numpy()
        plt.contourf(ug1.numpy(), ug2.numpy(), pi_density,
                     levels=100, cmap='Oranges', alpha=0.4)
    plt.contour(ug1.numpy(), ug2.numpy(),
                torch.exp(log_qx).reshape(u_res, u_res).numpy(),
                levels=8, colors='black', linewidths=0.8)
    plt.plot(xs_np[:, 0], xs_np[:, 1], '.b', markersize=4, alpha=0.5,
             label=r'Samples $x \sim q(x)$')
    plt.title(r"Target Space: $x = T(z)$")
    plt.xlabel("$u_1$"); plt.ylabel("$u_2$")
    plt.xlim(u1_range); plt.ylim(u2_range)
    plt.legend()

    # --- Right: latent space ---
    plt.subplot(1, 2, 2)
    pz_density = torch.exp(log_pz).reshape(z_res, z_res).numpy()
    max_pz = pz_density.max()
    levs_smooth = np.linspace(0, 1, 200) ** 1.5 * max_pz
    plt.contourf(zg1.numpy(), zg2.numpy(), pz_density,
                 levels=levs_smooth, cmap='Greens', alpha=0.85)
    plt.contour(zg1.numpy(), zg2.numpy(), pz_density,
                levels=levs_smooth[::25], colors='black', linewidths=0.6, alpha=0.4)
    plt.plot(zs_np[:, 0], zs_np[:, 1], '.b', markersize=4, alpha=0.5,
             label=r'Latent $z = T^{-1}(x)$')
    plt.title(r"Latent Space: $z = T^{-1}(x)$")
    plt.xlabel("$z_1$"); plt.ylabel("$z_2$")
    plt.xlim(z_range); plt.ylim(z_range)
    plt.legend()

    plt.tight_layout()
    plt.show()

    model.train()
