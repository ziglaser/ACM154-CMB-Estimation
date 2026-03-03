# Normalizing Flows
import torch
import torch.nn as nn
import torch.distributions as dist
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from IPython.display import HTML, display
import ipywidgets as widgets
from tqdm.auto import tqdm
from matplotlib.animation import ArtistAnimation

# --- RealNVP Model Components ---

class RealNVP(nn.Module):
    """
    Implementation of RealNVP (Real-valued Non-Volume Preserving) flow.
    It learns a bijective mapping T: Z -> X between a simple reference
    distribution (Z) and a complex target posterior (X).
    """
    def __init__(self, nets, nett, masks, pi_obj, ref_dist):
        super(RealNVP, self).__init__()
        self.pi = pi_obj   # Target log-posterior
        self.ref = ref_dist # Reference distribution (usually Standard Gaussian)
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

    def log_prob_transport(self, z):
        """
        Calculates log pi(T(z)) + log|det J_T(z)|.
        Maximizing this is equivalent to minimizing KL(q || pi).
        """
        x, logdet = self.T(z)
        return self.pi.log_prob(x) + logdet

    def approximate_log_prob(self, x):
        """
        Calculates the density of the push-forward distribution q(x).
        log q(x) = log ref(T_inv(x)) + log|det J_T_inv(x)|.
        """
        z, logdet = self.Tinv(x)
        return self.ref.log_prob(z) + logdet

    def sample(self, batch_size):
        """Generates samples from the variational distribution q(x)."""
        z = self.ref.sample((batch_size,))
        x, _ = self.T(z)
        return x

# --- Training Execution Function ---

def run_transport_vi(lr=1e-3, num_epochs=200, batch_size=64,
                     num_layers=6, hidden_dim=256, show_anim=False):
    """
    Executes the VI training loop and returns the trained RealNVP model.
    """
    # Setup Target and Reference
    ref = dist.MultivariateNormal(torch.zeros(2), torch.eye(2))

    # Define Network Architectures
    nets = lambda: nn.Sequential(nn.Linear(2, hidden_dim), nn.LeakyReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
                                 nn.Linear(hidden_dim, 2), nn.Tanh())
    nett = lambda: nn.Sequential(nn.Linear(2, hidden_dim), nn.LeakyReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
                                 nn.Linear(hidden_dim, 2))

    # Create alternating binary masks
    mask_list = [[0, 1], [1, 0]] * (num_layers // 2)
    masks = torch.tensor(mask_list, dtype=torch.float32)

    model = RealNVP(nets, nett, masks, pi, ref)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Visualization setup
    milestones = [0, int(num_epochs * 0.25), int(num_epochs * 0.5), num_epochs - 1]
    fig_static, axes = plt.subplots(1, 4, figsize=(20, 5))

    x1_line = torch.linspace(-1.5, 1.5, 100)
    x2_line = torch.linspace(-0.75, 2.25, 100)
    xg = torch.meshgrid(x1_line, x2_line, indexing='ij')
    xx = torch.stack([xg[0].flatten(), xg[1].flatten()], dim=1)

    frames = []
    if show_anim:
        fig_anim, ax_anim = plt.subplots(figsize=(6, 5))
        pi.plot_posterior(ax=ax_anim)
        plt.close(fig_anim)

    pbar = tqdm(range(num_epochs), desc="Training Transport VI")
    plot_idx = 0

    for epoch in pbar:
        # Optimization
        z_samples = model.ref.sample((batch_size,))
        loss = -model.log_prob_transport(z_samples).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Plotting Static Milestones
        if epoch in milestones:
            ax = axes[plot_idx]
            pi.plot_posterior(ax=ax)
            with torch.no_grad():
                xs = model.sample(batch_size).cpu().numpy()
                ax.plot(xs[:, 0], xs[:, 1], '.b', markersize=4, alpha=0.5)
                log_qx = model.approximate_log_prob(xx)
                qx = torch.exp(log_qx).reshape(100, 100).numpy()
                ax.contour(xg[0].numpy(), xg[1].numpy(), qx, colors='black', levels=7)
            ax.set_title(f"Epoch {epoch}")
            plot_idx += 1

        # Recording Animation
        if show_anim and (epoch % 5 == 0 or epoch == num_epochs - 1):
            with torch.no_grad():
                xs = model.sample(batch_size).cpu().numpy()
                qx = torch.exp(model.approximate_log_prob(xx)).reshape(100, 100).numpy()
                pts, = ax_anim.plot(xs[:, 0], xs[:, 1], '.b', markersize=4, alpha=0.5)
                cnt = ax_anim.contour(xg[0].numpy(), xg[1].numpy(), qx, colors='black', levels=7)
                txt = ax_anim.text(0.05, 0.92, f'Epoch: {epoch}', transform=ax_anim.transAxes, fontweight='bold')
                frames.append([pts, txt] + list(cnt.collections if hasattr(cnt, 'collections') else [cnt]))

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    plt.figure(fig_static.number); plt.tight_layout(); plt.show()

    if show_anim and frames:
        ani = ArtistAnimation(fig_anim, frames, interval=100, blit=False)
        display(HTML(ani.to_jshtml()))

    return model

# # --- UI Setup ---

# lr_in = widgets.FloatLogSlider(value=1e-3, base=10, min=-4, max=-1, step=0.1, description='LR:')
# epochs_in = widgets.IntSlider(value=100, min=50, max=1000, step=50, description='Epochs:')
# layers_in = widgets.IntSlider(value=6, min=2, max=12, step=2, description='Layers:')
# dim_in = widgets.Dropdown(options=[64, 128, 256, 512], value=256, description='Hidden Dim:')
# bs_in = widgets.IntSlider(value=64, min=32, max=256, step=32, description='Batch:')
# anim_in = widgets.Checkbox(value=False, description='Show Animation')

# ui_box = widgets.VBox([
#     widgets.HBox([lr_in, epochs_in, bs_in]),
#     widgets.HBox([layers_in, dim_in, anim_in])
# ])
# run_btn = widgets.Button(description="Run Transport VI", button_style='info')
# out = widgets.Output()

# # Placeholder for the trained model in the global scope
# flow = None

# def on_click(b):
#     global flow
#     with out:
#         out.clear_output(wait=True)
#         # Train and capture the model output
#         flow = run_transport_vi(lr_in.value, epochs_in.value, bs_in.value,
#                                 layers_in.value, dim_in.value, anim_in.value)

# run_btn.on_click(on_click)
# display(ui_box, run_btn, out)