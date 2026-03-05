import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import multivariate_normal


def triangle_plot(samples, param_names, bins1d=20, bins2d=10, figsize=None, color='steelblue',
                  hist2d_cmap='Blues', alpha=0.7, true_values=None, mean_values=None,
                  thin_factor=1, fig_name=None, dpi=150, bbox_inches='tight',
                  gaussian=None):
    """
    Make a triangle plot from Normalizing Flow samples.

    Parameters
    ----------
    samples : np.ndarray, shape (n_chains, n_params, n_samples)
    param_names : list of str
    bins1d : int
    bins2d : int
    figsize : tuple or None
    color : str, color for 1D histograms
    hist2d_cmap : str, colormap for 2D histograms
    alpha : float
    true_values : list or None
        True parameter values marked with red vertical lines / stars.
    mean_values : list or None
        Posterior mean values marked with green dashed lines / triangles.
    thin_factor : int or list of ints
        Thinning factor(s) for the chains.  If int, all parameters are thinned
        by the same factor.  If list, parameter i is thinned by thin_factor[i].
    fig_name : str or None
        If not None, the figure is saved under this name.
    dpi : int
    bbox_inches : str
    gaussian : tuple or None
        If provided, (mean, cov) where mean is shape (n_params,) and cov is
        shape (n_params, n_params).  The appropriate marginal Gaussian is
        overlaid in orange on every panel (1D curve on diagonal, contour on
        off-diagonal).
    """
    n_chains, n_params, n_samples = samples.shape

    # Normalise thin_factor to a list of length n_params
    if isinstance(thin_factor, int):
        thin_factors = [thin_factor] * n_params
    else:
        if len(thin_factor) != n_params:
            raise ValueError(f"thin_factor list length ({len(thin_factor)}) "
                             f"must match n_params ({n_params}).")
        thin_factors = thin_factor

    # Thin each parameter independently, then flatten chains
    flat = [samples[:, i, ::thin_factors[i]].ravel() for i in range(n_params)]

    if figsize is None:
        figsize = (2.5 * n_params, 2.5 * n_params)

    fig, axes = plt.subplots(n_params, n_params, figsize=figsize)
    if n_params == 1:
        axes = np.array([[axes]])

    # Unpack gaussian if provided
    if gaussian is not None:
        g_mean, g_cov = gaussian

    legend_handles = {}   # keyed by label, to build a shared figure legend

    for row in range(n_params):
        for col in range(n_params):
            ax = axes[row, col]
            if col > row:
                ax.set_visible(False)

            elif row == col:
                ax.hist(flat[row], bins=bins1d, color=color, alpha=alpha, density=True)
                if true_values is not None:
                    h = ax.axvline(true_values[row], color='red', linewidth=1.5,
                                   clip_on=False)
                    legend_handles['Truth'] = h
                if mean_values is not None:
                    h = ax.axvline(mean_values[row], color='orange', linewidth=1.5,
                                   linestyle='--', clip_on=False)
                    legend_handles['Posterior mean'] = h
                if gaussian is not None:
                    m = g_mean[row]
                    s = np.sqrt(g_cov[row, row])
                    x = np.linspace(flat[row].min(), flat[row].max(), 300)
                    h, = ax.plot(x, norm.pdf(x, m, s), color='orange', lw=1.5)
                    legend_handles['Gaussian overlay'] = h

            else:
                n = min(len(flat[col]), len(flat[row]))
                ax.hist2d(flat[col][:n], flat[row][:n], bins=bins2d, cmap=hist2d_cmap)
                if true_values is not None:
                    h, = ax.plot(true_values[col], true_values[row], 'r*',
                                 markersize=10, zorder=5, clip_on=False)
                    legend_handles['Truth'] = h
                if mean_values is not None:
                    h, = ax.plot(mean_values[col], mean_values[row], color='orange', marker='^',
                                 markersize=8, zorder=5, clip_on=False)
                    legend_handles['Posterior mean'] = h
                if gaussian is not None:
                    idx = [col, row]
                    m2 = g_mean[idx]
                    c2 = g_cov[np.ix_(idx, idx)]
                    x = np.linspace(flat[col].min(), flat[col].max(), 200)
                    y = np.linspace(flat[row].min(), flat[row].max(), 200)
                    X, Y = np.meshgrid(x, y)
                    pos = np.stack([X, Y], axis=-1)   # (200, 200, 2)
                    Z = multivariate_normal(mean=m2, cov=c2).pdf(pos)
                    h = ax.contour(X, Y, Z, levels=6, colors='orange', linewidths=1.5)
                    legend_handles['Gaussian overlay'] = h.legend_elements()[0][0]

            # Labels on edges only; diagonal panels always get an xlabel
            if col == 0:
                ax.set_ylabel(param_names[row])
            else:
                ax.set_ylabel('')
            if row == col or row == n_params - 1:
                ax.set_xlabel(param_names[col])
            else:
                ax.set_xlabel('')

    fig.tight_layout()

    # Shared figure-level legend
    if legend_handles:
        fig.legend(
            handles=list(legend_handles.values()),
            labels=list(legend_handles.keys()),
            loc='upper right',
            fontsize=9,
            framealpha=0.9,
            borderaxespad=0.5,
        )

    if fig_name is not None:
        plt.savefig(fig_name, dpi=dpi, bbox_inches=bbox_inches)
    return fig, axes


# --- Example usage ---
if __name__ == '__main__':
    rng = np.random.default_rng(42)
    n_chains, n_params, n_samples = 4, 3, 1000
    cov = np.array([[1.0, 0.7, -0.3],
                    [0.7, 1.0,  0.2],
                    [-0.3, 0.2, 1.0]])
    raw = rng.multivariate_normal(mean=[0, 1, 2], cov=cov,
                                  size=(n_chains, n_samples))
    samples = raw.transpose(0, 2, 1)   # (chains, params, samples)

    g_mean = np.array([0.1, 1.1, 1.9])

    fig, axes = triangle_plot(
        samples,
        param_names=[r'$\Omega_m$', r'$\sigma_8$', r'$H_0$'],
        true_values=[0, 1, 2],
        mean_values=[0.05, 1.05, 1.95],
        thin_factor=[1, 2, 3],
        gaussian=(g_mean, cov),
        fig_name='test_triangle_plot.png',
    )
    plt.show()
