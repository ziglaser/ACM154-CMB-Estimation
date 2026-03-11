import numpy as np
import matplotlib.pyplot as plt

def triangle_plot(samples, param_names, bins1d = 20, bins2d = 10, figsize=None, color='steelblue',
                  hist2d_cmap='Blues', alpha=0.7, true_values=None, thin_factor=1, fig_name = None, dpi=150, bbox_inches='tight'):
    """
    Make a triangle plot from MCMC samples.

    Parameters
    ----------
    samples : np.ndarray, shape (n_chains, n_params, n_samples)
    param_names : list of str
    bins : int
    figsize : tuple or None
    color : str, color for 1D histograms
    hist2d_cmap : str, colormap for 2D histograms
    alpha : float
    true_values : list or None, true parameter values to mark with red stars
    thin_factor : int or list of ints, thinning factor(s) for the chains.
                  If int, all parameters are thinned by the same factor.
                  If list, parameter i is thinned by thin_factor[i].
    fig_name: str or None. If not None, the triangle plot is saved under this name.
    dpi: int, triangle plot is saved with this dpi
    bbox_inches: str, triangle plot is saved with this bbox_inches argument to savefig
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

    for row in range(n_params):
        for col in range(n_params):
            ax = axes[row, col]
            if col > row:
                ax.set_visible(False)
            elif row == col:
                ax.hist(flat[row], bins=bins1d, color=color, alpha=alpha, density=True)
                ax.set_yticks([])
                if true_values is not None:
                    ax.axvline(true_values[row], color='red')
            else:
                # 2D histogram: use the shorter of the two thinned arrays
                n = min(len(flat[col]), len(flat[row]))
                ax.hist2d(flat[col][:n], flat[row][:n], bins=bins2d, cmap=hist2d_cmap)
                if true_values is not None:
                    ax.plot(true_values[col], true_values[row], 'r*', markersize=10,
                            zorder=5)

            # Labels on edges only
            '''if col == 0 and row != 0:
                ax.set_ylabel(param_names[row])'''
            if col == 0:
                ax.set_ylabel(param_names[row], fontweight = 'bold', fontsize = 14)
            else:
                ax.set_ylabel('')
            if row == n_params - 1:
                ax.set_xlabel(param_names[col], fontweight = 'bold', fontsize = 14)
            else:
                ax.set_xlabel('')

            # Remove inner tick labels to reduce clutter
            '''if col > 0:
                ax.set_yticklabels([])
            if row < n_params - 1:
                ax.set_xticklabels([])'''

    fig.tight_layout()
    if fig_name is not None:
        plt.savefig(fig_name, dpi = dpi, bbox_inches = bbox_inches)
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
    samples = raw.transpose(0, 2, 1)  # (chains, params, samples)

    fig, axes = triangle_plot(samples, param_names=[r'$\Omega_m$', r'$\sigma_8$', r'$H_0$'],
                              true_values=[0, 1, 2], thin_factor=[1, 2, 3], fig_name = 'test_triangle_plot.png')
    plt.savefig('triangle_plot.png', dpi=150, bbox_inches='tight')
    plt.show()