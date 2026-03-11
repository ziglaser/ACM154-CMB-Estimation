import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import energy_score
from scipy.stats import gaussian_kde
from scipy.optimize import minimize

toy_data = np.load("gaussian_samples.npy")
observed = toy_data[0,:]
print("observed data: ", observed)
covariance = np.array([
        [1.0, 0.5],   # variance in x = 1.0, covariance = 0.5
        [0.5, 2.0]    # covariance = 0.5, variance in y = 2.0
])

#get the analytic solution for the toy model
identity = np.array([
        [1.0, 0.0],
        [0.0, 1.0]])
inv_cov = np.linalg.inv(covariance)
analytic_posterior_covariance = np.linalg.inv(identity + inv_cov)
analytic_posterior_mean = np.matmul(analytic_posterior_covariance, np.matmul(inv_cov, observed))
mean2 = analytic_posterior_mean
cov2  = analytic_posterior_covariance


vi_file = np.load("toy_Gaussian_VI.npz")


mean1 = vi_file['means']
cov1  = vi_file['cov']

def make_plot():
    # Build a grid that comfortably covers both distributions
    all_means = np.stack([mean1, mean2])
    margin = 2  # standard-deviation-like padding
    x_min = all_means[:, 0].min() - margin
    x_max = all_means[:, 0].max() + margin
    y_min = all_means[:, 1].min() - margin
    y_max = all_means[:, 1].max() + margin
    resolution = 300
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))          # shape (res, res, 2)
    # Evaluate both PDFs on the grid
    Z1 = multivariate_normal(mean=mean1, cov=cov1).pdf(pos)
    Z2 = multivariate_normal(mean=mean2, cov=cov2).pdf(pos)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    # Distribution 1 — filled colour map (heatmap)
    heatmap = ax.imshow(Z1,
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        cmap="Blues",
        aspect="auto",
        alpha=0.85,
        )
    cbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("PDF — Variational Inference", fontsize=10)
    
    # Distribution 2 — contour lines only
    contours = ax.contour(
        X, Y, Z2,
        levels=8,
        colors="crimson",
        linewidths=1.5,
        linestyles="solid",
        )
    ax.clabel(contours, inline=True, fontsize=8, fmt="%.3f")
    
    # Mark the means
    ax.plot(*mean1, marker="x", color="navy",   markersize=10, markeredgewidth=2,
                label=f"VI Posterior Mean ({mean1[0]:.4f}, {mean1[1]:.4f})")
    ax.plot(*mean2, marker="x", color="crimson", markersize=10, markeredgewidth=2,
                label=f"Analytic Posterior Mean  ({mean2[0]:.4f}, {mean2[1]:.4f})")
    ax.set_xlabel(r"$\mu_1$")
    ax.set_ylabel(r"$\mu_2$")
    ax.set_title("Toy Model Comparison\n"
                "VI Gaussian: filled heatmap  |  Analytic Gaussian: contours", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("gaussians_comparison.png", dpi=150)
    plt.show()
    print("Plot saved to gaussians_comparison.png")
    
    #now print the percent error in the posterior mean and posterior covariance
    percent_error_means = np.abs(mean1 - mean2)/mean2
    percent_error_covs = np.abs(cov1 - cov2)/cov2
    print("percent error mean", percent_error_means)
    print("percent error covs", percent_error_covs)

    squared_error = np.sum(percent_error_means**2)
    print("MAP squared error", squared_error)


def compute_energy_score():
    rng = np.random.default_rng()
    #get a weighted r2 norm of a vector. Rescale so that the norm is not dominated by a single parameter.
    def abs_fn(x):
        return np.linalg.norm(x/mean2, axis=-1)
    
    def sample_p1(sample_size, key = None, mean = mean1, cov = cov1): #samples the vi distribution
        return (multivariate_normal.rvs(mean = mean, cov = cov, size = sample_size, random_state = rng), rng)
    
    def sample_p2(sample_size, key = None, mean = mean2, cov = cov2): #samples the vi distribution
        return (multivariate_normal.rvs(mean = mean, cov = cov, size = sample_size, random_state = rng), rng)

    es = energy_score.energy_square_distance(sample_p1, sample_p2, abs_fn, 6000)
    es2 = energy_score.energy_square_distance(sample_p1, sample_p1, abs_fn, 6000)

    print("energy score vi vs analytic", es)
    print("energy score analytic vs analytic", es2)


def get_squared_error():
    fmh = np.load("toy_mcmc_mh_chains.npz")
    mh_mean1_chain = fmh['mean1']
    mh_mean2_chain = fmh['mean2']

    final_matrix = np.empty((2, mh_mean1_chain.shape[0]))
    final_matrix[0,:] = mh_mean1_chain[:]
    final_matrix[1,:] = mh_mean2_chain[:]

    print("final matrix shape", final_matrix.shape)
    num_params = 2
    samples_flat = final_matrix.transpose(1, 0, 2).reshape(num_params, -1)
    kde = gaussian_kde(samples_flat)
    x0 = samples_flat[:, np.argmax(kde.logpdf(samples_flat))]
    result = minimize(lambda x: -kde.logpdf(x.reshape(-1, 1)), x0, method='Nelder-Mead')
    map_estimate = result.x
    print("kernel density estimation map estimate: ", map_estimate)
    squared_error = np.sum(((mean2 - map_estimate)/mean2)**2)
    print("MCMC mh squared error", squared_error)


    fnuts = np.load("toy_mcmc_nuts_chains.npz")
    nuts_mean1_chain = fnuts['mean1']
    nuts_mean2_chain = fnuts['mean2']

make_plot()
#compute_energy_score()