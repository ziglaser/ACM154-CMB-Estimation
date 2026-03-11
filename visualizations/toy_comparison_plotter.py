import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import energy_score
from scipy.stats import gaussian_kde
from scipy.optimize import minimize

#load in analytic posterior
#load in MH samples
#load in NUTS samples

analytic_file = np.load('analytic_posterior_params.npz')
mh_file = np.load('toy_mcmc_mh_chains.npz')
nuts_file = np.load('toy_mcmc_nuts_chains.npz')


posterior_mean = analytic_file['posterior_mean']
posterior_covariance = analytic_file['posterior_covariance']
mh_mean1_chains = mh_file['mean1']
mh_mean2_chains = mh_file['mean2']
nuts_mean1_chains = nuts_file['mean1']
nuts_mean2_chains = nuts_file['mean2']

num_chains = 5
num_params = 2
num_samples = mh_mean1_chains.shape[1]

num_thinned_samples = mh_mean1_chains[0,::10].shape[0]
mh_final_matrix = np.empty((num_chains, num_params, num_thinned_samples))
mh_final_matrix[:, 0, :] = mh_mean1_chains[:,::10]
mh_final_matrix[:, 1, :] = mh_mean2_chains[:,::10]
mh_reshaped = mh_final_matrix.transpose(1, 0, 2).reshape(num_params, -1)

nuts_final_matrix = np.empty((num_chains, num_params, num_samples))
nuts_final_matrix[:, 0, :] = nuts_mean1_chains[:,:]
nuts_final_matrix[:, 1, :] = nuts_mean2_chains[:,:]
nuts_reshaped = mh_final_matrix.transpose(1, 0, 2).reshape(num_params, -1)

def make_plots():
    #make a contour plot
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(6, 10))
    vmin = 0
    vmax = 0.35

    #mh_hist = ax1.hist2d(mh_mean1_chains.flatten(), mh_mean2_chains.flatten(), bins=50, cmap='Blues', alpha=0.7, density=True)
    mh_hist = ax1.hist2d(mh_mean1_chains.flatten(), mh_mean2_chains.flatten(), bins=50, cmap='Blues', alpha=0.7, density=True, vmin = vmin, vmax = vmax)
    ax1.set_xlabel("mean1", fontsize = 14, fontweight = 'bold')
    ax1.set_ylabel("mean2", fontsize = 14, fontweight = 'bold')
    #plt.colorbar(mh_hist[3], ax=ax1, label='MCMC Sample Density')

    def analytic_posterior(mean1, mean2):
        means = np.array((mean1, mean2))
        return stats.multivariate_normal.pdf(means, posterior_mean, posterior_covariance)

    vectorized_analytic_posterior = np.vectorize(analytic_posterior)
    # Create grid for analytic posterior contours
    x_min, x_max = -1, 4
    y_min, y_max = -2, 2
    x_grid = np.linspace(x_min, x_max, 50)
    y_grid = np.linspace(y_min, y_max, 50)
    X, Y = np.meshgrid(x_grid, y_grid)
    # Evaluate analytic posterior on grid
    Z = vectorized_analytic_posterior(X, Y)
    # Overlay contours of analytic posterior
    contours = ax1.contour(X, Y, Z, levels=8, colors='red', linewidths=2, alpha=0.6)
    ax1.clabel(contours, inline=True, fontsize=8)

    ax1.set_title("Metropolis Hastings", fontweight = 'bold', fontsize = 16)




    #now get nuts posterior
    #nuts_hist = ax2.hist2d(nuts_mean1_chains.flatten(), nuts_mean2_chains.flatten(), bins=50, cmap='Blues', alpha=0.7, density=True)
    nuts_hist = ax2.hist2d(nuts_mean1_chains.flatten(), nuts_mean2_chains.flatten(), bins=50, cmap='Blues', alpha=0.7, density=True, vmin = vmin, vmax = vmax)
    ax2.set_xlabel("mean1", fontsize = 14, fontweight = 'bold')
    ax2.set_ylabel("mean2", fontsize = 14, fontweight = 'bold')
    fig.colorbar(mh_hist[3], ax=[ax1, ax2], label='MCMC Sample Density')
    #fig.colorbar(mh_hist[3], ax=ax2, label='MCMC Sample Density')
    vectorized_analytic_posterior = np.vectorize(analytic_posterior)
    # Create grid for analytic posterior contours
    x_min, x_max = -1, 4
    y_min, y_max = -2, 2
    x_grid = np.linspace(x_min, x_max, 50)
    y_grid = np.linspace(y_min, y_max, 50)
    X, Y = np.meshgrid(x_grid, y_grid)
    # Evaluate analytic posterior on grid
    Z = vectorized_analytic_posterior(X, Y)
    # Overlay contours of analytic posterior
    contours = ax2.contour(X, Y, Z, levels=8, colors='red', linewidths=2, alpha=0.6)
    ax2.clabel(contours, inline=True, fontsize=8)

    ax2.set_title("NUTS", fontweight = 'bold', fontsize = 16)

    ax1.set_xlim(-1, 4)
    ax2.set_xlim(-1,4)
    ax1.set_ylim(-2.5, 2.5)
    ax2.set_ylim(-2.5, 2.5)

    plt.savefig("toy_mcmc_sidebyside.png")
    plt.show()


def compute_energy_distance():
    num_chains = 5
    num_params = 2
    num_samples = mh_mean1_chains.shape[1]
    rng = np.random.default_rng()
    #rng = np.random.default_rng(0)

    #this samples from the MCMC distribution
    def sample_mcmc(sample_size, key = None, data_array = None, rng = None):
        n = data_array.shape[1]
        indices = rng.choice(n, size=sample_size, replace=True)
        return (data_array[:, indices].transpose(), rng) #this samples jointly, as required for correctly using MCMC samples
    
    def sample_mh(sample_size, key = None):
        return sample_mcmc(sample_size, data_array=mh_reshaped, rng = rng)
    
    def sample_nuts(sample_size, key = None):
        return sample_mcmc(sample_size, data_array=nuts_reshaped, rng = rng)
    
    def sample_analytic(sample_size, key = None):
        return (stats.multivariate_normal.rvs(mean = posterior_mean, cov = posterior_covariance, size = sample_size, random_state = rng), rng)
    
    def abs_fn(x): #scale by the fiducial values (here, the posterior mean) so that size differences of variables don't dominate
        return np.linalg.norm(x/posterior_mean, axis=-1)
    
    es1 = energy_score.energy_square_distance(sample_mh, sample_analytic, abs_fn, 6500) #energy distance mh vs analytic
    es2 = energy_score.energy_square_distance(sample_nuts, sample_analytic, abs_fn, 6500) #energy distance nuts vs analytic
    es3 = energy_score.energy_square_distance(sample_nuts, sample_mh, abs_fn, 6500) #energy distance nuts vs mh
    es4 = energy_score.energy_square_distance(sample_analytic, sample_analytic, abs_fn, 6500) #energy distance nuts vs mh

    print("mh vs analytic energy distance squared ", es1)
    print("nuts vs analytic energy distance squared ", es2)
    print("nuts vs mh energy distance squared", es3)
    print("analytic vs analytic energy distance squared", es4)


def compute_square_error():
    print("mh samples flat shape", mh_final_matrix.shape)
    mh_samples_flat = mh_final_matrix.transpose(1, 0, 2).reshape(num_params, -1)
    mh_kde = gaussian_kde(mh_samples_flat)
    print("computing x0", flush = True)
    x0 = np.median(mh_samples_flat, axis=1)
    print("computing MH kde MAP", flush = True)
    result = minimize(lambda x: -mh_kde.logpdf(x.reshape(-1, 1)), x0, method='Nelder-Mead')
    mh_map = result.x
    mh_squared_error = np.sum(((posterior_mean - mh_map)/posterior_mean)**2)

    print("nuts samples flat shape", nuts_final_matrix.shape)
    nuts_samples_flat = nuts_final_matrix.transpose(1, 0, 2).reshape(num_params, -1)
    nuts_kde = gaussian_kde(nuts_samples_flat)
    print("computing x0", flush = True)
    x0 = np.median(nuts_samples_flat, axis=1)
    print("computing NUTS kde MAP", flush = True)
    result = minimize(lambda x: -nuts_kde.logpdf(x.reshape(-1, 1)), x0, method='Nelder-Mead')
    nuts_map = result.x
    nuts_squared_error = np.sum(((posterior_mean - nuts_map)/posterior_mean)**2)


    print("mh squared error: ", mh_squared_error)
    print("nuts squared error: ", nuts_squared_error)

#compute_energy_distance()
#make_plots()
compute_square_error()
