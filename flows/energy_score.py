import numpy as np

def energy_square_distance(sample_p1, sample_p2, abs_fn, sample_size, key = None):
    '''Empirically computes the energy squared distance.
    This is more efficient than the other function in this script, but it does require that the samples output np.arrays of shape (sample_size, d) where d is such that p1, p2 are distributions on R^d.
    
    Parameters:
    sample_p1: a function with signature sample_p1(sample_size, key = None) that returns sample_size independent samples from distribution p1, as well as an rng key. The returned sample should be an array where the default '-' does subtraction in the vector space.
    sample_p2: a function with signature sample_p1(sample_size, key = None) that returns sample_size independent samples from distribution p2, as well as an rng key. The returned sample should be an array where the default '-' does subtraction in the vector space.
    Note that the rng key pattern here of accepting and returning keys is intended to give nice reproducibility with jax, but will work generally for properly defined sample functions.
    abs_fn: a vectorized function that computes the norm of the vectors returned by sample_p1 and sample_p2
    sample_size: an integer that is the number of samples to use for the empirical calculation.
    key: an RNG key. The type should be that required by sample_p1 and sample_p2

    Returns: a scalar value that is the empirical estimate of the energy square distance
    '''
    u, key = sample_p1(sample_size, key=key)
    v, key = sample_p2(sample_size, key=key)
    uprime, key = sample_p1(sample_size, key=key)
    vprime, key = sample_p2(sample_size, key=key)

    # All n^2 cross-distribution pairs: E[|u - v|]
    # shape (sample_size, sample_size) -> mean over all pairs
    term1 = 2.0 * np.mean(abs_fn(u[:, None] - v[None, :]))

    # All n(n-1) within-p1 off-diagonal pairs: E[|u - u'|]
    # Using the full matrix and masking the diagonal avoids bias from |u_i - u_i| = 0
    u_diff = abs_fn(u[:, None] - uprime[None, :])
    np.fill_diagonal(u_diff, 0.0)
    term2 = np.sum(u_diff) / (sample_size * (sample_size - 1))

    # All n(n-1) within-p2 off-diagonal pairs: E[|v - v'|]
    v_diff = abs_fn(v[:, None] - vprime[None, :])
    np.fill_diagonal(v_diff, 0.0)
    term3 = np.sum(v_diff) / (sample_size * (sample_size - 1))

    val = term1 - term2 - term3
    return max(val, 0.0)


def old_energy_square_distance(sample_p1, sample_p2, abs_fn, sample_size, key = None):
    '''Empirically computes the energy squared distance.
    Requirements for the interface of samplers are not as strict, but this has a higher variance in the estimate.
    
    Parameters:
    sample_p1: a function with signature sample_p1(sample_size, key = None) that returns sample_size independent samples from distribution p1, as well as an rng key. The returned sample should be an array where the default '-' does subtraction in the vector space.
    sample_p2: a function with signature sample_p1(sample_size, key = None) that returns sample_size independent samples from distribution p2, as well as an rng key. The returned sample should be an array where the default '-' does subtraction in the vector space.
    Note that the rng key pattern here of accepting and returning keys is intended to give nice reproducibility with jax, but will work generally for properly defined sample functions.
    abs_fn: a vectorized function that computes the norm of the vectors returned by sample_p1 and sample_p2
    sample_size: an integer that is the number of samples to use for the empirical calculation.
    key: an RNG key. The type should be that required by sample_p1 and sample_p2

    Returns: a scalar value that is the empirical estimate of the energy square distance
    '''
    u, key = sample_p1(sample_size, key = key)
    v, key = sample_p2(sample_size, key = key)
    term1 = 2.0 * np.mean(abs_fn(u-v))
    u, key = sample_p1(sample_size, key = key)
    uprime, key = sample_p1(sample_size, key = key)
    term2 = np.mean(abs_fn(u-uprime))
    v, key = sample_p2(sample_size, key = key)
    vprime, key = sample_p2(sample_size, key = key)
    term3 = np.mean(abs_fn(v-vprime))
    val = term1 - term2 - term3
    if val < 0:
        return 0.0 #the actual energy distance squared is always positive; any negative values are just noisy estimates of values that are small and nonnegative
    else:
        return val
    

#example usage
if __name__ == '__main__':
    import scipy.stats as st
    
    def sample_p1(sample_size, key = None):
        return (st.norm.rvs(size = sample_size), None)
    
    def sample_p2(sample_size, key = None):
        return (st.norm.rvs(loc = 1, size = sample_size), None)

    #verify that for large sample sizes, the energy score of a distribution with itself goes to zero.
    es1 = energy_square_distance(sample_p1, sample_p1, np.abs, 1000)
    es1_old = old_energy_square_distance(sample_p1, sample_p2, np.abs, 10000)
    print("es1", es1)
    print("es1 old", es1_old)

    print()

    #now look at an energy score that is nonzero.
    es2 = energy_square_distance(sample_p1, sample_p2, np.abs, 1000)
    es2_old = old_energy_square_distance(sample_p1, sample_p2, np.abs, 1000)
    print("es2", es2)
    print("es2 old", es2_old)