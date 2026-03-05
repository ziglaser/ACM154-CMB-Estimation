import numpy as np

def energy_square_distance(sample_p1, sample_p2, abs_fn, sample_size, key = None):
    '''Empirically computes the energy squared distance.
    
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
    print("term1", term1)
    print("term2", term2)
    print('term3', term3)
    return term1 - term2 - term3


#example usage
if __name__ == '__main__':
    import scipy.stats as st
    
    def sample_p1(sample_size, key = None):
        return (st.norm.rvs(size = sample_size), None)
    
    def sample_p2(sample_size, key = None):
        return (st.norm.rvs(loc = 1, size = sample_size), None)

    #verify that for large sample sizes, the energy score of a distribution with itself goes to zero.
    es1 = energy_square_distance(sample_p1, sample_p1, np.abs, 500000)
    print("es1", es1)

    print()

    #now look at an energy score that is nonzero.
    es2 = energy_square_distance(sample_p1, sample_p2, np.abs, 10000)
    print("es2", es2)