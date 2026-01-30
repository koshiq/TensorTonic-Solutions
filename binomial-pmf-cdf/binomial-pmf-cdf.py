import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    # Write code here
    pmf = comb(n, k) * (p**k) * ((1 - p)**(n - k))
    ks = np.arange(0, k + 1)

    allPMF = comb(n, ks) * (p**ks) * ((1 - p)**(n - ks))
    cdf = np.sum(allPMF)

    return pmf, cdf