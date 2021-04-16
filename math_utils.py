#!/usr/bin/env python
# Module containing mathematical utility functions used elsewhere.

import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm


def ecdf(x):
    """Compute the formal empirical CDF.

    Parameters
    ----------
    x : array for which ecdf is to be computed

    Returns
    -------
    xs : support
    ys : value of the computed ecdf

    """
    xs = np.sort(x)
    ys = np.arange(1, len(x) + 1) / float(len(x))
    return xs, ys


def gen_samples(support, P_x, N=1000, low=None, high=None):
    """Generate samples from a specific probability distribution.
    The probability distribution in question is specified as an array of samples,
    in P_x, with a support given in the eponymous variable. N controls how many
    samples to generate, via the inverse transform sampling method.

    Parameters
    ----------
    support : ndarray, support for the probability distribution
    P_x : ndarray of function values, the probability distribution to sample from
    NUM_SAMP : integer, optional, the number of samples to generate

    Returns
    -------
    samples : ndarray, the samples generated from P_x

    """
    if np.trapz(P_x, support) > 1.0:
        P_x = P_x / np.trapz(P_x, support)  # normalize, if not normalized.

    if low is None:
        low = support.min()
    if high is None:
        high = support.max()

    dx = support[1] - support[0]  # spacing between the points, assumed to be uniform
    ecdf = np.cumsum(P_x) * dx  # compute empirical CDF

    if np.any(ecdf[1:] == ecdf[:-1]):
        # remove final data point (DONE: formalize this fix!)
        # >  We do this, since sometimes ecdf[-1] == ecdf[-2], due to floating point
        #    errors. Though one less (very close by) point does not impact the
        #    interpolation, but it is necessary to make input one-to-one.
        ecdf = ecdf[:-1]
        support = support[:-1]

    ecdf_inv_interp = interp1d(
        ecdf, support, kind="cubic"
    )  # fit a cubic spline to the inverse CDF

    samples = np.zeros(N)
    count = 0
    bar = tqdm(desc="Generating samples... ", total=N)
    rng = np.random.default_rng()

    while count < N:
        sample = ecdf_inv_interp(rng.random(1))
        if sample < low or sample > high:
            continue
        else:
            samples[count] = sample
            count += 1
            bar.update()

    return samples


def love_c(c_ns):
    """Compute the tidal deformability, given the compactness.
    This function computes the tidal deformability of a neutron star, given the
    compactness of the neutron star, by solving the C-Love relation for the
    tidal deformability.

    Parameters
    ----------
    c_ns : float, compactness of the neutron star

    Returns
    -------
    lambda_ns : float, tidal deformability of the neutron star

    """
    a_0 = 0.360
    a_1 = -0.0355
    a_2 = 0.000705
    lambda_ns = np.exp((-a_1 - np.sqrt(a_1 ** 2 - 4 * a_2 * (a_0 - c_ns))) / (2 * a_2))
    return lambda_ns
