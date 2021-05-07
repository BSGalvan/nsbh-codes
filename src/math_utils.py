#!/usr/bin/env python
# Module containing mathematical utility functions used elsewhere.

import numpy as np
from numba import njit
from scipy.interpolate import interp1d
from tqdm import tqdm

from nsbh_merger import G, C, PI, M_SUN


def f_lso(m_tot):
    """Compute the last stable orbit of a black hole with total mass m_tot."""
    return C ** 3 / (6 ** 1.5 * PI * (m_tot * M_SUN) * G)


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


@njit
def rcap_isco(chi_bh=0):
    """Calculate the normalized ISCO radius for a given BH spin."""
    # Function Body
    z1 = 1 + (1 - chi_bh ** 2) ** (1 / 3) * (
        (1 + chi_bh) ** (1 / 3) + (1 - chi_bh) ** (1 / 3)
    )
    z2 = np.sqrt(3 * chi_bh ** 2 + z1 ** 2)
    retval = 3 + z2 - np.sign(chi_bh) * np.sqrt((3 - z1) * (3 + z1 + 2 * z2))
    return retval


@njit
def c_love(lambda_NS):
    """Compute the compactness of a NS using the C-Love relation."""
    # Function Body
    return 0.36 - 0.0355 * np.log(lambda_NS) + 0.000705 * np.log(lambda_NS) ** 2


@njit
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


@njit
def baryonic_mass(m_NS, C_NS):
    """Calculate the total baryonic mass."""
    # Function Body
    m_b = m_NS * (1 + (0.6 * C_NS) / (1 - 0.5 * C_NS))
    return m_b


def compute_masses(m_BH, chi_BH, m_NS=1.4, lambda_NS=330):
    """Compute masses left outside the BH apparent radius, bound & unbound.
    The various masses are the remnant mass, m_out, which further consists of
    the dynamic mass, m_dyn, and the disc mass, m_disc. Thus,
    m_out = m_disc + m_out  (if the NS does not plunge into the BH)
    """
    # Common binary parameters
    q = m_BH / m_NS
    eta = q / (1 + q) ** 2
    rho = (15 * lambda_NS) ** (-1 / 5)
    C_NS = c_love(lambda_NS)
    m_b = baryonic_mass(m_NS, C_NS)
    f = 0.3  # upper-limit to m_dyn, as a fraction of m_out

    # Compute mass_out. Formula from ^Foucart et al., 2018
    alpha, beta, gamma, delta = 0.308, 0.124, 0.283, 1.536
    term_alpha = alpha * (1 - 2 * rho) / (eta ** (1 / 3))
    term_beta = -beta * rcap_isco(chi_BH) * rho / eta
    term_gamma = gamma
    mass_out = m_b * (np.maximum(term_alpha + term_beta + term_gamma, 0.0)) ** delta

    # Compute mass_dyn. Formula from ^Kawaguchi et al., 2016
    a1, a2, a3, a4, n1, n2 = 4.464e-2, 2.269e-3, 2.431, -0.4159, 0.2497, 1.352
    term_a1 = a1 * (q ** n1) * (1 - 2 * C_NS) / C_NS
    term_a2 = -a2 * (q ** n2) * (rcap_isco(chi_BH))
    term_a3 = a3 * (1 - m_NS / m_b)
    term_a4 = a4
    mass_dyn = m_b * np.maximum(term_a1 + term_a2 + term_a3 + term_a4, 0)

    # Enforce upper limits on mass_dyn.
    if mass_dyn.size > 1:
        mask = mass_dyn > f * mass_out
        mass_dyn[mask] = f * mass_out[mask]
    else:
        mass_dyn = f * mass_out if mass_dyn > f * mass_dyn else mass_dyn

    # Compute m_disc
    mass_disc = np.maximum(mass_out - mass_dyn, 0)

    return mass_out, mass_dyn, mass_disc


def compute_velmax(velrms, velmin):
    """Compute the maximum velocity, given RMS and minimum velocities.

    :returns: maximum velocity of a given ejecta

    """
    velmax = np.sqrt(3 * velrms ** 2 - 0.75 * velmin ** 2) - 0.5 * velmin

    return velmax


def cart2sph(x, y, z):
    """Convert cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    x : float or ndarray
    y : float or ndarray
    z : float or ndarray

    Returns
    -------
    Spherical coordinates of the point (x, y, z)

    """
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    phi = np.arctan(y / x)
    return r, theta, phi


def sph2cart(r, theta, phi):
    """Convert spherical coordinates to cartesian coordinates.

    Parameters
    ----------
    r : radial coordinate
    theta : latitudinal coordinate, measured from the pole.
    phi : longitudinal coordinate

    Returns
    -------
    Cartesian coordinates of the point (r, theta, phi)

    """
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    return x, y, z
