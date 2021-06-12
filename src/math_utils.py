#!/usr/bin/env python
# Module containing mathematical utility functions used elsewhere.
# Original Author: B.S. Bharath Saiguhan, github.com/bsgalvan

import numpy as np
from numba import njit
from scipy.interpolate import interp1d
from scipy.optimize import toms748
from tqdm import tqdm

from math_constants import G, C, PI, M_SUN


def f_lso(m_tot):
    """Compute the frequency of last stable orbit of a black hole with total mass m_tot.

    Parameters
    ----------
    m_tot : float
    Total mass of the system post-merger

    Returns
    -------
    f_lso : float
    Frequency corresponding to last stable orbit

    """
    return C ** 3 / (6 ** 1.5 * PI * (m_tot * M_SUN) * G)


@njit
def ecdf(x):
    """Compute the formal empirical CDF of a given array, x

    Parameters
    ----------
    x : numpy.ndarray
    Array for which ecdf is to be computed

    Returns
    -------
    xs : numpy.ndarray
    Support
    ys : numpy.ndarray
    Value of the computed ecdf

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
    support : numpy.ndarray
    Support for the probability distribution
    P_x : numpy.ndarray of function values
    The probability distribution to sample from
    NUM_SAMP : integer, optional. Default = 1000
    The number of samples to generate

    Returns
    -------
    samples : numpy.ndarray
    The samples generated from P_x

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
def rcap_isco(chi_bh=0.0):
    """Calculate the normalized ISCO radius for a given BH spin.

    Parameters
    ----------
    chi_bh : float or numpy.ndarray, optional. Default = 0.0
    Dimensionless spin of the Black Hole

    Returns
    -------
    Normalized radius of the Innermost Stable Circular Orbit

    """
    # Function Body
    z1 = 1 + (1 - chi_bh ** 2) ** (1 / 3) * (
        (1 + chi_bh) ** (1 / 3) + (1 - chi_bh) ** (1 / 3)
    )
    z2 = np.sqrt(3 * chi_bh ** 2 + z1 ** 2)
    retval = 3 + z2 - np.sign(chi_bh) * np.sqrt((3 - z1) * (3 + z1 + 2 * z2))
    return retval


@njit
def f(nu=0.25):
    """Compute the transition function given mass ratio.
    See Pannarale, 2013 for more information.
    """
    if nu <= 0.16:
        retval = 0
    elif 0.16 < nu < 2 / 9:
        retval = 0.5 * (1 - np.cos((PI * (nu - 0.16)) / (2 / 9 - 0.16)))
    elif 2 / 9 <= nu <= 0.25:
        retval = 1
    return retval


@njit
def l_z(r, a):
    """Compute the orbital angular momentum per unit mass of a test particle.

    This is computed in the Kerr spacetime, using Boyer-Lindquist coordinates.

    Parameters
    ----------
    r : radial Boyer-Lindquist coordinate
    a : dimensionless spin of the (remnant) black hole

    Returns
    -------
    Orbital Angular momentum per unit mass of the test particle orbiting the BH

    """
    numerator = r ** 2 - np.sign(a) * 2 * a * np.sqrt(r) + a ** 2
    denominator = np.sqrt(r) * np.sqrt(
        (r ** 2 - 3 * r + np.sign(a) * 2 * a * np.sqrt(r))
    )
    return np.sign(a) * numerator / denominator


@njit
def e(r, a):
    """Compute the energy per unit mass of a test particle.

    This is computed in the Kerr spacetime, using Boyer-Lindquist coordinates.

    Parameters
    ----------
    r : radial Boyer-Lindquist coordinate
    a : dimensionless spin of the (remnant) black hole

    Returns
    -------
    Energy per unit mass of the test particle orbiting the BH

    """
    numerator = r ** 2 - 2 * r + np.sign(a) * np.sqrt(r)
    denominator = r * np.sqrt((r ** 2 - 3 * r + np.sign(a) * 2 * a * np.sqrt(r)))
    return numerator / denominator


@njit
def pannarale_func(x, chi_bh=0.1, mass_bh=3, mass_ns=1.4, C_ns=0.18, m_rem=0.4):
    """Compute function defined in Eq. 11 of [Pannarale, 2013] in root-solving form.

    Essentially, this is a function g(x) such that we want to solve:
        g(x) = x - f(x) = 0

    And f(x) is the big, complicated function from RHS of Eq. 11, [Pannarale, 2013]

    """

    # Redefine masses in geometrized units

    geom_mass_bh, geom_mass_ns, geom_m_rem = (
        (mass_bh * M_SUN * G) / (C ** 2),
        (mass_ns * M_SUN * G) / (C ** 2),
        (m_rem * M_SUN * G) / (C ** 2),
    )

    mass_ratio = geom_mass_bh / geom_mass_ns
    nu = mass_ratio / (1 + mass_ratio) ** 2
    f_nu = f(nu)
    risco_i = rcap_isco(chi_bh)
    risco_f = rcap_isco(x)
    lz = l_z(risco_f, x)
    ez_i = e(risco_i, chi_bh)
    ez_f = e(risco_f, x)
    geom_mb = (baryonic_mass(mass_ns, C_ns) * G * M_SUN) / (C ** 2)
    geom_M = ((mass_ns + mass_bh) * G * M_SUN) / (C ** 2)

    numerator = chi_bh * geom_mass_bh ** 2 + lz * geom_mass_bh * (
        (1 - f_nu) * geom_mass_ns + f_nu * geom_mb - geom_m_rem
    )
    denominator = (geom_M * (1 - (1 - ez_i) * nu) - ez_f * geom_m_rem) ** 2

    return x - numerator / denominator


def rootfind(chi_bh=0.1, mass_bh=3, mass_ns=1.4, C_ns=0.18, m_rem=0.4):
    """Solve for the root (remnant BH mass) using the given inputs."""
    root = toms748(
        pannarale_func,
        0,
        0.99999,
        args=(chi_bh, mass_bh, mass_ns, C_ns, m_rem),
    )
    return root


@njit
def c_love(lambda_NS):
    """Compute the compactness of a NS using the C-Love relation.

    For more information, see Yagi & Yunes, 2017.
    The compactness of a neutron star with a radius R_NS and mass M_NS
    is given by
    C_NS = G * M_NS / (R_NS * C**2)
      Where G = Universal Gravitational Constant
            C = Speed of light in vacuum.

    Parameters
    ----------
    lambda_NS : float
    Tidal deformability of the NS

    Returns
    -------
    Compactness of the NS, defined as C_NS = G * M_NS / (R_NS * C**2)

    """
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
    """Calculate the total baryonic mass for a given NS mass and compactness.

    Parameters
    ----------
    m_NS : float, or numpy.ndarray
    Mass(es) of the Neutron Star
    C_NS : float or numpy.ndarray
    Compactness(es) of the Neutron Star

    Returns
    -------
    Baryonic mass(es) corresponding to the inputs

    """
    # Function Body
    return m_NS * (1 + (0.6 * C_NS) / (1 - 0.5 * C_NS))


def compute_masses(m_BH, chi_BH, m_NS=1.4, lambda_NS=330):
    """Compute masses left outside the BH apparent radius, bound & unbound.

    The various masses are the remnant mass, m_out, which further consists of
    the dynamic mass, m_dyn, and the disc mass, m_disc. Thus,
    m_out = m_disc + m_out  (if the NS does not plunge into the BH)

    Parameters
    ----------
    m_BH : float or numpy.ndarray
    Mass(es) of the black hole(s)
    chi_BH : float or numpy.ndarray
    Spin(s) of the black hole(s)
    m_NS : float, optional. Default = 1.4 (M_SUN)
    Mass of the neutron star
    lambda_NS : float, optional. Default = 330
    Tidal deformability of the neutron star

    Returns
    -------
    m_out : float or numpy.ndarray
    Remnant mass (as defined in Foucart et al., 2018)
    m_dyn : float or numpy.ndarray
    Dynamical mass (as defined in Kawaguchi et al., 2016)
    m_disc : float or numpy.ndarray
    Disc mass (as defined in Barbieri et al., 2019)

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


@njit
def cart2sph(x, y, z):
    """Convert cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    x : float or numpy.ndarray
    y : float or numpy.ndarray
    z : float or numpy.ndarray

    Returns
    -------
    Spherical coordinates of the point (x, y, z)

    """
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    phi = np.arctan(y / x)
    return r, theta, phi


@njit
def sph2cart(r, theta, phi):
    """Convert spherical coordinates to cartesian coordinates.

    Parameters
    ----------
    r : float or numpy.ndarray
    The radial coordinate
    theta : float or numpy.ndarray
    The latitudinal coordinate, measured from the "north pole"
    phi : float or numpy.ndarray
    The longitudinal coordinate

    Returns
    -------
    Cartesian coordinates for the point (r, theta, phi)

    """
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    return x, y, z
