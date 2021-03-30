#!/bin/python3
# Program to create a population of NS-BH binaries with various prior distributions.
# %% Imports, Auxiliary Function Definitions and constants.

import h5py
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import powerlaw

from nsbh_merger import M_SUN, G, C, PI, KPC

M_MIN = 2 * M_SUN  # lower cut-off for BH distribution considered
M_MAX = 10 * M_SUN  # upper cut-off for BH distribution considered
NUM_SAMP = 5000  # number of samples in population


def gen_samples(support, P_x, N=1000):
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

    dx = support[1] - support[0]  # spacing between the points, assumed to be uniform
    ecdf = np.cumsum(P_x) * dx  # compute empirical CDF

    if np.any(ecdf[1:] == ecdf[:-1]):
        # remove final data point (TODO: formalize this fix!)
        ecdf = ecdf[:-1]
        support = support[:-1]

    ecdf_inv_interp = interp1d(
        ecdf, support, kind="cubic"
    )  # fit a cubic spline to the inverse CDF

    samples = ecdf_inv_interp(np.random.random(N))  # generate samples
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
    lambda_ns = np.exp(-a_1 + np.sqrt(a_1 ** 2 - 4 * a_2 * (a_0 - c_ns)) / (2 * a_2))
    return lambda_ns


# Define path to the 'truncated' mass ppd file
ppd_file = "./o1o2o3_mass_b_iid_mag_two_comp_iid_tilt_powerlaw_redshift_mass_data.h5"

# Define path to the 'default' spin magnitude distribution file
spin_file = (
    "./o1o2o3_mass_c_iid_mag_two_comp_iid_tilt_powerlaw_redshift_magnitude_data.h5"
)

# Define dummy arrays
mass_ratio_dummy = np.linspace(0.1, 1, 500)  # for integration of 2D ppd
mass_1_dummy, dx = np.linspace(
    3, 100, 1000, retstep=True
)  # for interpolation of 1D ppd
spin_mags_dummy = np.linspace(0, 1, 1000)  # for normalization of spin dist.

with h5py.File(ppd_file, "r") as f:
    mass_ppd = f["ppd"]
    mass_1_ppd = np.trapz(mass_ppd, mass_ratio_dummy, axis=0)

with h5py.File(spin_file, "r") as f:
    ppd = f["ppd"]
    lines = f["lines"]
    num_lines = len(lines["a_1"])
    # Normalize to get probability distributions on a_1
    normalized_lines = np.array(
        [
            lines["a_1"][i, :] / np.trapz(lines["a_1"][i, :], spin_mags_dummy)
            for i in range(num_lines)
        ]
    )
    spin_probs_ppd = np.mean(normalized_lines, axis=0)

# %% Component masses

M_NS = 1.4 * M_SUN  # neutron star mass

# draw samples from the (primary) black hole mass distribution,
# and truncate to be within [M_MIN, M_MAX]
mass_bh = gen_samples(mass_1_dummy, mass_1_ppd, NUM_SAMP)
mask = np.logical_and(mass_bh <= M_MAX, mass_bh >= M_MIN)
mass_bh_trunc = mass_bh[mask]

# %% Component spins

CHI_NS = 0  # neutron star spin

# draw samples from the black hole spin magnitude distribution
chi_bh = gen_samples(spin_mags_dummy, spin_probs_ppd, NUM_SAMP)

# %% Component tidal deformabilities / compactness

LAMBDA_BH = 0  # black hole tidal deformability

R_NS = 11 * 1e5  # 11 km (in cgs)  # neutron star radius
C_NS = G * M_NS / (R_NS * C ** 2)  # neutron star compactness
LAMBDA_NS = love_c(C_NS)

# %% Sky location of the binary || Constant comoving volume distribution

cos_theta = 2 * np.random.random(NUM_SAMP) - 1
cos_iota = 2 * np.random.random(NUM_SAMP) - 1
psi = 2 * PI * np.random.random(NUM_SAMP)
phi = 2 * PI * np.random.random(NUM_SAMP)
# Luminosity Distance ~ const. in comoving volume uptil D_L ~ 800 Mpc
lum_dist = powerlaw.rvs(3, scale=800 * KPC * 1000, size=NUM_SAMP)

# %% Waveform model, noise curves and misc stuff

WF_MODEL = "IMRPhenomPv2"
IS_PSD = False
psd_files = {
    "H1": "./O3-H1-C01_CLEAN_SUB60HZ-1251752040.0_sensitivity_strain_asd.txt",
    "L1": "./O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt",
    "V1": "./O3-V1_sensitivity_strain_asd.txt",
}
