#!/bin/python3
# Program to create a population of NS-BH binaries with various prior distributions.
# %% Imports, Auxiliary Function Definitions and constants.

import h5py
import numpy as np
# from scipy.interpolate import interp1d

from nsbh_merger import M_SUN

M_MIN = 2 * M_SUN  # lower cut-off for BH distribution considered
M_MAX = 10 * M_SUN  # upper cut-off for BH distribution considered
NUM_SAMP = 5e3  # number of samples in population


# Define path to the 'truncated' mass ppd file
filename = "o1o2o3_mass_b_iid_mag_two_comp_iid_tilt_powerlaw_redshift_mass_data.h5"

# Define dummy arrays
mass_ratio_dummy = np.linspace(0.1, 1, 500)  # for integration of 2D ppd
mass_1_dummy = np.linspace(3, 100, 1000)  # for interpolation of 1D ppd

with h5py.File(filename, "r") as f:
    mass_ppd = f["ppd"]
    mass_1_ppd = np.trapz(mass_ppd, mass_ratio_dummy, axis=0)

# %% Component masses

M_NS = 1.4 * M_SUN  # neutron star mass

# draw samples from the primary mass distribution
# mass_bh = gen_samples(mass_1_dummy, mass_1_ppd, NUM_SAMP)
