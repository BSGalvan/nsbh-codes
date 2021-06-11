#!/usr/bin/env python
# Program to create a population of NS-BH binaries with various prior distributions.
# %% Imports, Auxiliary Function Definitions and constants.

import json
import time
from os.path import abspath

import h5py
import numpy as np

from scipy.stats import powerlaw

from math_utils import gen_samples, love_c
from math_constants import M_SUN, G, C, PI, MPC

from d_max import d_max

start = time.time()

print("Start of the Program")

NUM_SAMP = 100000  # number of samples in population
M_MIN, M_MAX = 3.0, 20.0  # lower, upper cut-off for BH distribution considered
SPIN_MU, SPIN_SIGMA = 0.7, 0.2  # change these if you want different gaussians
D_MAX = (
    d_max(100.0, 2000.0, 1000) * MPC  # ~ 1124 Mpc
)  # maximal distance for q = 20:1.4, NWSNR = 10 NSBH merger

DATA_DIR = abspath("../data/")
POST_DIR = f"{DATA_DIR}/gwtc2_posteriors"
POP_DIR = f"{DATA_DIR}/test_populations"
NOISE_DIR = f"{DATA_DIR}/sensitivity_curves"

# Define path to the 'truncated' mass ppd file
MASS_FILE = "o1o2o3_mass_b_iid_mag_two_comp_iid_tilt_powerlaw_redshift_mass_data.h5"
MASS_PATH = f"{POST_DIR}/{MASS_FILE}"

# Define path to the 'default' spin magnitude distribution file
SPIN_FILE = (
    "o1o2o3_mass_c_iid_mag_two_comp_iid_tilt_powerlaw_redshift_magnitude_data.h5"
)
SPIN_PATH = f"{POST_DIR}/{SPIN_FILE}"

# Define dummy arrays for integrations
# See https://dcc.ligo.org/public/0171/P2000434/002/Produce-Figures.ipynb

print("Extracting PPDs")

mass_ratio_dummy = np.linspace(0.1, 1, 500)  # for integration of 2D mass-q ppd
mass_1_dummy = np.linspace(3, 100, 1000)  # for interpolation of 1D ppd

with h5py.File(MASS_PATH, "r") as f:
    mass_ppd = f["ppd"]
    mass_1_ppd = np.trapz(mass_ppd, mass_ratio_dummy, axis=0)

print("Finished extracting mass PPD")

spin_mags_dummy = np.linspace(0, 1, 1000)  # for normalization of spin dist.
with h5py.File(SPIN_PATH, "r") as f:
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

print("Finished extracting spin PPD")

print("Setting component masses")

# %% Component masses

M_NS = 1.4 * M_SUN  # neutron star mass

# draw samples from the (primary) black hole mass distribution, guaranteed to be
# within [M_MIN, M_MAX]
mass_bh = (
    gen_samples(mass_1_dummy, mass_1_ppd, N=NUM_SAMP, low=M_MIN, high=M_MAX) * M_SUN
)

print("Finished setting component masses")

print("Setting component spins")

# %% Component spins

CHI_NS = 0  # neutron star spin

# draw samples from the black hole spin magnitude distribution
chi_bh = gen_samples(spin_mags_dummy, spin_probs_ppd, NUM_SAMP)

print("Finished setting component spins")

print("Setting component Lambdas")

# %% Component tidal deformabilities / compactness

LAMBDA_BH = 0  # black hole tidal deformability

R_NS = 11 * 1e5  # 11 km (in cgs); neutron star radius
C_NS = G * M_NS / (R_NS * C ** 2)  # neutron star compactness
LAMBDA_NS = love_c(C_NS)

print("Finished setting component Lambdas")

print("Setting binary 3D sky locations")

# %% Sky location of the binary || Constant comoving volume distribution

theta = np.arccos(2 * np.random.random(NUM_SAMP) - 1)
iota = np.arccos(2 * np.random.random(NUM_SAMP) - 1)
psi = 2 * PI * np.random.random(NUM_SAMP)
phi = 2 * PI * np.random.random(NUM_SAMP)

# Luminosity Distance ~ const. in comoving volume until D_L ~ D_MAX Mpc
lum_dist = powerlaw.rvs(3, scale=D_MAX, size=NUM_SAMP)

print("Finished setting binary 3D sky locations")

print("Setting detector parameters")

# %% Waveform model, noise curves and misc stuff

WF_MODEL = "IMRPhenomPv2"
IS_PSD = True
psd_files = {
    "H1": "data/aLIGO_ZERO_DET_high_P.txt",
    "L1": "data/aLIGO_ZERO_DET_high_P.txt",
    "V1": "data/aLIGO_ZERO_DET_high_P.txt",
}


print("Finished setting detector parameters")

print("Packing data --> HDF5 file")

# %% Pack all the related variables together

popln_params = np.stack(
    (
        mass_bh,  # masses of black holes (g) [0]
        M_NS * np.ones_like(mass_bh),  # masses of neutron stars (g) [1]
        chi_bh,  # spins of the black holes (dim_less) [2]
        CHI_NS * np.ones_like(chi_bh),  # spins of neutron stars (dim_less) [3]
        LAMBDA_BH
        * np.ones_like(mass_bh),  # tidal deformabilities of black holes (dim_less) [4]
        LAMBDA_NS
        * np.ones_like(
            mass_bh
        ),  # tidal deformabilities of neutron stars (dim_less) [5]
        theta,  # RA of the binary on the sky (rad) [6]
        phi,  # DEC of the binary on the sky (rad) [7]
        lum_dist,  # luminosity distance to binary (cm) [8]
        iota,  # inclinations of binaries wrt LOS (rad) [9]
        psi,  # polarization angle of binary (rad) [10]
    )
)

# Detector metadata --> dict --> JSON --> attribute in final HDF5 file
detector_params = json.dumps(
    dict(wf_model=WF_MODEL, is_psd=IS_PSD, psd_paths=psd_files), indent=4
)


# %% Create the HDF5 File

FILE_NAME = f"population_{NUM_SAMP}_test.hdf5"

with h5py.File(f"{POP_DIR}/{FILE_NAME}", "a") as f:
    dset = f.create_dataset(
        "/data/popln_parameters",
        shape=popln_params.shape,
        dtype=popln_params.dtype,
        data=popln_params,
    )
    dset.attrs["detector_params"] = detector_params

print(f"Finished making {FILE_NAME}")

print("End of Program")
print(f"Took {time.time() - start}s...")
