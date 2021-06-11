#!/usr/bin/env python
# Program to check if the prompt emission code
# correctly computes the profile given a population

from os.path import abspath
from time import sleep

import h5py
import numpy as np
from scipy.integrate import nquad
from tqdm import tqdm

PI = np.pi
GAUSS_CUT = PI / 3
NUM_SAMPLES = int(1e5)
theta_E = 0.1  # core angle for the energy profile function, in radians
theta_gamma = 0.2  # core angle for the Lorentz factor function, in radians
gamma_0 = 100  # on-axis gamma
eta = 0.1  # kinetic energy --> gamma ray efficiency
EK = 1e49

DATA_DIR = abspath("../data/")

POP_DIR = f"{DATA_DIR}/test_populations"
POP_FILE = "population_test.hdf5"
POP_PATH = f"{POP_DIR}/{POP_FILE}"


def gaussian(theta, phi, theta_v):
    """Return function value at (theta, phi), corresponding to Gaussian jet."""
    print("==================================================================")
    print(f"theta={np.degrees(theta)}")
    print(f"phi={np.degrees(phi)}")
    print(f"theta_v={np.degrees(theta_v)}")
    gamma = 1 + (gamma_0 - 1) * np.exp(-((theta / theta_gamma) ** 2))
    print(f"gamma={gamma}")
    beta = np.sqrt(1 - 1 / gamma ** 2)
    print(f"beta={beta}")
    E_c = EK / (PI * theta_E ** 2)
    print(f"E_c={E_c}")
    dE_dOmega = E_c * np.exp(-((theta / theta_E) ** 2))
    print(f"dE_dOmega={dE_dOmega}")
    cos_alpha = np.cos(theta_v) * np.cos(theta) + np.sin(theta_v) * np.sin(
        theta
    ) * np.cos(phi)
    print(f"cos_alpha={cos_alpha}")
    retval = np.sin(theta) * dE_dOmega / ((gamma ** 4)) * ((1 - beta * cos_alpha) ** 3)
    print(f"retval={retval}")
    print("==================================================================")
    # sleep(1)
    return eta * retval


def do_gauss_cutoff_integral(theta_v, cutoff_angle):
    """Return the value after integrating over a gaussian jet with a cutoff."""
    ans, err, out_dict = nquad(
        gaussian,
        ranges=[[0, cutoff_angle], [0, 2 * PI]],
        args=(theta_v,),
        full_output=True,
    )
    # print("Got gamma_0 of ", gamma_0)
    return ans, err, out_dict


if __name__ == "__main__":
    E_iso = np.zeros(NUM_SAMPLES)
    with h5py.File(POP_PATH, "r") as f:
        grp = f["data"]
        popln_params = grp["popln_parameters"]
        popln_array = np.array(popln_params)

        print("Done reading in HDF5 file")

    iota = popln_array[9, :]  # in radians
    theta_v = np.minimum(iota, PI - iota)  # in radians

    for idx, angle in tqdm(
        enumerate(theta_v[:10]), desc="Calculating E_iso(theta_v)...", total=10,
    ):
        E_iso[idx] = do_gauss_cutoff_integral(angle, GAUSS_CUT)[0]
