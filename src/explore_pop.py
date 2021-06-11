#!/usr/bin/env python
# Program to explore the properties of a population generated using create_pop.py
# %% Imports, Auxiliary Function Definitions and constants.

# import concurrent.futures
import json
import time
from os.path import abspath

import h5py
import numpy as np
from tqdm import tqdm

from math_utils import f_lso, compute_masses, c_love
from math_constants import M_SUN, MPC, PI
from plot_utils import plot_snrs, plot_fluences, plot_thetav
from prompt_emission import do_gauss_cutoff_integral
from snr_function_lalsim import optimal_snr

DATA_DIR = abspath("../data/")

BIN_DIR = f"{DATA_DIR}/bin"
BIN_FILE = "snrs_gauss0.7_test.npy"
BIN_PATH = f"{BIN_DIR}/{BIN_FILE}"

POP_DIR = f"{DATA_DIR}/test_populations"
POP_FILE = "population_gauss0.7_test.hdf5"
POP_PATH = f"{POP_DIR}/{POP_FILE}"

NWSNR_MIN = 10  # minimum network SNR
F_MIN = 2e-7  # FERMI lower limit for fluence
GAUSS_CUT = PI / 3  # cutoff for structured Gaussian Jet
DETS = ["L1", "H1", "V1"]


if __name__ == "__main__":

    start = time.time()

    print("Start of the Program")

    print("Read in HDF5 file")

    with h5py.File(POP_PATH, "r") as f:
        grp = f["data"]
        popln_params = grp["popln_parameters"]
        NUM_SAMPLES = popln_params.shape[1]
        detector_params = json.loads(popln_params.attrs["detector_params"])
        popln_array = np.array(popln_params)

        print("Done reading in HDF5 file")

    mass_bh = popln_array[0, :] / M_SUN
    mass_ns = popln_array[1, :] / M_SUN
    chi_bh = popln_array[2, :]
    chi_ns = popln_array[3, :]
    lambda_bh = popln_array[4, :]
    lambda_ns = popln_array[5, :]
    c_ns = c_love(lambda_ns)
    theta = popln_array[6, :]  # in radians
    phi = popln_array[7, :]  # in radians
    lum_dist = popln_array[8, :] / MPC
    iota = popln_array[9, :]  # in radians
    psi = popln_array[10, :]  # in radians
    theta_v = np.minimum(iota, PI - iota)  # in radians

    # %% Calculating the GW SNRs

    print("Compute the SNRs")

    try:
        snrs = np.load(BIN_PATH)
        print("NOTE : Found precached SNRs file for this population!")
    except FileNotFoundError:
        print("Didn't find a precached SNR file for this population")
        print("Salvaging the situation: Computing SNRs...")
        snrs = np.zeros((NUM_SAMPLES, len(DETS)))

        for idx in tqdm(range(NUM_SAMPLES), desc="Calculating SNRs..."):
            snrs[idx] = optimal_snr(
                m1det=mass_bh[idx],
                m2det=mass_ns[idx],
                DL=lum_dist[idx],
                theta=theta[idx],
                phi=phi[idx],
                psi=psi[idx],
                iota=iota[idx],
                S1z=chi_bh[idx],
                S2z=chi_ns[idx],
                f_min=20,
                f_max=f_lso(mass_bh[idx] + mass_ns[idx]),
                Lambda1=lambda_bh[idx],
                Lambda2=lambda_ns[idx],
            )
        np.save(BIN_PATH, snrs)

    print("Done computing SNRs")

    print("Plotting the Masked SNRs")

    plot_snrs(snrs, DETS, nwsnr_min=NWSNR_MIN)

    print("Done plotting SNR Distributions")

    # %% Calculating the EM emission (prompt)

    print("Computing the disc masses")

    mass_rem, _, mass_disc = compute_masses(mass_bh, chi_bh, mass_ns, lambda_ns)

    print("Done computing disc masses")

    print("Computing E_iso(theta_v)")

    E_iso = np.zeros(NUM_SAMPLES)

    # for idx, (angle, disc, spin) in tqdm(
    # enumerate(zip(theta_v, mass_disc, chi_bh)),
    # desc="Calculating E_iso(theta_v)... ",
    # total=NUM_SAMPLES,
    # ):
    # E_iso[idx] = do_gauss_cutoff_integral(angle, GAUSS_CUT, disc, spin)[0]

    for idx, (angle, disc, spin, bh_mass, ns_mass, compactness, rem_mass) in tqdm(
        enumerate(zip(theta_v, mass_disc, chi_bh, mass_bh, mass_ns, c_ns, mass_rem)),
        desc="Calculating E_iso(theta_v)... ",
        total=NUM_SAMPLES,
    ):
        E_iso[idx] = do_gauss_cutoff_integral(
            angle, GAUSS_CUT, disc, spin, bh_mass, ns_mass, compactness, rem_mass
        )[0]

    print("Done computing E_iso(theta_v)")

    print("Computing fluences")

    fluence = E_iso / (4 * PI * (lum_dist * MPC) ** 2)

    print("Done computing fluences")

    print("Plotting fluences above FERMI limit")

    fluence_mask = fluence > F_MIN
    fluence_masked = fluence[fluence_mask]

    plot_fluences(fluence_masked, style="pdf")

    plot_fluences(fluence_masked)

    print("Peeking at the Viewing Angle Distribution")

    iota_masked = iota[fluence_mask]
    theta_v_masked = theta_v[fluence_mask]

    plot_thetav(theta_v_masked, style="pdf")

    plot_thetav(theta_v_masked)

    print("End of Program")

    print(f"Took {time.time() - start}s...")
