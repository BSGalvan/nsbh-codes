#!/usr/bin/env python
# Program to explore the properties of a population generated using create_pop.py
# %% Imports, Auxiliary Function Definitions and constants.

# import concurrent.futures
import json
import logging
import time

import h5py
import numpy as np
from tqdm import tqdm

from model_consistency import compute_masses
from math_utils import f_lso
from nsbh_merger import M_SUN, MPC, PI
from plot_utils import plot_snrs, plot_fluences, plot_thetav
from prompt_emission import do_gauss_cutoff_integral
from snr_function_lalsim import optimal_snr


NWSNR_MIN = 10  # minimum network SNR
F_MIN = 2e-7  # FERMI lower limit for fluence
GAUSS_CUT = PI / 3  # cutoff for structured Gaussian Jet
DETS = ["L1", "H1", "V1"]


# def optimal_snrp(population):
# """Wraps optimal_snr, for multithreaded computation.

# Parameters
# ----------
# population : NumPy array of the 1 population parameter sample

# Returns
# -------
# Optimal SNR as computed by optimal_snr()

# """
# snr = optimal_snr(
# m1det=population[0] / M_SUN,
# m2det=population[1] / M_SUN,
# S1z=population[2],
# S2z=population[3],
# Lambda1=population[4],
# Lambda2=population[5],
# theta=population[6],
# phi=population[7],
# DL=population[8] / MPC,
# iota=population[9],
# psi=population[10],
# )
# return snr


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.disable()

    start = time.time()

    logging.debug("Start of the Program")

    logging.debug("Read in HDF5 file")

    with h5py.File("population_unif_test.hdf5", "r") as f:
        grp = f["data"]
        popln_params = grp["popln_parameters"]
        NUM_SAMPLES = popln_params.shape[1]
        detector_params = json.loads(popln_params.attrs["detector_params"])
        popln_array = np.array(popln_params)

        logging.debug("Done reading in HDF5 file")

    mass_bh = popln_array[0, :] / M_SUN
    mass_ns = popln_array[1, :] / M_SUN
    chi_bh = popln_array[2, :]
    chi_ns = popln_array[3, :]
    lambda_bh = popln_array[4, :]
    lambda_ns = popln_array[5, :]
    theta = popln_array[6, :]  # in radians
    phi = popln_array[7, :]  # in radians
    lum_dist = popln_array[8, :] / MPC
    iota = popln_array[9, :]  # in radians
    psi = popln_array[10, :]  # in radians
    theta_v = np.minimum(iota, PI - iota)  # in radians

    # %% Calculating the GW SNRs

    logging.debug("Compute the SNRs")

    try:
        snrs = np.load("snrs_unif_test.npy")
        logging.debug("NOTE : Found precached SNRs file for this population!")
    except FileNotFoundError:
        print("Didn't find a precached SNR file for this population")
        print("Salvaging the situation: Computing SNRs...")
        snrs = np.zeros((NUM_SAMPLES, len(DETS)))
        # bar = tqdm(desc="SNR Computation", total=NUM_SAMPLES)

        # with concurrent.futures.ThreadPoolExecutor() as exe:
        # futures = [
        # exe.submit(optimal_snrp, popln_array[:, sample])
        # for sample in range(NUM_SAMPLES)
        # ]
        # for idx, future in enumerate(concurrent.futures.as_completed(futures)):
        # try:
        # snrs[idx, :] = future.result()
        # bar.update()
        # except Exception as err:
        # print(f"Sample #{idx} caused an exception: {err}")
        # # else:
        # # print(f"Sample #{idx} produced an SNR: {snrs[idx, :]}")

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
        np.save("snrs_unif_test.npy", snrs)

    logging.debug("Done computing SNRs")

    logging.debug("Plotting the Masked SNRs")

    plot_snrs(snrs, DETS, nwsnr_min=NWSNR_MIN)

    logging.debug("Done plotting SNR Distributions")

    # %% Calculating the EM emission (prompt)

    logging.debug("Computing the disc masses")

    _, _, mass_disc = compute_masses(mass_bh, chi_bh, mass_ns, lambda_ns)

    logging.debug("Done computing disc masses")

    logging.debug("Computing E_iso(theta_v)")

    E_iso = np.zeros(NUM_SAMPLES)

    for idx, (angle, disc, spin) in tqdm(
        enumerate(zip(theta_v, mass_disc, chi_bh)),
        desc="Calculating E_iso(theta_v)... ",
        total=NUM_SAMPLES,
    ):
        E_iso[idx] = do_gauss_cutoff_integral(angle, GAUSS_CUT, disc, spin)[0]

    logging.debug("Done computing E_iso(theta_v)")

    logging.debug("Computing fluences")

    fluence = E_iso / (4 * PI * (lum_dist * MPC) ** 2)

    logging.debug("Done computing fluences")

    logging.debug("Plotting fluences above FERMI limit")

    fluence_mask = fluence > F_MIN
    fluence_masked = fluence[fluence_mask]

    plot_fluences(fluence_masked, style="pdf")

    plot_fluences(fluence_masked)

    logging.debug("Peeking at the Viewing Angle Distribution")

    iota_masked = iota[fluence_mask]
    theta_v_masked = theta_v[fluence_mask]

    plot_thetav(theta_v_masked, style="pdf")

    plot_thetav(theta_v_masked)

    logging.debug("End of Program")

    print(f"Took {time.time() - start}s...")
