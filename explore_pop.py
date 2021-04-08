#!/usr/bin/env python
# Program to explore the properties of a population generated using create_pop.py
# %% Imports, Auxiliary Function Definitions and constants.

import json
import logging
import time

import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from nsbh_merger import M_SUN, MPC, PI
from snr_function_lalsim import optimal_snr
from model_consistency import compute_masses
from prompt_emission import calc_E_kin_jet, calc_onaxis

NWSNR_MIN = 10  # minimum network SNR
DETS = ["L1", "H1", "V1"]

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    start = time.time()

    logging.debug("Start of the Program")

    logging.debug("Read in HDF5 file")

    with h5py.File("population.hdf5", "r") as f:
        grp = f["data"]
        popln_params = grp["popln_parameters"]
        NUM_SAMPLES = popln_params.shape[1]
        detector_params = json.loads(popln_params.attrs["detector_params"])
        popln_array = np.array(popln_params)

        logging.debug("Done reading in HDF5 file")

    # %% Calculating the GW SNRs

    logging.debug("Compute the SNRs")

    try:
        snrs = np.load("snrs.npy")
        logging.debug("NOTE : Found precached SNRs file for this population!")
    except FileNotFoundError:
        print("Didn't find a precached SNR file for this population")
        print("Salvaging the situation: Computing SNRs...")
        snrs = np.zeros((NUM_SAMPLES, len(DETS)))
        for i in tqdm(range(NUM_SAMPLES)):
            snrs[i, :] = optimal_snr(
                m1det=popln_array[0, i] / M_SUN,
                m2det=popln_array[1, i] / M_SUN,
                S1z=popln_array[2, i],
                S2z=popln_array[3, i],
                Lambda1=popln_array[4, i],
                Lambda2=popln_array[5, i],
                theta=popln_array[6, i],
                phi=popln_array[7, i],
                DL=popln_array[8, i] / MPC,
                iota=popln_array[9, i],
                psi=popln_array[10, i],
            )
        np.save("snrs.npy", snrs)

    logging.debug("Done computing SNRs")

    logging.debug("Masking the SNRs")

    snr_111mask = (
        (np.sum(snrs ** 2, axis=1) > NWSNR_MIN ** 2)
        * (snrs[:, 0] > 4)
        * (snrs[:, 1] > 4)
        * (snrs[:, 2] > 4)
    )

    snr_110mask = (
        (np.sum(snrs ** 2, axis=1) > NWSNR_MIN ** 2)
        * (snrs[:, 0] > 4)
        * (snrs[:, 1] > 4)
        * (snrs[:, 2] < 4)
    )

    logging.debug("Plotting the Masked SNRs")

    fig, axs = plt.subplots(1, 3)
    for i in range(snrs.shape[1]):
        axs[i].grid(True)
        axs[i].set_axisbelow(True)
        axs[i].hist(
            snrs[snr_110mask][:, i],
            bins=50,
            density=True,
            label=f"{DETS[i]}",
        )
        axs[i].legend()

    # plt.show()

    logging.debug("Done plotting SNR Distributions")

    # %% Calculating the EM emission (prompt)

    logging.debug("Computing the disc masses")

    _, _, mass_disc = compute_masses(
        popln_array[0, :] / M_SUN,
        popln_array[2, :],
        popln_array[1, :] / M_SUN,
        popln_array[5, :],
    )

    logging.debug("Done computing disc masses")

    logging.debug("Computing E_iso(0)")

    E_iso = calc_onaxis(calc_E_kin_jet(mass_disc, popln_array[2, :])) * 1e7

    logging.debug("Done computing E_iso(0)")

    logging.debug("Computing fluences")

    fluence = E_iso / (4 * PI * popln_array[8, :] ** 2)

    logging.debug("Done computing fluences")

    logging.debug("Histogramming fluences above FERMI limit")

    fluence_masked = fluence[fluence > 2e-7]
    print(f"Obtained {fluence_masked.size} 'visible' events!")

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.hist(
        fluence_masked,
        bins=np.logspace(
            np.log10(fluence_masked.min()), np.log10(fluence_masked.max()), 50
        ),
        density=True,
        log=True,
    )

    plt.show()

    print(f"Took {time.time() - start}s...")
