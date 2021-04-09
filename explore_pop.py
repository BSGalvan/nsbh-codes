#!/usr/bin/env python
# Program to explore the properties of a population generated using create_pop.py
# %% Imports, Auxiliary Function Definitions and constants.

import json
import logging
import time

import h5py
import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
from tqdm import tqdm

from nsbh_merger import M_SUN, MPC, PI
from snr_function_lalsim import optimal_snr
from model_consistency import compute_masses
from prompt_emission import calc_E_kin_jet, calc_onaxis

NWSNR_MIN = 10  # minimum network SNR
DETS = ["L1", "H1", "V1"]

style.use(["fivethirtyeight", "seaborn-ticks"])


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


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.disable()

    start = time.time()

    logging.debug("Start of the Program")

    logging.debug("Read in HDF5 file")

    with h5py.File("population_100k.hdf5", "r") as f:
        grp = f["data"]
        popln_params = grp["popln_parameters"]
        NUM_SAMPLES = popln_params.shape[1]
        detector_params = json.loads(popln_params.attrs["detector_params"])
        popln_array = np.array(popln_params)

        logging.debug("Done reading in HDF5 file")

    # %% Calculating the GW SNRs

    logging.debug("Compute the SNRs")

    try:
        snrs = np.load("snrs_100k.npy")
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
        np.save("snrs_100k.npy", snrs)

    logging.debug("Done computing SNRs")

    logging.debug("Masking the SNRs")

    snr_mask = np.sum(snrs ** 2, axis=1) > NWSNR_MIN ** 2

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

    fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
    for i in range(snrs.shape[1]):
        axs[i].grid(True)
        axs[i].set_axisbelow(True)
        xs, ys = ecdf(snrs[snr_mask][:, i])
        median_x = np.median(xs)
        axs[i].step(xs, ys)
        # axs[i].set_title(f"{DETS[i]}")
        axs[i].set_xscale("log")
        axs[i].plot(median_x, 0.5, 'o')
        axs[i].annotate(
            f"({np.round(median_x, 2)}, 0.5)",
            xy=(median_x, 0.5),
            xytext=(median_x * 0.35, 0.6),
            textcoords="data",
            arrowprops=dict(color="#000000", arrowstyle="->", connectionstyle="angle3"),
            size=15,
            horizontalalignment="center",
            verticalalignment="bottom",
        )
        axs[i].set_xlabel(fr"SNR at {DETS[i]}, $\rho_{{{DETS[i]}}} $")
        axs[i].set_ylabel(fr"$\tilde{{F}}(\rho_{{{DETS[i]}}} | \rho_{{NW}} > 10)$")
        # axs[i].legend()

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

    logging.debug("Plotting fluences above FERMI limit")

    fluence_masked = fluence[fluence > 2e-7]
    print(f"Obtained {fluence_masked.size} 'visible' events!")

    fig, ax = plt.subplots(tight_layout=True)
    ax.set_xscale("log")
    fxs, fys = ecdf(fluence_masked)
    median_f = np.median(fxs)
    ax.step(fxs, fys)
    ax.plot(median_f, 0.5, 'o')
    ax.annotate(
        f"({median_f:.3E}, 0.5)",
        xy=(median_f, 0.5),
        xytext=(median_f * 0.35, 0.6),
        textcoords="data",
        arrowprops=dict(color="#000000", arrowstyle="->", connectionstyle="angle3"),
        size=15,
        horizontalalignment="center",
        verticalalignment="bottom",
    )
    ax.set_xlabel(r"Fluence, $\mathcal{F}_\gamma$ (erg/cm$^2$)")
    ax.set_ylabel(
        r"$\tilde{F}(\mathcal{F}_\gamma|\mathcal{F}_\gamma>\mathcal{F}_{min.})$"
    )
    ax.set_title("ECDF of Fluence above FERMI limit")

    logging.debug("Peeking at the Viewing Angle Distribution")

    theta_v = np.minimum(popln_array[9, :], PI - popln_array[9, :])
    tvxs, tvys = ecdf(theta_v)
    median_thetav = np.median(tvxs)
    fig, ax = plt.subplots(tight_layout=True)
    ax.step(np.degrees(tvxs), tvys)
    ax.set_xlabel(r"Viewing Angle, $\theta_v$ (deg.)")
    ax.set_ylabel(r"$\tilde{\mathcal{F}}(\theta_v)$")
    ax.plot(np.degrees(median_thetav), 0.5, "o")
    ax.annotate(
        f"({np.round(np.degrees(median_thetav), 2)}, 0.5)",
        xy=(np.degrees(median_thetav), 0.5),
        xytext=(np.degrees(median_thetav) * 0.35, 0.6),
        textcoords="data",
        arrowprops=dict(color="#000000", arrowstyle="->", connectionstyle="angle3"),
        size=15,
        horizontalalignment="center",
        verticalalignment="bottom",
    )
    ax.set_title("ECDF of the viewing angle of 'visible' NSBH Mergers")

    plt.show()
    print(f"Took {time.time() - start}s...")
