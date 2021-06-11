#!/usr/bin/env python
# Program to compute the 'posterior' for the disc mass, using other posteriors
# for other quantities as given in GWTC-1 for GW170817


import h5py
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from tqdm import tqdm

import math_utils as mu
from prompt_emission import do_gauss_cutoff_integral


if __name__ == "__main__":
    style.use(["fivethirtyeight", "seaborn-ticks"])

    DATA_PATH = "/home/bharath/Desktop/a-tale-of-two-dead-stars/codes/data_store"
    FILE_NAME = "GW170817_GWTC-1.hdf5"
    FILE_PATH = f"{DATA_PATH}/{FILE_NAME}"

    with h5py.File(FILE_PATH, "r") as f:
        dset = f["IMRPhenomPv2NRT_highSpin_posterior"]
        arr = np.array(dset)

    m1 = arr["m1_detector_frame_Msun"]
    m2 = arr["m2_detector_frame_Msun"]

    if np.median(m1) > np.median(m2):
        mass_bh, mass_ns = m1, m2
        spin_bh = arr["spin1"]
        lambda_ns = arr["lambda1"]
    else:
        mass_bh, mass_ns = m2, m1
        spin_bh = arr["spin2"]
        lambda_ns = arr["lambda2"]

    mass_ratio = mass_bh / mass_ns
    mass_rem, _, mass_disc = mu.compute_masses(mass_bh, spin_bh, mass_ns, lambda_ns)

    disc_mask = mass_disc > 0

    THETA_V = np.radians(20)
    CUTOFF_ANGLE = np.pi / 3

    E_iso = np.zeros(disc_mask.sum())

    for idx, (disc, spin, bh_mass, ns_mass, compactness, rem_mass) in tqdm(
        enumerate(
            zip(
                mass_disc[disc_mask],
                spin_bh[disc_mask],
                mass_bh[disc_mask],
                mass_ns[disc_mask],
                mu.c_love(lambda_ns[disc_mask]),
                mass_rem[disc_mask],
            ),
        ),
        total=disc_mask.sum(),
    ):
        E_iso[idx] = do_gauss_cutoff_integral(
            THETA_V, CUTOFF_ANGLE, disc, spin, bh_mass, ns_mass, compactness, rem_mass,
        )[0]

    plt.figure()
    plt.scatter(
        mass_ratio[disc_mask],
        spin_bh[disc_mask],
        c=np.log10(mass_disc[disc_mask]),
        cmap="viridis",
    )
    plt.colorbar(label=r"$\log_{10}M_{disc}$ [M$_\odot$]")
    plt.scatter(
        mass_ratio[~disc_mask], spin_bh[~disc_mask], c="white", s=12, edgecolors="k"
    )
    plt.xlabel(r"Mass Ratio, $\mathcal{Q}$")
    plt.ylabel(r"Black Hole Spin, $\chi_{BH}$")
    plt.title(r"GW170817 in the $\mathcal{Q}-\chi_{BH}$; High Spin Posterior")
    plt.tight_layout()
    plt.show()
