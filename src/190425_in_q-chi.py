#!/usr/bin/env python
# Program to compute the 'posterior' for the disc mass, using other posteriors
# for other quantities as given in GWTC-2 for GW190425


import h5py
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from tqdm import tqdm

import math_utils as mu
from math_constants import MPC
from prompt_emission import do_gauss_cutoff_integral

if __name__ == "__main__":
    style.use(["fivethirtyeight", "seaborn-ticks"])

    DATA_PATH = "/home/bharath/Desktop/a-tale-of-two-dead-stars/codes/data_store"
    FILE_NAME = "GW190425.h5"
    FILE_PATH = f"{DATA_PATH}/{FILE_NAME}"

    with h5py.File(FILE_PATH, "r") as f:
        grp = f["PhenomPNRT-HS"]
        posteriors = grp["posterior_samples"]
        arr = np.array(posteriors)

    m1 = arr["mass_1"]
    m2 = arr["mass_2"]

    if np.median(m1) > np.median(m2):
        mass_bh, mass_ns = m1, m2
        spin_bh = arr["a_1"]
        lambda_ns = arr["lambda_1"]
    else:
        mass_bh, mass_ns = m2, m1
        spin_bh = arr["a_2"]
        lambda_ns = arr["lambda_2"]

    mass_ratio = mass_bh / mass_ns
    mass_rem, _, mass_disc = mu.compute_masses(mass_bh, spin_bh, mass_ns, lambda_ns)

    disc_mask = mass_disc > 0

    iota = arr["iota"]
    theta_v = np.minimum(iota, np.pi - iota)
    lum_dist = arr["luminosity_distance"]

    CUTOFF_ANGLE = np.pi / 3

    E_iso = np.zeros(disc_mask.sum())

    for idx, (angle, disc, spin, bh_mass, ns_mass, compactness, rem_mass) in tqdm(
        enumerate(
            zip(
                theta_v[disc_mask],
                mass_disc[disc_mask],
                spin_bh[disc_mask],
                mass_bh[disc_mask],
                mass_ns[disc_mask],
                mu.c_love(lambda_ns[disc_mask]),
                mass_rem[disc_mask],
            )
        ),
        total=disc_mask.sum(),
    ):
        E_iso[idx] = do_gauss_cutoff_integral(
            angle, CUTOFF_ANGLE, disc, spin, bh_mass, ns_mass, compactness, rem_mass
        )[0]

    fluence = E_iso / (4 * np.pi * (lum_dist[disc_mask] * MPC) ** 2)

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
    plt.title(r"GW190425 in the $\mathcal{Q}-\chi_{BH}$; High Spin Posterior")
    plt.tight_layout()
    plt.show()
