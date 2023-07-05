#!/usr/bin/env python
# Program to see where other known SGRBs fall in the Q-chi_BH plane.
# Original Author: B.S. Bharath Saiguhan, github.com/bsgalvan

from os.path import abspath

from astropy.cosmology import Planck13
import matplotlib.pyplot as plt
# from matplotlib import style
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
# from scipy.interpolate import RectBivariateSpline

from prompt_emission import do_gauss_cutoff_integral
from math_utils import compute_masses, c_love

if __name__ == "__main__":
    sns.set_theme(context="poster", style="ticks", font="sans-serif", font_scale=1.15)
    # style.use(["fivethirtyeight", "seaborn-ticks"])
    DATA_PATH = abspath("../../../sgrb-stuff/sgrbs_z_confident.csv")
    df = pd.read_csv(DATA_PATH, skiprows=1)

    sgrb_names = df["SGRB"]
    redshift = df["redshift"]
    fluence = df["fluence"].to_numpy()
    err_fluence = df["err_fluence"]
    lum_dist = Planck13.luminosity_distance(redshift.to_numpy()).to("cm").value

    e_iso = fluence * (4 * np.pi * lum_dist**2)

    N = 200
    mass_bh = np.linspace(1, 15, N)
    mass_ns = 1
    q = mass_bh / mass_ns
    chi_bh = np.linspace(0, 0.99999, N)
    qq, cc = np.meshgrid(q, chi_bh)

    _, _, m_disc = compute_masses(mass_bh, cc, mass_ns, lambda_NS=330)
    # disc_mask = m_disc > 0
    c_ns = c_love(330)
    eiso = np.asarray(
        [
            do_gauss_cutoff_integral(
                np.radians(7.5),
                np.pi / 2,
                mdisc,
                c,
                q * mass_ns,
                mass_ns,
                c_ns,
            )[0]
            for mdisc, c, q in tqdm(
                zip(m_disc.ravel(), cc.ravel(), qq.ravel()),
                total=m_disc.size,
            )
        ]
    )
    # ekin = calc_E_kin_jet(m_disc, cc, mass_ns)
    # log_eiso = np.log10(eiso).reshape(
    #     N, N
    # )  # should throw an error complaining about log(0.0)

    # interpolant = RectBivariateSpline(q, chi_bh, log_eiso)

    choices = [1, 12, 16, 23, 36, 42, 45]
    locs = [(2, 0.4), (2, 0.5), (2.5, 0.55), (2, 0.65), (2, 0.7), (2.5, 0.8), (4.1, 0.9)]
    fig, ax = plt.subplots(tight_layout=True)
    for idx, choice in enumerate(choices):
        cp = ax.contour(
            qq,
            cc,
            np.reshape(eiso / (4 * np.pi * lum_dist[choice] ** 2), (N, N)),
            levels=[fluence[choice]],
            colors=f"C{idx}",
        )
        # ax.clabel(
        #     cp,
        #     inline=True,
        #     fmt=f"GRB{sgrb_names[choice]}(z={redshift[choice]:.2f})",
        #     manual=[(locs[idx])],
        #     fontsize=10,
        # )

    ax.set_xlabel(r"Mass Ratio, $\mathcal{Q}$")
    ax.set_ylabel(r"Black Hole Spin, $\chi_{BH}$")
    # c = ax.colorbar(label=r"$\log_{10} E_{\mathrm{iso}}$ (in erg/cm$^2$)")

    ax.set_title(
        r"$\mathcal{F}(\theta_v = 1.5 \cdot \theta_{c, E})$ Level Sets as a function of $\mathcal{Q}$ and $\chi_{BH}$"
    )
    plt.show()
