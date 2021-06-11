#!/usr/bin/env python
# Program to see where other known SGRBs fall in the Q-chi_BH plane.

from os.path import abspath

from astropy.cosmology import Planck13
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline
from prompt_emission import calc_E_kin_jet, calc_onaxis
from math_utils import compute_masses

if __name__ == "__main__":
    style.use(["fivethirtyeight", "seaborn-ticks"])
    DATA_PATH = abspath("../../../sgrb-stuff/sgrbs_z_confident.csv")
    df = pd.read_csv(DATA_PATH, skiprows=1)

    redshift = df["redshift"]
    fluence = df["fluence"].to_numpy()
    err_fluence = df["err_fluence"]
    lum_dist = Planck13.luminosity_distance(redshift).to("cm").value

    e_iso = fluence * (4 * np.pi * lum_dist ** 2)

    mass_bh = np.linspace(3, 20, 200)
    mass_ns = 1.4
    q = mass_bh / mass_ns
    chi_bh = np.linspace(0.0, 1.0, 200)
    qq, cc = np.meshgrid(q, chi_bh)

    _, _, m_disc = compute_masses(mass_bh, cc, mass_ns, lambda_NS=330)
    # disc_mask = m_disc > 0
    # ekin = calc_E_kin_jet(m_disc[disc_mask], cc[disc_mask])
    ekin = calc_E_kin_jet(m_disc, cc)
    log_eiso = np.log10(
        calc_onaxis(ekin)
    )  # should throw an error complaining about log(0.0)

    interpolant = RectBivariateSpline(q, chi_bh, log_eiso)

    fig, ax = plt.subplots(tight_layout=True)
    cp = ax.contour(
        qq, cc, log_eiso, levels=np.sort(np.log10(e_iso)), cmap="viridis", linewidths=1
    )
    # ax.clabel(cp, inline=True)
    ax.set_xlabel(r"Mass Ratio, $\mathcal{Q}$")
    ax.set_ylabel(r"Black Hole Spin, $\chi_{BH}$")
    # c = ax.colorbar(label=r"$\log_{10} E_{\mathrm{iso}}$ (in erg/cm$^2$)")

    ax.set_title(r"$E_{\mathrm{iso}}$ as a function of $\mathcal{Q}$ and $\chi_{BH}$")
    plt.show()
