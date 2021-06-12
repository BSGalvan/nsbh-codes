#!/usr/bin/env python
# Program to compute the variation of on-axis fluence, for NSBH mergers at 100 Mpc,
# with the mass-ratio (q) and spin of the black hole (chi_bh).
# Original Author: B.S. Bharath Saiguhan, github.com/bsgalvan

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

from prompt_emission import calc_E_kin_jet, calc_onaxis
from math_utils import compute_masses

if __name__ == "__main__":
    style.use(["fivethirtyeight", "seaborn-ticks"])

    mass_bh = np.linspace(1.4, 20, 200)
    mass_ns = 1.4
    q = mass_bh / mass_ns
    chi_bh = np.linspace(-0.9, 0.9, 200)
    qq, cc = np.meshgrid(q, chi_bh)

    _, _, m_disc = compute_masses(mass_bh, cc, mass_ns, lambda_NS=330)
    disc_mask = m_disc > 0
    log_disc = np.log10(m_disc[disc_mask])
    ekin = calc_E_kin_jet(m_disc[disc_mask], cc[disc_mask])
    log_eiso = np.log10(calc_onaxis(ekin))

    plt.scatter(
        qq[disc_mask],
        cc[disc_mask],
        c=log_eiso,
        s=5,
        cmap="magma",
    )
    plt.xlabel(r"Mass Ratio, $\mathcal{Q}$")
    plt.ylabel(r"Black Hole Spin, $\chi_{BH}$")
    c = plt.colorbar(label=r"$\log_{10} E_{\mathrm{iso}}$ (in erg/cm$^2$)")
    plt.tight_layout()
    plt.title(r"$E_{\mathrm{iso}}$ as a function of $\mathcal{Q}$ and $\chi_{BH}$")
    plt.show()
