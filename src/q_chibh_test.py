#!/usr/bin/env python
# Program to compute the variation of on-axis fluence, for NSBH mergers at 100 Mpc,
# with the mass-ratio (q) and spin of the black hole (chi_bh).

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

    # For GW190426_152155
    # m1_med, m1_min, m1_max = 5.7, 3.4, 9.4
    # m2_med, m2_min, m2_max = 1.5, 1.0, 2.3
    # q_med, q_min, q_max = m1_med / m2_med, m1_min / m2_max, m1_max / m2_min
    # chi_med, chi_min, chi_max = (
    # -0.03,
    # -0.33,
    # 0.3,
    # )
    # This is calculated from chi_eff, not chi! Fix by taking the posteriors of chi_eff
    # and iota_tilt, and thus computing chi_bh.

    # _, _, mdisc_min = compute_masses(m1_max, chi_min, m2_min)
    # _, _, mdisc_max = compute_masses(m1_min, chi_max, m2_max)
    # _, _, mdisc_med = compute_masses(m1_med, chi_med, m2_med)

    plt.scatter(
        qq[disc_mask], cc[disc_mask], c=log_eiso, s=5, cmap="magma",
    )
    plt.xlabel(r"Mass Ratio, $\mathcal{Q}$")
    plt.ylabel(r"Black Hole Spin, $\chi_{BH}$")
    c = plt.colorbar(label=r"$\log_{10} E_{\mathrm{iso}}$ (in erg/cm$^2$)")
    # plt.errorbar(
    # q_med,
    # chi_med,
    # xerr=np.array([q_med - q_min, q_max - q_med]).reshape(2, -1),
    # yerr=np.array([chi_med - chi_min, chi_max - chi_med]).reshape(2, -1),
    # mec="k",
    # marker="*",
    # ms=15,
    # mfc="w",
    # )
    # plt.plot(q_min, chi_max, mec="k", marker="*", ms=15, mfc="yellow")
    plt.tight_layout()
    plt.title(r"$E_{\mathrm{iso}}$ as a function of $\mathcal{Q}$ and $\chi_{BH}$")
    plt.show()
