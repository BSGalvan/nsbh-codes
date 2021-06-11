#!/usr/bin/env python
# Program to compute the Gamma-ray fluence of a prototypical 5-1.4 M_sun binary,
# in the q-chi_bh plane.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

from math_utils import compute_masses
from math_constants import PI, MPC
from prompt_emission import calc_E_kin_jet

if __name__ == "__main__":

    style.use(["fivethirtyeight", "seaborn-ticks"])

    # Constants of the problem
    mass_bh = 10  # in Msun
    mass_ns = 1.4  # in Msun
    lambda_ns = 330
    lum_dist = 200  # in Mpc
    THETA_E = 0.1  # scale factor for jet energy profile, in radians

    # Arrays to compute over
    chi_bh = np.linspace(0, 1, 100)
    theta_v = np.linspace(0, np.pi / 2, 100)
    ss, tt = np.meshgrid(chi_bh, theta_v)

    # Compute disc mass --> jet KE --> E_iso(theta_v) --> fluence
    _, _, mass_disc = compute_masses(mass_bh, ss, mass_ns, lambda_ns)
    Ekin_jet = calc_E_kin_jet(mass_disc, ss)
    Eiso = 4 * PI * Ekin_jet / (PI * THETA_E ** 2) * np.exp(-((tt / THETA_E) ** 2))
    fluence = np.log10(Eiso / (4 * PI * (lum_dist * MPC) ** 2))

    # Plot stuff
    fig, ax = plt.subplots()
    pos = ax.imshow(
        fluence.T,
        origin="lower",
        cmap="inferno",
        interpolation="none",
        vmin=np.log10(2e-7),
        vmax=fluence.max(),
        extent=[
            np.degrees(theta_v.min()),
            np.degrees(theta_v.max()),
            chi_bh.min(),
            chi_bh.max(),
        ],
        aspect="auto",
    )
    fig.colorbar(
        pos, ax=ax, extend="min", label=r"Fluence, $\log_{10}(\mathcal{F_{\gamma}})$"
    )
    ax.set_ylabel(r"Black Hole Spin, $\chi_{BH}$")
    ax.set_xlabel(r"Viewing Angle, $\theta_v$")
    ax.set_title(r"Variation of Fluence with $\chi_{BH}$ \& $\theta_v$")
    fig.tight_layout()
    plt.show()
