#!/usr/bin/env python
# Program to test the consistency of the various models used in the thesis.
# First from Foucart et al., 2018 and second from Kawaguchi et al., 2016.


# Imports and Auxiliary Functions

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import math_utils as mu

if __name__ == "__main__":

    plt.style.use(["fivethirtyeight", "seaborn-ticks"])

    # Checking for Model Consistency

    mass_NS = 1.4
    Lambda = 330  # corresponding to SFHo EoS
    mass_bh = np.linspace(2, 10, 1000)  # 2.1428 <= q<= 7.1428
    spin_bh = np.linspace(0, 0.9, 1000)  # Higher spins not discussed!

    mm, ss = np.meshgrid(mass_bh, spin_bh)
    m_out, m_dyn, m_disc = mu.compute_masses(mm, ss)

    # Plotting

    lvls = np.array([1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.2, 0.3, 0.4, 0.5])
    norm = LogNorm(vmin=lvls.min(), vmax=lvls.max())

    # m_dyn
    fig1, ax1 = plt.subplots()
    ax1.set_xlabel(r"$M_{BH}[M_\odot]$")
    ax1.set_ylabel(r"$\chi_{BH}$")
    ax1.set_title(r"$M_{dyn} [M_\odot]$")
    c1 = ax1.contourf(mm, ss, m_dyn, cmap="viridis", levels=lvls, norm=norm)
    cbar1 = plt.colorbar(c1)
    cbar1.set_ticks(lvls)
    cbar1.set_ticklabels(lvls)
    cbar1.ax.minorticks_off()

    # m_disc
    fig2, ax2 = plt.subplots()
    ax2.set_xlabel(r"$M_{BH}[M_\odot]$")
    ax2.set_ylabel(r"$\chi_{BH}$")
    ax2.set_title(r"$M_{disc} [M_\odot]$")
    c2 = ax2.contourf(mm, ss, m_disc, cmap="viridis", levels=lvls, norm=norm)
    cbar2 = plt.colorbar(c2)
    cbar2.set_ticks(lvls)
    cbar2.set_ticklabels(lvls)
    cbar2.ax.minorticks_off()

    # m_out
    fig3, ax3 = plt.subplots()
    ax3.set_xlabel(r"$M_{BH}[M_\odot]$")
    ax3.set_ylabel(r"$\chi_{BH}$")
    ax3.set_title(r"$M_{out} [M_\odot]$")
    c3 = ax3.contourf(mm, ss, m_out, cmap="viridis", levels=lvls, norm=norm)
    cbar3 = plt.colorbar(c3)
    cbar3.set_ticks(lvls)
    cbar3.set_ticklabels(lvls)
    cbar3.ax.minorticks_off()

    # ratio
    lvls2 = np.logspace(0, 3, 11)
    norm2 = LogNorm(vmin=lvls2.min(), vmax=lvls2.max())
    m_dyn_mask = np.ma.masked_where(m_dyn == 0, m_dyn)

    fig4, ax4 = plt.subplots()
    ax4.set_xlabel(r"$M_{BH}[M_\odot]$")
    ax4.set_ylabel(r"$\chi_{BH}$")
    ax4.set_title(r"Ratio of $M_{out}$ to $M_{dyn}$")
    c4 = ax4.contourf(
        mm, ss, m_out / m_dyn_mask, cmap="viridis", levels=lvls2, norm=norm2
    )
    cbar4 = plt.colorbar(c4)
    cbar4.set_ticks(lvls2)
    cbar4.set_ticklabels(lvls2)
    cbar4.ax.minorticks_off()

    plt.show()
