#!/usr/bin/env python
# Program to test the consistency of the various models used in the thesis.
# First from Foucart et al., 2018 and second from Kawaguchi et al., 2016.


# Imports and Auxiliary Functions

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from matplotlib.colors import LogNorm

G = 6.67e-11  # Universal Gravitational Constant


@njit
def rcap_isco(chi_bh=0):
    """Calculate the normalized ISCO radius for a given BH spin."""
    # Function Body
    z1 = 1 + (1 - chi_bh ** 2) ** (1 / 3) * (
        (1 + chi_bh) ** (1 / 3) + (1 - chi_bh) ** (1 / 3)
    )
    z2 = np.sqrt(3 * chi_bh ** 2 + z1 ** 2)
    retval = 3 + z2 - np.sign(chi_bh) * np.sqrt((3 - z1) * (3 + z1 + 2 * z2))
    return retval


@njit
def c_love(lambda_NS=330):
    """Compute the compactness of a NS using the C-Love relation."""
    # Function Body
    a = np.array([0.36, -0.0355, 0.000705])
    C_NS = a.dot(np.array([1, np.log(lambda_NS), np.log(lambda_NS) ** 2]))
    return C_NS


@njit
def baryonic_mass(m_NS, C_NS):
    """Calculate the total baryonic mass."""
    # Function Body
    m_b = m_NS * (1 + (0.6 * C_NS) / (1 - 0.5 * C_NS))
    return m_b


def compute_masses(m_BH, chi_BH, m_NS=1.4, lambda_NS=330):
    """Compute masses left outside the BH apparent radius, bound & unbound.
    The various masses are the remnant mass, m_out, which further consists of
    the dynamic mass, m_dyn, and the disc mass, m_disc. Thus,
    m_out = m_disc + m_out  (if the NS does not plunge into the BH)
    """
    # Common binary parameters
    q = m_BH / m_NS
    eta = q / (1 + q) ** 2
    rho = (15 * lambda_NS) ** (-1 / 5)
    C_NS = c_love(lambda_NS)
    m_b = baryonic_mass(m_NS, C_NS)
    f = 0.3  # upper-limit to m_dyn, as a fraction of m_out

    # Compute m_out. Formula from ^Foucart et al., 2018
    alpha, beta, gamma, delta = 0.308, 0.124, 0.283, 1.536
    term_alpha = alpha * (1 - 2 * rho) / (eta ** (1 / 3))
    term_beta = -beta * rcap_isco(chi_BH) * rho / eta
    term_gamma = gamma
    m_out = m_b * (np.maximum(term_alpha + term_beta + term_gamma, 0.0)) ** delta

    # Compute m_dyn. Formula from ^Kawaguchi et al., 2016
    a1, a2, a3, a4, n1, n2 = 4.464e-2, 2.269e-3, 2.431, -0.4159, 0.2497, 1.352
    term_a1 = a1 * (q ** n1) * (1 - 2 * C_NS) / C_NS
    term_a2 = -a2 * (q ** n2) * (rcap_isco(chi_BH))
    term_a3 = a3 * (1 - m_NS / m_b)
    term_a4 = a4
    m_dyn = m_b * np.maximum(term_a1 + term_a2 + term_a3 + term_a4, 0)

    # Enforce upper limits on m_dyn.

    mask = m_dyn > f * m_out
    m_dyn[mask] = f * m_out[mask]

    # Compute m_disc
    m_disc = np.maximum(m_out - m_dyn, 0)

    return m_out, m_dyn, m_disc


# Checking for Model Consistency

mass_NS = 1.4
Lambda = 330  # corresponding to SFHo EoS
mass_bh = np.linspace(3, 10, 1000)  # 2.1428 <= q<= 7.1428
spin_bh = np.linspace(0, 1.0, 1000)  # Higher spins not discussed!

mm, ss = np.meshgrid(mass_bh, spin_bh)
m_out, m_dyn, m_disc = compute_masses(mm, ss)

# Plotting

lvls = np.array([1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1.0])
norm = LogNorm(vmin=lvls.min(), vmax=lvls.max())

# m_dyn
fig1, ax1 = plt.subplots()
ax1.set_xlabel(r"$M_{BH}[M_\odot]$", fontsize=14)
ax1.set_ylabel(r"$\chi_{BH}$", fontsize=14)
ax1.set_title(r"$M_{dyn} [M_\odot]$", fontsize=14)
c1 = ax1.contourf(mm, ss, m_dyn, levels=lvls, cmap="turbo", norm=norm)
cbar1 = plt.colorbar(c1)
cbar1.set_ticks(lvls)
cbar1.set_ticklabels(lvls)
cbar1.ax.minorticks_off()

# m_disc
fig2, ax2 = plt.subplots()
ax2.set_xlabel(r"$M_{BH}[M_\odot]$", fontsize=14)
ax2.set_ylabel(r"$\chi_{BH}$", fontsize=14)
ax2.set_title(r"$M_{disc} [M_\odot]$", fontsize=14)
c2 = ax2.contourf(mm, ss, m_disc, levels=lvls, cmap="turbo", norm=norm)
cbar2 = plt.colorbar(c2)
cbar2.set_ticks(lvls)
cbar2.set_ticklabels(lvls)
cbar2.ax.minorticks_off()

# m_out
fig3, ax3 = plt.subplots()
ax3.set_xlabel(r"$M_{BH}[M_\odot]$", fontsize=14)
ax3.set_ylabel(r"$\chi_{BH}$", fontsize=14)
ax3.set_title(r"$M_{out} [M_\odot]$", fontsize=14)
c3 = ax3.contourf(mm, ss, m_out, levels=lvls, cmap="turbo", norm=norm)
cbar3 = plt.colorbar(c3)
cbar3.set_ticks(lvls)
cbar3.set_ticklabels(lvls)
cbar3.ax.minorticks_off()

# ratio
lvls2 = np.logspace(0, 3, 11)
norm2 = LogNorm(vmin=lvls2.min(), vmax=lvls2.max())
m_dyn_mask = np.ma.masked_where(m_dyn == 0, m_dyn)

fig4, ax4 = plt.subplots()
ax4.set_xlabel(r"$M_{BH}[M_\odot]$", fontsize=14)
ax4.set_ylabel(r"$\chi_{BH}$", fontsize=14)
ax4.set_title(r"Ratio of $M_{out}$ to $M_{dyn}$", fontsize=14)
c4 = ax4.contourf(mm, ss, m_out / m_dyn_mask, levels=lvls2, cmap="turbo", norm=norm2)
cbar4 = plt.colorbar(c4)
cbar4.set_ticks(lvls2)
cbar4.set_ticklabels(lvls2)
cbar4.ax.minorticks_off()


plt.show()
