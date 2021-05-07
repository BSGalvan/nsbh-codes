#!/usr/bin/env python
# Program to compute the prompt emission, given a disc mass M_disc

from numba import jit
import numpy as np
from nsbh_merger import M_SUN, PI, C
import matplotlib.pyplot as plt
from scipy.integrate import nquad
from tqdm import tqdm

theta_E = 0.1  # core angle for the energy profile function, in radians
theta_gamma = 0.2  # core angle for the Lorentz factor function, in radians
gamma_0 = 100  # on-axis gamma
eta = 0.1  # kinetic energy --> gamma ray efficiency


@jit(nopython=True)
def calc_omega_H(chi_bh=1):
    """ Calculate dimensionless ang. freq. at the horizon, given chi_bh"""
    return chi_bh / (2 * (1 + np.sqrt(1 - chi_bh ** 2)))


@jit(nopython=True)
def calc_E_kin_jet(M_disc=1, chi_bh=1):
    """ Calculate the kinetic energy, given the disc mass in M_sun"""
    epsilon = 0.015  # as specified in Barbieri et al.
    xi_wind = 0.01  # fraction of M_disc ejected as wind
    xi_secular = 0.2  # fraction of M_disc which is secular ejecta
    omega_H = calc_omega_H(chi_bh)
    f = 1 + 1.38 * omega_H ** 2 - 9.2 * omega_H ** 4
    return (
        epsilon
        * (1 - xi_wind - xi_secular)
        * M_disc
        * M_SUN
        * C ** 2
        * omega_H ** 2
        * f
    )


@jit(nopython=True)  # need for faster code! DO NOT REMOVE!
def gaussian(theta, phi, theta_v, M_disc=1, chi_bh=1):
    """Return function value at (theta, phi), corresponding to Gaussian jet."""
    gamma = 1 + (gamma_0 - 1) * np.exp(-((theta / theta_gamma) ** 2))
    beta = np.sqrt(1 - 1 / gamma ** 2)
    E_c = calc_E_kin_jet(M_disc, chi_bh) / (PI * theta_E ** 2)
    dE_dOmega = E_c * np.exp(-((theta / theta_E) ** 2))
    cos_alpha = np.cos(theta_v) * np.cos(theta) + np.sin(theta_v) * np.sin(
        theta
    ) * np.cos(phi)
    # Define return values
    retval = np.sin(theta) * dE_dOmega / ((gamma ** 4) * ((1 - beta * cos_alpha) ** 3))
    return eta * retval


def do_gauss_cutoff_integral(theta_v, cutoff_angle, M_disc=1, chi_bh=1):
    """Return the value after integrating over a gaussian jet with a cutoff."""
    ans, err, out_dict = nquad(
        gaussian,
        ranges=[[0, cutoff_angle], [0, 2 * PI]],
        args=(theta_v, M_disc, chi_bh),
        full_output=True,
    )
    return ans, err, out_dict


@jit(nopython=True)
def calc_onaxis(E_kin_jet, theta_E=0.1, eta=0.1):
    return eta * E_kin_jet / (1 - np.cos(theta_E))


if __name__ == "__main__":
    view_angle = np.radians(np.logspace(0, 2, 500))

    lorentz = 1 + (gamma_0 - 1) * np.exp(-((view_angle / theta_gamma) ** 2))

    E_kin_iso = (
        4
        * PI
        * calc_E_kin_jet(0.2, 0.8)
        / (PI * theta_E ** 2)
        * np.exp(-((view_angle / theta_E) ** 2))
    )

    E_iso = np.zeros(view_angle.size)
    cutoff = PI / 3
    for idx, angle in tqdm(
        enumerate(view_angle), desc="Gauss Status", total=view_angle.size
    ):
        E_iso[idx] = do_gauss_cutoff_integral(angle, cutoff)[0]

    E_iso = np.asarray(E_iso)
    fig, ax1 = plt.subplots()
    ax1.set_xlabel(r"$\theta_{view}$ or $\theta$ [deg]", size=14)
    ax1.set_ylabel("Isotropic Equivalent Energy", size=14)
    ax1.set_ylim([1, 1e-10])
    ax1.set_yticks(np.logspace(-10, 0, 11))
    ax1.tick_params(axis="x", labelsize=14)
    ax1.tick_params(axis="y", labelsize=14)
    ax1.invert_yaxis()
    l1 = ax1.loglog(
        np.degrees(view_angle),
        E_kin_iso / E_kin_iso[0],
        "--",
        color="gray",
        label="Kinetic",
        lw=4,
    )
    l2 = ax1.loglog(
        np.degrees(view_angle), E_iso / E_iso[0], "r", label="Radiated", lw=4
    )

    ax2 = ax1.twinx()
    ax2.set_ylabel(r"$\Gamma(\theta)$", size=14)
    ax2.set_ylim([1e2, 1])
    ax2.invert_yaxis()
    ax2.tick_params(axis="y", labelsize=14)
    l3 = ax2.loglog(np.degrees(view_angle), lorentz, "b", label="Lorentz Factor", lw=4)

    ls = l1 + l2 + l3
    labs = [i.get_label() for i in ls]
    ax1.legend(ls, labs, loc="lower left", fontsize=14)

    plt.show()
