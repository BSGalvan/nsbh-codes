#!/bin/python3
# %% Imports, Auxiliary Function Definitions and constants

from astropy import constants as const
import numpy as np

# All constants are in CGS units!
G = const.G.cgs.value  # Universal Gravitational Constant
C = const.c.cgs.value  # Speed of light
H = const.h.cgs.value  # Planck's constant
K = const.k_B.cgs.value  # Boltzmann's constant
SIGMA = const.sigma_sb.cgs.value  # Stefan-Boltzmannn constant
M_SUN = const.M_sun.cgs.value  # Solar mass
KPC = const.kpc.cgs.value  # Kiloparsec
PI = np.pi


class NSBHMerger:

    """Class for defining top level aspects of an NSBH merger."""

    def __init__(self, m_bh=5, m_ns=1.4, spin_bh=0.5, td_ns=330):
        """Initialise logical defaults for objects. """
        self.mass_bh = m_bh * M_SUN  # mass of the primary
        self.mass_ns = m_ns * M_SUN  # mass of the secondary.
        self.chi_bh = spin_bh  # (effective) spin of the primary.
        self.lambda_ns = td_ns  # tidal deformability of the secondary.
        self.q = self.mass_bh / self.mass_ns  # mass ratio
        self.eta = self.q / (1 + self.q) ** 2  # symmetric mass ratio
        self.c_ns = self.c_love()  # compactness of the NS, via C-Love relations
        self.mass_bary = self.baryonic_mass()  # baryonic mass of the NS
        self.rcap_isco = self.rcap_isco()  # normalized ISCO radius

    def c_love(self):
        """Compute the compactness of a NS using the C-Love relation."""
        # Function Body
        a = np.array([0.36, -0.0355, 0.000705])  # From Yagi & Yunes, 2016
        return a.dot(np.array([1, np.log(self.lambda_ns), np.log(self.lambda_ns) ** 2]))

    def baryonic_mass(self):
        """Calculate the total baryonic mass."""
        # Function Body
        return self.mass_ns * (1 + (0.6 * self.c_ns) / (1 - 0.5 * self.c_ns))

    def rcap_isco(self):
        """Calculate the normalized ISCO radius for a given BH spin."""
        # Function Body
        z1 = 1 + (1 - self.chi_bh ** 2) ** (1 / 3) * (
            (1 + self.chi_bh) ** (1 / 3) + (1 - self.chi_bh) ** (1 / 3)
        )
        z2 = np.sqrt(3 * self.chi_bh ** 2 + z1 ** 2)
        return 3 + z2 - np.sign(self.chi_bh) * np.sqrt((3 - z1) * (3 + z1 + 2 * z2))

    def compute_masses(self, f=0.3):
        """Compute masses left outside the BH apparent radius, bound & unbound.
        The various masses are the remnant mass, m_out, which further consists of
        the dynamic mass, m_dyn, and the disc mass, m_disc. Thus,
        m_out = m_disc + m_out  (if the NS does not plunge into the BH)
        """
        rho = (15 * self.lambda_ns) ** (-1 / 5)

        # Compute mass_out. Formula from ^Foucart et al., 2018
        alpha, beta, gamma, delta = 0.308, 0.124, 0.283, 1.536
        term_alpha = alpha * (1 - 2 * rho) / (self.eta ** (1 / 3))
        term_beta = -beta * self.rcap_isco * rho / self.eta
        term_gamma = gamma
        self.mass_out = (
            self.mass_bary
            * (np.maximum(term_alpha + term_beta + term_gamma, 0.0)) ** delta
        )

        # Compute mass_dyn. Formula from ^Kawaguchi et al., 2016
        a1, a2, a3, a4, n1, n2 = 4.464e-2, 2.269e-3, 2.431, -0.4159, 0.2497, 1.352
        term_a1 = a1 * (self.q ** n1) * (1 - 2 * self.c_ns) / self.c_ns
        term_a2 = -a2 * (self.q ** n2) * (self.rcap_isco)
        term_a3 = a3 * (1 - self.mass_ns / self.mass_bary)
        term_a4 = a4
        self.mass_dyn = self.mass_bary * np.maximum(
            term_a1 + term_a2 + term_a3 + term_a4, 0
        )

        # Enforce upper limits on mass_dyn.
        if self.mass_dyn.size > 1:
            mask = self.mass_dyn > f * self.mass_out
            self.mass_dyn[mask] = f * self.mass_out[mask]
        else:
            self.mass_dyn = (
                f * self.mass_out
                if self.mass_dyn > f * self.mass_dyn
                else self.mass_dyn
            )

        # Compute m_disc
        self.mass_disc = np.maximum(self.mass_out - self.mass_dyn, 0)

    def velrms_dyn(self):
        """Compute the RMS velocity of the dynamical ejecta."""
        self.velrms_dyn = (0.01533 * self.q + 0.1907) * C

    def __str__(self):
        message = (
            f"User Inputs\n"
            f"===================\n"
            f"Mass of the Black Hole: {self.mass_bh / M_SUN} M_SUN\n"
            f"Mass of the Neutron Star: {self.mass_ns / M_SUN} M_SUN\n"
            f"Tidal Deformability of the Neutron Star: {self.lambda_ns}\n"
            f"Effective Spin of the Black Hole: {self.chi_bh}\n\n"
            f"Computed Quantities\n"
            f"===================\n"
            f"Mass Ratio: {self.q}\n"
            f"Symmetric Mass Ratio: {self.eta}\n"
            f"Compactness (via C-Love relation): {self.c_ns}\n"
            f"Baryonic Mass: {self.mass_bary / M_SUN} M_SUN\n"
            f"Normalized ISCO radius: {self.rcap_isco}\n"
        )
        return message
