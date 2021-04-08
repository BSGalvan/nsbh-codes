#!/bin/python3
# Module to define the major EM signals' derived classes.
# These are derived from the NSBH Merger base class.

# %% Imports, Auxiliary Function Definitions and constants.

import numpy as np

import nsbh_merger


class KiloNova(nsbh_merger.NSBHMerger):

    """Class for defining aspects of a Kilonova, arising from an NSBH merger."""

    def __init__(self):
        """Define logical defaults for the parameters."""
        nsbh_merger.NSBHMerger.__init__(self)

        # make sure we compute the relevant masses
        nsbh_merger.NSBHMerger.compute_masses()

        # also compute the dynamical ejecta RMS velocity
        nsbh_merger.NSBHMerger.velrms_dyn()

        # set defaults for the nuclear heating, from ^Barbieri et al., 2019
        self.eps_0 = 1e18  # in erg g^(-1) s^(-1)
        self.eps_th = 0.35  # avg(0.2, 0.5)
        self.sigma = 0.11  # in s
        self.t_0 = 1.3  # in s

        # set timespan for the calculations
        self.timespan = np.arange(0.5, 11.0, 0.5) * nsbh_merger.DAY


class DynamicalEjecta(KiloNova):

    """Class for defining Dynamical Ejecta objects."""

    def __init__(self):
        """Set logical defaults for dynamical ejecta objects. """
        KiloNova.__init__(self)

        # dynamical ejecta
        self.kappa_dyn = 20  # in cm^2 g^(-1)
        self.phi_dyn = nsbh_merger.PI  # azimuthal extent, in rad
        self.theta_dyn = 0.35  # latitudinal extent, avg(0.2, 0.5), in rad
        self.velmin_dyn = 0.1 * nsbh_merger.C
        self.velmax_dyn = self.compute_velmax(self.velrms_dyn, self.velmin_dyn)

        # Define the velocity-space mesh over which to calculate
        # the density and opacity functions
        self.vel_grid, self.theta_grid, self.phi_grid = (
            np.linspace(
                self.velmin_dyn, self.velmax_dyn, 100
            ),  # velocity/radial coordinate
            np.linspace(
                -self.theta_dyn, self.theta_dyn, 100
            ),  # latitudinal coordinate (from equator!)
            np.linspace(0, self.phi_dyn, 100),  # longitudinal coordinate
        )

        self.velsp_mesh = np.meshgrid(self.vel_grid, self.theta_grid, self.phi_grid)

    def compute_velmax(velrms, velmin):
        """Compute the maximum velocity, given RMS and minimum velocities.

        :returns: maximum velocity of a given ejecta

        """
        velmax = np.sqrt(3 * velrms ** 2 - 0.75 * velmin ** 2) - 0.5 * velmin

        return velmax

    def rho_dyn(self, vel_dyn, t):
        """Compute the dynamical ejecta density at the point (v_dyn, t) in spacetime.

        :vel_dyn: velocity coordinate for the dynamic ejecta
        :t: time coordinate
        :returns: density as a function of (vel_dyn, t)

        """
        rho = (
            self.mass_dyn
            / (2 * self.phi_dyn * self.theta_dyn * (self.velmax_dyn - self.velmin_dyn))
            * (vel_dyn ** -2)
            * (t ** -3)
        )
        return rho

    def calc_bounds(self, theta, t):
        """Compute the various boundaries in the dynamical ejecta.

        Parameters
        ----------
        theta : array of polar angles for which to calculate radial coordinate
        t : (1 X N) array of times at which to calculate radial coordinate

        Returns
        -------
        2 arrays of radial coordinates such that boundary points are at (r_min, theta)
        and (r_max, theta) at t

        """
        r_min = self.velmin_dyn * (self.theta_dyn - theta + 1) * t
        r_max = self.velmax_dyn * (theta - self.theta_dyn + 1) * t

        return r_min, r_max


class NeutrinoWindEjecta(KiloNova):

    """Class for defining NeutrinoWindEjecta objects, and their properties."""

    def __init__(self):
        """Set logical defaults for NeutrinoWindEjecta objects. """
        KiloNova.__init__(self)

        # neutrino wind ejecta
        self.xi_n = 0.01
        self.mass_n = self.xi_n * self.mass_disc
        self.kappa_n = 1  # in cm^2 g^(-1)
        self.theta_n = nsbh_merger.PI / 3  # latitudinal extent of neutrino-driven wind
        self.velrms_n = 0.0667 * nsbh_merger.C
        self.velmax_n = 3 * self.velrms_n

        # Define the velocity-space mesh over which to calculate
        # the density and opacity functions
        self.vel_grid, self.theta_grid, self.phi_grid = (
            np.linspace(0, self.velmax_n, 100),  # velocity/radial coordinate
            np.linspace(
                0, self.theta_n, 100
            ),  # latitudinal coordinate (from north pole!)
            np.linspace(0, 2 * nsbh_merger.PI, 100),  # longitudinal coordinate
        )

        self.velsp_mesh = np.meshgrid(self.vel_grid, self.theta_grid, self.phi_grid)

    def rho_n(self, vel_n, t):
        """Compute the neutrino-wind ejecta density at the point (v_n, t) in spacetime.

        Parameters
        ----------
        vel_n : velocity coordinate for the neutrino wind ejecta
        t : time coordinate

        Returns
        -------
        Density as a function of (vel_n, t)

        """
        rho = (
            (35 * self.mass_n)
            / (
                64
                * nsbh_merger.PI
                * (1 - np.cos(self.theta_n))
                * self.velmax_n
                * vel_n ** 2
                * t ** 3
            )
            * (1 - (vel_n / self.velmax_n) ** 2) ** 3
        )
        return rho


class ViscousWindEjecta(KiloNova):

    """Class for defining ViscousWindEjecta objects, and their properties."""

    def __init__(self):
        """Set logical defaults for ViscousWindEjecta objects. """
        KiloNova.__init__(self)

        # viscous wind ejecta
        self.xi_v = 0.2
        self.mass_v = self.xi_v * self.mass_disc
        self.kappa_v = 5  # in cm^2 g^(-1)
        self.theta_v = nsbh_merger.PI / 2  # latitudinal extent of viscous-driven wind
        self.velrms_v = 0.035 * nsbh_merger.C  # avg(0.03-0.04) * C
        self.velmax_v = 3 * self.velrms_v

        # Define the velocity-space mesh over which to calculate
        # the density and opacity functions
        self.vel_grid, self.theta_grid, self.phi_grid = (
            np.linspace(0, self.velmax_v, 100),  # velocity/radial coordinate
            np.linspace(
                0, self.theta_v, 100
            ),  # latitudinal coordinate (from north pole!)
            np.linspace(0, 2 * nsbh_merger.PI, 100),  # longitudinal coordinate
        )

        self.velsp_mesh = np.meshgrid(self.vel_grid, self.theta_grid, self.phi_grid)

    def rho_v(self, vel_v, theta, t):
        """Compute the density of viscous wind ejecta at (vel_v, theta, t) in spacetime.

        Parameters
        ----------
        vel_v : velocity coordinate of the viscous wind ejecta
        theta : angular position of the viscous wind ejecta
        t : time coordinate

        Returns
        -------
        Density as a function of (vel_v, t)

        """
        rho = (
            (105 * self.mass_v * np.sin(theta) ** 2)
            / (128 * nsbh_merger.PI * self.velmax_v * vel_v ** 2 * t ** 3)
            * (1 - (vel_v / self.velmax_v) ** 2) ** 3
        )
        return rho
