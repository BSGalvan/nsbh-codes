#!/usr/bin/env python
# Program to compute the maximum distance at which the
# Advanced LIGO network will yield an optimal SNR of 10,
# given that all detectors operate at the design sensitivities
# Usage: ./d_max [--steps=1000 | --eps=1e-10 | --verbose=False ] LOW HIGH
# Get help with ./d_max.py -h,--help
# Original Author: B.S. Bharath Saiguhan, github.com/bsgalvan

from argh import arg, dispatch_command
import numpy as np

from snr_function_lalsim import optimal_snr


def calc_nwsnr(dl):
    """Compute network SNR, for a given luminosity distance, by calling optimal_snr"""
    return (
        np.sum(np.array(optimal_snr(m1det=20.0, m2det=1.4, DL=dl, Lambda2=228)) ** 2)
        ** 0.5
        - 10.0
    )


def bisection(func, bracket, steps=10 ** 3, epsilon=1e-6, verbose=False):
    """Compute the root to a function, using the bisection method

    Parameters
    ----------
    func : function object.
           The function for which a root is required
    bracket : tuple.
              Initial guess for interval in which root lies
    steps : int, default 1000.
            Number of steps to compute for before stopping, and it is
            used in the stopping condition as: "Exit if step_count > steps"
    epsilon : float, default 1e-6.
              Tolerance for the root found, used in stopping condition as:
              "Exit if ||bracket|| < epsilon"

    Returns
    -------
    root : root of the function.

    """
    left, right = bracket[0], bracket[1]
    f_left, f_right = func(left), func(right)
    mid = (left + right) / 2
    f_mid = func(mid)
    step_count = 0
    if verbose:
        print(f"Bracket: {[left, right]},\tRoot: {mid}")

    while abs(left - right) > epsilon and step_count <= steps:
        if f_mid * f_left < 0:
            right = mid
            f_right = func(right)
        if f_mid * f_right < 0:
            left = mid
            f_left = func(left)
        mid = (left + right) / 2
        f_mid = func(mid)
        step_count += 1
        if verbose:
            print(f"Bracket: {[left, right]},\tRoot: {mid}")

    return mid


@arg("low", help="Lower bracket")
@arg("high", help="Higher bracket")
@arg("--steps", "-s", help="Number of bisection steps to perform.")
@arg("--eps", "-e", help="Tolerance for the root to be found via bisection.")
@arg("--verbose", "-v", help="Chatty outputs. Useful for debugging.")
def d_max(low, high, steps=1000, eps=1e-10, verbose=False):
    """Compute the maximal (luminosity) distance at which SNR(d_max) = 10.

    This is computed as the optimal network SNR possible for an NSBH merger, with a
    mass ratio of 20:1.4 and the network consisting of the LIGO Livingston, LIGO
    Hanford and VIRGO detectors.

    Parameters
    ----------
    low : float.
        Lower bracket around the guessed distance
    high : float.
        Higher bracket around the guessed distance
    steps : int, optional. Default is 1000.
        Number of bisection steps to perform.
    eps : float, optional. Default is 1e-10.
        Tolerance for the root to be found via bisection.
    verbose : bool, optional. Default is False.
        Chatty calculations. Useful for debugging.

    Returns
    -------
    max_dl : float, in Mpc.
        The maximum luminosity distance desired.
    """
    init_bracket = [float(low), float(high)]
    max_dl = bisection(calc_nwsnr, init_bracket, int(steps), float(eps), verbose)
    if verbose:
        print("=" * 64)
        print(f"==> D_MAX for a SNR=10, q=20:1.4 NSBH merger is ~ {max_dl} Mpc")
        print("=" * 64)
    return max_dl


if __name__ == "__main__":
    dispatch_command(d_max)
