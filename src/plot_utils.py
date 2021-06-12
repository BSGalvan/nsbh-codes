#!/usr/bin/env python
# Module to define plotting functions to be used elsewhere
# Original Author: B.S. Bharath Saiguhan, github.com/bsgalvan

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

from math_utils import ecdf

style.use(["fivethirtyeight", "seaborn-ticks"])


def mask_snrs(snrs, detectors, nwsnr_min=10.0, mask_pattern="xxx"):
    """Mask a given SNR array, with mask_pattern as the guiding constraint.

    Parameters
    ----------
    snrs : NumPy array.
        The array of snrs calculated at the detectors, after masking.
    detectors : list.
        The list of detectors where SNRs were computed.
    nwsnr_min : float, optional. Default is 10.0.
        The minimum network SNR to consider.
    mask_pattern : 3-character string, optional. Default is 'xxx'.
        One-hot encoding pattern for masking detections/non-detections at
        L/H/V detectors. For example: '10x' selects SNRs which are network
        detections AND Livingston detections AND Hanford non-detections
        (regardless of SNR at VIRGO), whereas 'xxx' selects SNRs which
        are network detections (regardless of SNR at L/H/V).
    Returns
    -------
    snrs_masked : NumPy array.
        A subset of snrs, masked according to the mask_pattern and nwsnr_min.

    """
    snr_mask = np.sum(snrs ** 2, axis=1) ** 0.5 > nwsnr_min  # case xxx

    for j in range(len(detectors)):
        if mask_pattern[j] == "1":  # cases 1__, _1_, __1 ...
            snr_mask *= snrs[:, j] > 4
        elif mask_pattern[j] == "x":  # cases x__, _x_, __x
            pass
        elif mask_pattern[j] == "0":  # cases 0__, _0_, __0
            snr_mask *= snrs[:, j] < 4
        else:
            raise ValueError("incorrect mask pattern!")

    snrs_masked = snrs[snr_mask]

    return snrs_masked


def plot_snrs(snrs, detectors, style="ecdf", nwsnr_min=10.0, mask_pattern="xxx"):
    """Plot the snrs at detectors, using the preferred style.

    Parameters
    ----------
    snrs : NumPy array.
        The array of snrs calculated at the detectors, after masking.
    detectors : list.
        The list of detectors where snrs were computed.
    style : string, optional. Default is 'ecdf'.
        Plot either the probability density function ('pdf') or the
        empirical cumulative distribution function.

    Returns
    -------
    fig : matplotlib.pyplot.figure Object.
        Handle to the figure object created.

    """
    N = len(detectors)
    fig, axs = plt.subplots(1, N, sharey=True, tight_layout=True)
    snrs_masked = mask_snrs(snrs, detectors, nwsnr_min, mask_pattern)

    if style not in ("ecdf", "pdf"):
        raise ValueError(f"style='{style}' not supported. Use 'ecdf' or 'pdf' instead.")
    elif style == "ecdf":
        fig.suptitle(
            f"Empirical Cumulative Distribution Functions for SNRs at {detectors}"
        )
        for i in range(N):
            xs, ys = ecdf(snrs_masked[:, i])
            axs[i].grid(True)
            axs[i].set_axisbelow(True)
            median_x = np.median(snrs_masked[:, i])
            axs[i].step(xs, ys)
            axs[i].set_xscale("log")
            axs[i].plot(median_x, 0.5, "o")
            axs[i].annotate(
                f"({median_x:.2}, 0.5)",
                xy=(median_x, 0.5),
                xytext=(median_x * 0.35, 0.6),
                textcoords="data",
                arrowprops=dict(
                    color="#000000", arrowstyle="->", connectionstyle="angle3"
                ),
                size=15,
                horizontalalignment="center",
                verticalalignment="bottom",
            )
            axs[i].set_xlabel(fr"SNR at {detectors[i]}, $\rho_{{{detectors[i]}}} $")
            axs[i].set_ylabel(
                fr"$\tilde{{F}}(\rho_{{{detectors[i]}}} | \rho_{{NW}} > 10)$"
            )

    else:
        fig.suptitle(f"Probability Density Functions for SNRs at {repr(detectors)}")
        for i in range(snrs.shape[1]):
            axs[i].grid(True)
            axs[i].set_axisbelow(True)
            median_x = np.median(snrs_masked[:, i])
            axs[i].set_xscale("log")
            density, _, _ = axs[i].hist(
                snrs_masked[:, i],
                bins=np.logspace(
                    np.log10(snrs_masked[:, i].min()),
                    np.log10(snrs_masked[:, i].max()),
                    50,
                ),
                density=True,
                log=True,
            )
            y_lo, y_hi = density.min(), density.max()
            axs[i].vlines(
                median_x,
                y_lo,
                y_hi,
                linestyles="dashed",
                color="C1",
                label=f"Median = {median_x:.3}",
            )
            axs[i].set_xlabel(fr"SNR at {detectors[i]}, $\rho_{{{detectors[i]}}} $")
            axs[i].set_ylabel(fr"$P(\rho_{{{detectors[i]}}} | \rho_{{NW}} > 10)$")
            axs[i].legend()

    # fig.show()

    return fig


def plot_fluences(fluences_masked, style="ecdf"):
    """Plot the masked fluences, that is those with fluence lesser than fluence_min.

    Parameters
    ----------
    fluences_masked : array-like.
        Fluences whose value is guaranteed to be > fluence_min
    style : string, optional. Default is 'ecdf'.
        Plot either the probability density function ('pdf') or the
        empirical cumulative distribution function.

    Returns
    -------
    fig : matplotlib.pyplot.figure Object.
        Handle to the figure object created.
    """

    fig, ax = plt.subplots(tight_layout=True)
    ax.set_xscale("log")
    median_f = np.median(fluences_masked)
    print(f"Got {fluences_masked.size} 'visible' events!")
    if style not in ("ecdf", "pdf"):
        raise ValueError(f"style='{style}' not supported. Use 'ecdf' or 'pdf' instead.")
    elif style == "ecdf":
        fxs, fys = ecdf(fluences_masked)
        ax.step(fxs, fys)
        ax.set_ylabel(
            r"$\tilde{F}(\mathcal{F}_\gamma|\mathcal{F}_\gamma>\mathcal{F}_{min.})$"
        )
        ax.plot(median_f, 0.5, "o")
        ax.annotate(
            f"({median_f:.3E}, 0.5)",
            xy=(median_f, 0.5),
            xytext=(median_f * 0.35, 0.6),
            textcoords="data",
            arrowprops=dict(color="#000000", arrowstyle="->", connectionstyle="angle3"),
            size=15,
            horizontalalignment="center",
            verticalalignment="bottom",
        )
        ax.set_title("ECDF of Fluence above INTEGRAL limit")
    else:
        density, _, _ = ax.hist(
            fluences_masked,
            bins=np.logspace(
                np.log10(fluences_masked.min()), np.log10(fluences_masked.max()), 50
            ),
            density=True,
            log=True,
        )
        ax.set_ylabel(r"$P(\mathcal{F}_\gamma|\mathcal{F}_\gamma>\mathcal{F}_{min.})$")
        y_lo, y_hi = density.min(), density.max()
        ax.vlines(
            median_f,
            y_lo,
            y_hi,
            linestyles="dashed",
            color="C1",
            label=f"Median = {median_f:.3}",
        )
        ax.set_title("PDF of Fluence above INTEGRAL limit")

    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel(r"Fluence, $\mathcal{F}_\gamma$ (erg/cm$^2$)")
    # fig.show()

    return fig


def plot_thetav(thetas, style="ecdf"):
    """Plot the viewing angle distribution.

    Parameters
    ----------
    thetas : array-like.
        Array of viewing angles for 'visible' NSBH mergers.
    style : string, optional. Default is 'ecdf'.
        Plot either the probability density function ('pdf') or the
        empirical cumulative distribution function.

    Returns
    -------
    fig : matplotlib.pyplot.figure Object.
        Handle to the figure object created.

    """
    median_thetav = np.median(thetas)
    fig, ax = plt.subplots(tight_layout=True)
    if style not in ("ecdf", "pdf"):
        raise ValueError(f"style='{style}' not supported. Use 'ecdf' or 'pdf' instead.")
    elif style == "ecdf":
        tvxs, tvys = ecdf(thetas)
        ax.step(np.degrees(tvxs), tvys)
        ax.set_ylabel(r"$\tilde{\mathcal{F}}(\theta_v)$")
        ax.plot(np.degrees(median_thetav), 0.5, "o")
        ax.annotate(
            f"({np.round(np.degrees(median_thetav), 2)}, 0.5)",
            xy=(np.degrees(median_thetav), 0.5),
            xytext=(np.degrees(median_thetav) * 0.35, 0.6),
            textcoords="data",
            arrowprops=dict(color="#000000", arrowstyle="->", connectionstyle="angle3"),
            size=15,
            horizontalalignment="center",
            verticalalignment="bottom",
        )
        ax.set_title("ECDF of the viewing angle of 'visible' NSBH Mergers")
    else:
        density, _, _ = ax.hist(np.degrees(thetas), bins=50, density=True)
        ax.set_ylabel(r"$P(\theta_v)$")
        y_lo, y_hi = density.min(), density.max()
        ax.vlines(
            np.degrees(median_thetav),
            y_lo,
            y_hi,
            linestyles="dashed",
            color="C1",
            label=f"Median = {median_thetav:.3}",
        )
        ax.set_title("PDF of the viewing angle of 'visible' NSBH Mergers")

    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel(r"Viewing Angle, $\theta_v$ (deg.)")
    # fig.show()

    return fig
