#!/usr/bin/env python
# Program to calculate luminosity distance at which network SNR = 10.

# Results (from manual bracketing)
# D_max ~ 575.2572572572573 i.e in (576.2552552552553, 576.2592592592592
import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np

from snr_function_lalsim import optimal_snr


def calc_optsnr(dl):
    """Call optimal_snr"""
    retval = np.sum(np.array(optimal_snr(m_bh, m_ns, dl, Lambda2=228)) ** 2) ** 0.5
    print(f"SNR for {dl} Mpc is {retval}")
    return retval


if __name__ == "__main__":
    m_bh = 20.0
    m_ns = 1.4
    dl_test = np.linspace(575, 577, 1000)

    with mp.Pool() as pool:
        res = pool.map(calc_optsnr, dl_test)

    res = np.array(res)

    print(dl_test[np.argmin((res - 10) ** 2)])

    plt.plot(dl_test, res)
    plt.show()
