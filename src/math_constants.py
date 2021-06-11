#!/usr/bin/env python
# Module defining all the relevant astronomical constants, in consistent units

import numpy as np
from astropy import constants as const

# All constants are in CGS units!
G = const.G.cgs.value  # Universal Gravitational Constant
C = const.c.cgs.value  # Speed of light
H = const.h.cgs.value  # Planck's constant
K = const.k_B.cgs.value  # Boltzmann's constant
SIGMA = const.sigma_sb.cgs.value  # Stefan-Boltzmannn constant
DAY = 86400  # 1 day in seconds
M_SUN = const.M_sun.cgs.value  # Solar mass
KPC = const.kpc.cgs.value  # Kiloparsec
MPC = KPC * 1000  # Megaparsec
PI = np.pi
