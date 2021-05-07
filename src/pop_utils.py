import numpy as np

# import scipy
# import sys
import scipy.interpolate as si
from scipy.integrate import quad

# from scipy.integrate import simps

""" Conversion factors """

""" Standard LCDM cosmology"""
OmegaM = 0.3
OmegaK = 0.0
OmegaL = 0.7

c = 3.0e5
# km/sec
h = 0.7
H0 = 100 * h
# km  sec^-1 Mpc^-1
DH = c / H0
# Mpc    (Hubble distance)


def fn_E(z):
    return np.sqrt(OmegaM * (1 + z) ** 3 + OmegaL)


def Dc(z):  # in Mpc (since DH is in Mpc)
    def integrand(zprime):
        return DH / fn_E(zprime)

    return quad(integrand, 0, z,)[0]


def DL(z):
    return (1 + z) * Dc(z)


def z_vs_D_interp():
    z = np.linspace(0.0, 10, 20000)
    dc = z * 0
    for i in range(len(dc)):
        dc[i] = Dc(z[i])
    dL = dc * (1 + z)
    dc_fof_z = si.interp1d(z, dc, kind="slinear")
    z_fof_dc = si.interp1d(dc, z, kind="slinear")
    dL_fof_z = si.interp1d(z, dL, kind="slinear")
    z_fof_dL = si.interp1d(dL, z, kind="slinear")
    return dc_fof_z, z_fof_dc, dL_fof_z, z_fof_dL


#############################################################
# 	making interpolated functions so that it can be quickly
# 	accessed without interpolating everytime.

# 	these functions can be used for conversions between Dc vs z or DL vs z and vice versa

# 	each of the four terms in LHS are functions

# 	Eg, dc_fof_z (5) will return the Dc corresponding to z=5

dc_fof_z, z_fof_dc, dL_fof_z, z_fof_dL = z_vs_D_interp()


def dVbydz(z):  # comoving differential volume for flat (using interpolated Dc)
    Dcomoving = dc_fof_z(z)
    Ez = fn_E(z)
    return 4 * np.pi * Dcomoving ** 2 * DH / Ez  # in Mpc^3


def dVbydz_GpcCube(z):  #
    return 1e-9 * dVbydz(z)  # in Gpc^3


def dNz_dz(
    Rz, z
):  # Number of sources between z and z+dz, Rz should be rate density function
    return Rz(z) * dVbydz_GpcCube(z) / (1 + z)


def dNz_dz_nonevolving(z=1):  # Number of sources with redshift between z and z+dz,
    constantRate = 1
    return constantRate * dVbydz_GpcCube(z) / (1 + z)


def gen_random(xmin, xmax, Px, N=None):
    """Generate N samples from the distribution given by Px."""
    x = np.linspace(xmin, xmax, 300)  # create points for interpolation
    px = np.zeros_like(x)  # create array for samples
    dx = x[2] - x[1]  # spacing between points
    dPx = Px(x)  # evaluate PDF at points.
    px[0] = dPx[0] * dx  # probability = function_value * spacing
    for i in range(len(x) - 1):
        px[i + 1] = px[i] + dPx[i + 1] * dx
    # Create interpolations for interconversion between x and P(x).
    # x2px = si.interp1d(x, px, kind="slinear")  # kind = 1st order spline
    px2x = si.interp1d(px, x, kind="slinear")
    px_min = min(px)
    px_max = max(px)
    if N is None:
        N = int(
            px_max
        )  # N is chosen as predicted by the particular model  ie, N = integral N(z) dz
    pxRandom = np.random.uniform(px_min, px_max, N)
    xrandom = px2x(pxRandom)
    return xrandom


def populate_constant_comoving(zmax=1, N=50000):
    z = gen_random(xmin=0, xmax=zmax, Px=dNz_dz_nonevolving, N=N)
    return z


def uniform_location_orientation(N):
    # location choosen
    costheta = np.random.uniform(-1, 1, N)
    theta = np.arccos(costheta)
    phi = np.random.uniform(0, 2 * np.pi, N)
    # orientation chosen
    psi = np.random.uniform(0, 2 * np.pi, N)
    cosiota = np.random.uniform(-1, 1, N)
    iota = np.arccos(cosiota)
    return theta, phi, psi, iota


def Pm1(m):
    return m ** (-2.35)


def populate_m1m2(min, max, N):
    mass1 = gen_random(min, max, Pm1, N)
    mass2 = np.zeros_like(mass1)
    for i in range(N):
        mass2[i] = np.random.uniform(min, mass1[i], 1)
    return mass1, mass2


# ======================= noise curves =======================
# In some cases, noise curves are given as asd, in some cases as psd
# Do mention which one you have


def psdInterp(filename, filetype="psd"):
    data = np.loadtxt(filename)
    f = data[:, 0]
    snf = data[:, 1]
    if filetype == "asd":
        snf = snf ** 2
    snf_interp = si.interp1d(f, snf, kind="slinear")
    return snf_interp


# Below are the interpolated strings from files which can be directly used in Fisher
# estimations. The label on LHS is what is to be passed as argument in the Fisher
# function. If a new psd file (eg: ET, CE, LISA etc) is available, then add a line
# below for that with appropriate label, so that it can be also used

# CAUTION: whether the available file gives 'asd' or 'psd' must be CORRECTLY mentioned.
# aplus       = psdInterp(filename='../data/PSD/Aplus.txt',filetype='asd')
# avirgo_sqzd = psdInterp(filename='../data/PSD/AdVirgo_sqz.txt',filetype='asd')
# kagra       = psdInterp(filename='../data/PSD/Kagra.txt',filetype='asd')


allo_o3a = psdInterp(
    filename="data/O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt",
    filetype="asd",
)
alho_o3a = psdInterp(
    filename="data/O3-H1-C01_CLEAN_SUB60HZ-1251752040.0_sensitivity_strain_asd.txt",
    filetype="asd",
)
avirgo_o3a = psdInterp(filename="data/O3-V1_sensitivity_strain_asd.txt", filetype="asd")

allo_des = psdInterp(filename="data/aLIGO_ZERO_DET_high_P.txt", filetype="asd")
alho_des = psdInterp(filename="data/aLIGO_ZERO_DET_high_P.txt", filetype="asd")
avirgo_des = psdInterp(filename="data/aLIGO_ZERO_DET_high_P.txt", filetype="asd")

aligo_old = psdInterp(filename="data/aLIGO_old.txt", filetype="asd")
