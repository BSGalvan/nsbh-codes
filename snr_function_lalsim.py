from math import sqrt

import numpy as np
import pop_utils as pput
import lal
import lalsimulation as lalsim
import pycbc.detector as pydet
import pycbc.waveform
from numba import jit

DETS = {}
DETS["H1"] = pydet.Detector(detector_name="H1", reference_time=1126259462.0)
DETS["L1"] = pydet.Detector(detector_name="L1", reference_time=1126259462.0)
DETS["V1"] = pydet.Detector(detector_name="V1", reference_time=1126259462.0)
DETS["K1"] = pydet.Detector(detector_name="K1", reference_time=1126259462.0)
DETS["I1"] = pydet.Detector(detector_name="I1", reference_time=1126259462.0)
"""
DETS = {}
DETS['H1'] = pydet.Detector(detector_name='H1')
DETS['L1'] = pydet.Detector(detector_name='L1')
DETS['V1'] = pydet.Detector(detector_name='V1')
DETS['K1'] = pydet.Detector(detector_name='K1')
DETS['I1'] = pydet.Detector(detector_name='I1')
"""


@jit(nopython=True)
def gpst_2_gmst(gps):
    k = 13713.440712226984
    c = 45991.090394612074
    gmst = gps / k - c
    return "GMST = ", gmst, " GMST in radian = ", np.mod(gmst, 2 * np.pi)


@jit(nopython=True)
def gpst_2_gmstRad(gps):
    k = 13713.440712226984
    c = 45991.090394612074
    gmst = gps / k - c
    return np.mod(gmst, 2 * np.pi)


@jit(nopython=True)
def raDec2PhiTheta(ra, dec, gpst):
    gmstRad = gpst_2_gmstRad(gpst)
    theta = np.pi / 2 - dec
    phi = ra - gmstRad
    return phi, theta


@jit(nopython=True)
def PhiTheta2raDec(phi, theta, gpst):
    gmstRad = gpst_2_gmstRad(gpst)
    dec = np.pi / 2 - theta
    ra = gmstRad + phi
    if ra > (2 * np.pi):
        ra = ra - (2 * np.pi)
    return ra, dec


def fplus_fcross_pycbc(detector="H1", theta=0.0, phi=0.0, psi=0.0):
    """Compute a detector's sensitivity pattern for a given sky location."""
    det = DETS[detector]
    reference_time = 1126259462.0  # What is this?
    ra, dec = PhiTheta2raDec(phi=phi, theta=theta, gpst=reference_time)
    fp, fc = det.antenna_pattern(
        right_ascension=ra, declination=dec, polarization=psi, t_gps=reference_time
    )
    return fp, fc


@jit(nopython=True)
def fvec_4_hp(hp):
    N = len(hp.data.data)
    fmax = (N - 1) * hp.deltaF
    f = np.arange(hp.f0, fmax + hp.deltaF, hp.deltaF)
    return f


# %% waveform call
# currently include only dquadmon1 and 2 inputs
# can be generelized for any other params like lambda1 and 2 etc.


@jit(nopython=True)
def innerProduct(df, h1, h2):
    return sqrt((4.0 * np.sum(df * h1 * np.conj(h2))).real)


def optimal_snr(
    m1det=10.0,
    m2det=10.0,
    DL=500.0,
    theta=0,
    phi=0,
    psi=0,
    iota=0,
    S1x=0.0,
    S1y=0.0,
    S1z=0.0,
    S2x=0.0,
    S2y=0.0,
    S2z=0.0,
    phiRef=0,
    f_ref=0,
    eccentricity=0.0,
    meanPerAno=0.0,
    longAscNodes=0.0,
    Lambda1=0.0,
    Lambda2=0.0,
    dkappa1=0.0,
    dkappa2=0.0,
    f_min=16.0,
    f_max=1023.75,
    deltaF=0.005,
    detector=["L1", "H1", "V1"],
    psdfn=[pput.allo_o3a, pput.alho_o3a, pput.avirgo_o3a],
    wfmodel="TaylorF2",  # changed from IMRPhenomPv2
):

    """
    Compute SNR at any detector structures and sensitivity (default L1 + aplus)
    """
    hp, hc = pycbc.waveform.get_fd_waveform(
        mass1=m1det,
        mass2=m2det,
        spin1x=S1x,
        spin1y=S1y,
        spin1z=S1z,
        spin2x=S2x,
        spin2y=S2y,
        spin2z=S2z,
        distance=DL,
        inclination=iota,
        dquad_mon1=dkappa1,
        dquad_mon2=dkappa2,
        delta_f=deltaF,
        f_lower=f_min,
        f_final=f_max,
        approximant=wfmodel,
    )

    f = np.array(hp.sample_frequencies.data)
    # f         = np.arange(0, f_max + deltaF, deltaF)
    inBand = (f >= f_min) & (f <= f_max)
    f = f[inBand]  # trimmed the zero paddings
    hpf = np.array(hp.data)[inBand]
    hcf = np.array(hc.data)[inBand]
    # ------------------------------------
    rho = []
    for DET, PSD in zip(detector, psdfn):
        Snf = PSD(f)
        Fp, Fc = fplus_fcross_pycbc(DET, theta, phi, psi)
        hf = hpf * Fp + hcf * Fc
        # print len(hf), len(Snf)
        rho += [innerProduct(deltaF, hf, hf / Snf)]
    return rho


def get_lalsim_wf(
    m1=10.0,
    m2=5.0,  # both in Msun
    distance=1000.0,  # in Mpc
    inclination=0.0,
    #
    f_min=20.0,
    f_max=2048.0,
    deltaF=0.005,
    #
    S1x=0.0,
    S1y=0.0,
    S1z=0.0,
    S2x=0.0,
    S2y=0.0,
    S2z=0.0,
    #
    phiRef=0,
    f_ref=0,
    eccentricity=0.0,
    meanPerAno=0.0,
    longAscNodes=0.0,
    #
    Lambda1=0.0,
    Lambda2=0.0,
    dkappa1=0.0,
    dkappa2=0.0,
    model=lalsim.IMRPhenomPv2,
    returnObject=False,  # returns the hp,hc objects instead of data
):

    # process starts
    [m1SI, m2SI, distanceSI] = [
        m1 * lal.MSUN_SI,
        m2 * lal.MSUN_SI,
        distance * lal.PC_SI * 1e6,
    ]
    # LAL Dictionary with BH kappa (specify explicitly dQuadMon1 and 2)
    print("adding dk1,dk2 = ", dkappa1, dkappa2)
    params = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertTidalLambda1(params, Lambda1)
    lalsim.SimInspiralWaveformParamsInsertTidalLambda2(params, Lambda1)
    lalsim.SimInspiralWaveformParamsInsertdQuadMon1(params, dkappa1)
    lalsim.SimInspiralWaveformParamsInsertdQuadMon2(params, dkappa2)
    # m = m1 + m2
    # msols = m * lal.MTSUN_SI
    # Generating waveform
    hp, hc = lalsim.SimInspiralChooseFDWaveform(
        m1SI,  # mass of companion 1 (kg)
        m2SI,  # mass of companion 2 (kg)
        S1x,
        S1y,
        S1z,
        S2x,
        S2y,
        S2z,  # all the components of the dimensionless spins 1 and 2
        distanceSI,  # distance of source (m)
        inclination,  # inclination of source (rad)
        phiRef,  # reference orbital phase (rad)
        longAscNodes,  # longitude of ascending nodes,
                       # degenerate with the polarization angle,
                       # Omega in documentation
        eccentricity,  # eccentricity at reference epoch
        meanPerAno,  # mean anomaly of periastron
        deltaF,  # sampling interval (Hz)
        f_min,  # starting GW frequency (Hz)
        f_max,  # final GW frequency (Hz)
        f_ref,  # reference GW frequency (Hz)
        params,  # LAL dictionary containing accessory parameters
        model,  # post-Newtonian approximant to use for waveform production
    )

    hpf = hp.data.data
    hcf = hc.data.data
    f_lal = fvec_4_hp(hp)
    low_index = int(f_min / deltaF)
    f_lal = f_lal[low_index:]
    hpf = hpf[low_index:]
    hcf = hcf[low_index:]
    if returnObject is False:
        return f_lal, hpf, hcf
    else:
        return f_lal, hp, hc


#################################


def get_snr_using_lalsim_wf(
    m1det=10.0,
    m2det=10.0,
    DL=500.0,
    theta=0,
    phi=0,
    psi=0,
    iota=0,
    S1x=0.0,
    S1y=0.0,
    S1z=0.0,
    S2x=0.0,
    S2y=0.0,
    S2z=0.0,
    phiRef=0,
    f_ref=0,
    eccentricity=0.0,
    meanPerAno=0.0,
    longAscNodes=0.0,
    Lambda1=0.0,
    Lambda2=0.0,
    dkappa1=0.0,
    dkappa2=0.0,
    f_min=10.0,
    f_max=2048.0,
    deltaF=0.005,
    detector=["L1", "H1", "V1"],
    psdfn=[pput.allo_o3a, pput.alho_o3a, pput.avirgo_o3a],
    wfmodel=lalsim.IMRPhenomPv2,
):

    """
        Computes SNR at any detector structures and sensiivity (by default for L1 with aplus)
    ------------------------------------------------------
    All the inputs are in earth frame (theta,phi are the sky locations in geocentric frame)
    ------------------------------------------------------
    Masses are the detector-frame masses, so for a case where the redshift effect is non-negligible,
    the appropreate factor should be already included; Mdet = Msrc x (1+z)
    ----------------------------------------
    """

    f, hpf, hcf = get_lalsim_wf(
        m1=m1det,
        m2=m2det,
        distance=DL,
        inclination=iota,
        f_min=f_min,
        f_max=f_max,
        deltaF=deltaF,
        S1x=S1x,
        S1y=S1y,
        S1z=S1z,
        S2x=S2x,
        S2y=S2y,
        S2z=S2z,
        phiRef=phiRef,
        f_ref=f_ref,
        eccentricity=eccentricity,
        meanPerAno=meanPerAno,
        longAscNodes=longAscNodes,
        Lambda1=Lambda1,
        Lambda2=Lambda2,
        dkappa1=dkappa1,
        dkappa2=dkappa2,
        model=wfmodel,
    )
    # ------------------------------------
    rho = []
    for DET, PSD in zip(list(detector), list(psdfn)):
        Snf = PSD(f)
        Fp, Fc = fplus_fcross_pycbc(DET, theta, phi, psi)
        hf = hpf * Fp + hcf * Fc
        # print len(hf), len(Snf)
        rho += [np.sqrt(np.real(innerProduct(deltaF, hf, hf / Snf)))]
    return rho


######################


def LonLat2raDec(Lon, Lat, gpst):  # not approved, only a trial
    PI = np.pi
    PIbyTWO = 0.5 * PI
    gmstRad = gpst_2_gmstRad(gpst)
    if Lat > PIbyTWO:
        dec = PIbyTWO - Lat
    else:
        dec = Lat

    if Lon > PI:
        Lon = Lon - PI
    if Lon < -PI:
        Lon = Lon + PI

    if Lon > 0:
        ra = gmstRad + Lon
    if Lon < 0:
        ra = gmstRad + (2 * PI - abs(Lon))
    if ra > (2 * PI):
        ra = ra - (2 * PI)
    return ra, dec
