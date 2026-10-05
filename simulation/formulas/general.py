"""
Photocurrent statistics for setup 0-3, built from setup_0.py ... setup_3.py.

For every setup sN and current X in {I1, I2, Ip (= I+), Im (= I-)}:

    sN_expt_X(*P)          <I_X>                                         eq. (113)/(29)
    sN_V_X(*P)             V(I_X), time domain (expectation values + constant terms)
                           eq. (9)-(12) for I1, I2 and eq. (6) for I+-
    sN_VdB_X(*P)           10 log10( V / V|_{r=0} )                      eq. (13)/(14)
    sN_V_X_f(f, df, *P)    V_df(I_X)(f) = df * PSD_X(f), frequency domain eq. (19)
    sN_VdB_X_f(f, *P)      10 log10( PSD / PSD|_{r=0} )                  eq. (20)
    sN_Cov_I1I2(*P)        Cov(I1, I2), time domain                       eq. (7)-(8)
    sN_Cov_I1I2_f(f,df,*P) df * PSD_I1I2(f)

*P are the parameters in exactly the same order as in the respective setup file:

    setup 0: alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t
    setup 1: alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0
    setup 2: alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0
    setup 3: alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0

<dz dz^d> = B/pi (setup 1-3) and Bsqz/pi (setup 0), eq. (91).
Shot-noise reference: r = 0 (setup 2: r = rp = 0).
"""

import numpy as np

from . import setup_0
from . import setup_1
from . import setup_2
from . import setup_3

q_e = 1.602176634e-19
c = 299792458

### General formulas (plain numbers in, numbers out)

def L_to_tau(n, dL):
    # Transforms length difference to time difference in optical fibre
    return n * dL / c

def expt_I(eta_PD, bAd_bA, e_dAd_dA):
    # <I> = eta_PD e [ A^*A + <dA^d dA> ]
    return np.real(eta_PD*q_e*(bAd_bA + e_dAd_dA))


def V_I(eta_PD, bAd_bA, bAd_bAd, bA_bA, e_dAd_dA, e_dA_dAd, e_dA_dA, e_dAd_dAd, e_dz_dzd):
    # V(I) = eta e^2 [ eta ( A^*A (<dA^d dA> + <dA dA^d>) + A^*A^* <dA dA> + A A <dA^d dA^d>
    #                        + <dA^d dA^d><dA dA> + <dA^d dA><dA dA^d> )
    #                  + (1 - eta) ( A^*A + <dA^d dA> ) <dz dz^d> ]
    n = eta_PD
    return np.real(n*q_e**2*(
        n*(bAd_bA*(e_dAd_dA + e_dA_dAd) + bAd_bAd*e_dA_dA + bA_bA*e_dAd_dAd
           + e_dAd_dAd*e_dA_dA + e_dAd_dA*e_dA_dAd)
        + (1 - n)*(bAd_bA + e_dAd_dA)*e_dz_dzd))


def Cov_I(eta_PD1, eta_PD2, bA1d_bA2, bA1d_bA2d, e_dA1_dA2d, e_dA1_dA2, e_dA1d_dA2d, e_dA1d_dA2):
    # Cov(I1, I2) = eta1 eta2 e^2 [ 2Re{A1^* A2 <dA1 dA2^d>} + 2Re{A1^* A2^* <dA1 dA2>}
    #                               + <dA1^d dA2^d><dA1 dA2> + <dA1^d dA2><dA1 dA2^d> ]
    return np.real(eta_PD1*eta_PD2*q_e**2*(
        2*np.real(bA1d_bA2*e_dA1_dA2d) + 2*np.real(bA1d_bA2d*e_dA1_dA2)
        + e_dA1d_dA2d*e_dA1_dA2 + e_dA1d_dA2*e_dA1_dA2d))


def V_pm(V1, V2, Cov, sign):
    # V(I+-) = V(I1) + V(I2) +- 2 Cov(I1, I2),  sign = +1 or -1
    return V1 + V2 + sign*2*Cov


def V_dB(V, V_r0):
    # V_dB = 10 log10( V / V|_{r=0} )
    return 10*np.log10(V/V_r0)


def V_f(PSD, df):
    # V_df(I)(f) = df * PSD_I(f)
    return df*PSD


### Glue: collect the terms of one setup and pass them to the formulas above

def _port(m, a, ad, eta_PD, e_dz_dzd, P):
    """Arguments of V_I for one port. a/ad = field name, e.g. 'A1p'/'A1dp'."""
    g = lambda name: getattr(m, name)(*P)
    return (eta_PD,
            g(f"b{ad}_b{a}"), g(f"b{ad}_b{ad}"), g(f"b{a}_b{a}"),
            g(f"expt_d{ad}_d{a}"), g(f"expt_d{a}_d{ad}"),
            g(f"expt_d{a}_d{a}"), g(f"expt_d{ad}_d{ad}"),
            e_dz_dzd)


def _cross(m, a1, a1d, a2, a2d, eta_PD1, eta_PD2, P):
    """Arguments of Cov_I."""
    g = lambda name: getattr(m, name)(*P)
    return (eta_PD1, eta_PD2,
            g(f"b{a1d}_b{a2}"), g(f"b{a1d}_b{a2d}"),
            g(f"expt_d{a1}_d{a2d}"), g(f"expt_d{a1}_d{a2}"),
            g(f"expt_d{a1d}_d{a2d}"), g(f"expt_d{a1d}_d{a2}"))


def _expt(port, j, P):
    t = port(j, P)
    return expt_I(t[0], t[1], t[4])


def _V(port, j, P):
    return V_I(*port(j, P))


def _Cov(cross, P):
    return Cov_I(*cross(P))


def _Vpm(port, cross, sign, P):
    return V_pm(_V(port, 1, P), _V(port, 2, P), _Cov(cross, P), sign)


def _r0(P, idx):
    """Same parameters but with the squeezing parameter(s) at index idx set to 0."""
    P = list(P)
    for i in idx:
        P[i] = 0.0
    return tuple(P)


### Setup 0   P = (alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t)

_R0_s0 = (5,)
def _s0_port(j, P):  return _port(setup_0, f"A{j}D", f"A{j}dD", P[2 + j], P[7]/np.pi, P)
def _s0_cross(P):    return _cross(setup_0, "A1D", "A1dD", "A2D", "A2dD", P[3], P[4], P)

def s0_expt_I1(*P):  return _expt(_s0_port, 1, P)
def s0_expt_I2(*P):  return _expt(_s0_port, 2, P)
def s0_expt_Ip(*P):  return s0_expt_I1(*P) + s0_expt_I2(*P)
def s0_expt_Im(*P):  return s0_expt_I1(*P) - s0_expt_I2(*P)

def s0_V_I1(*P):     return _V(_s0_port, 1, P)
def s0_V_I2(*P):     return _V(_s0_port, 2, P)
def s0_Cov_I1I2(*P): return _Cov(_s0_cross, P)
def s0_V_Ip(*P):     return _Vpm(_s0_port, _s0_cross, +1, P)
def s0_V_Im(*P):     return _Vpm(_s0_port, _s0_cross, -1, P)

def s0_VdB_I1(*P):   return V_dB(s0_V_I1(*P), s0_V_I1(*_r0(P, _R0_s0)))
def s0_VdB_I2(*P):   return V_dB(s0_V_I2(*P), s0_V_I2(*_r0(P, _R0_s0)))
def s0_VdB_Ip(*P):   return V_dB(s0_V_Ip(*P), s0_V_Ip(*_r0(P, _R0_s0)))
def s0_VdB_Im(*P):   return V_dB(s0_V_Im(*P), s0_V_Im(*_r0(P, _R0_s0)))

def s0_V_I1_f(f, df, *P):     return V_f(setup_0.PSD_I1(f, *P), df)
def s0_V_I2_f(f, df, *P):     return V_f(setup_0.PSD_I2(f, *P), df)
def s0_Cov_I1I2_f(f, df, *P): return V_f(setup_0.PSD_I1I2(f, *P), df)
def s0_V_Ip_f(f, df, *P):     return V_f(setup_0.PSD_Ip(f, *P), df)
def s0_V_Im_f(f, df, *P):     return V_f(setup_0.PSD_Im(f, *P), df)

def s0_VdB_I1_f(f, *P): return V_dB(setup_0.PSD_I1(f, *P), setup_0.PSD_I1(f, *_r0(P, _R0_s0)))
def s0_VdB_I2_f(f, *P): return V_dB(setup_0.PSD_I2(f, *P), setup_0.PSD_I2(f, *_r0(P, _R0_s0)))
def s0_VdB_Ip_f(f, *P): return V_dB(setup_0.PSD_Ip(f, *P), setup_0.PSD_Ip(f, *_r0(P, _R0_s0)))
def s0_VdB_Im_f(f, *P): return V_dB(setup_0.PSD_Im(f, *P), setup_0.PSD_Im(f, *_r0(P, _R0_s0)))


### Setup 1   P = (alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0)

_R0_s1 = (7,)
def _s1_port(j, P):  return _port(setup_1, f"A{j}p", f"A{j}dp", P[3 + j], P[9]/np.pi, P)
def _s1_cross(P):    return _cross(setup_1, "A1p", "A1dp", "A2p", "A2dp", P[4], P[5], P)

def s1_expt_I1(*P):  return _expt(_s1_port, 1, P)
def s1_expt_I2(*P):  return _expt(_s1_port, 2, P)
def s1_expt_Ip(*P):  return s1_expt_I1(*P) + s1_expt_I2(*P)
def s1_expt_Im(*P):  return s1_expt_I1(*P) - s1_expt_I2(*P)

def s1_V_I1(*P):     return _V(_s1_port, 1, P)
def s1_V_I2(*P):     return _V(_s1_port, 2, P)
def s1_Cov_I1I2(*P): return _Cov(_s1_cross, P)
def s1_V_Ip(*P):     return _Vpm(_s1_port, _s1_cross, +1, P)
def s1_V_Im(*P):     return _Vpm(_s1_port, _s1_cross, -1, P)

def s1_VdB_I1(*P):   return V_dB(s1_V_I1(*P), s1_V_I1(*_r0(P, _R0_s1)))
def s1_VdB_I2(*P):   return V_dB(s1_V_I2(*P), s1_V_I2(*_r0(P, _R0_s1)))
def s1_VdB_Ip(*P):   return V_dB(s1_V_Ip(*P), s1_V_Ip(*_r0(P, _R0_s1)))
def s1_VdB_Im(*P):   return V_dB(s1_V_Im(*P), s1_V_Im(*_r0(P, _R0_s1)))

def s1_V_I1_f(f, df, *P):     return V_f(setup_1.PSD_I1(f, *P), df)
def s1_V_I2_f(f, df, *P):     return V_f(setup_1.PSD_I2(f, *P), df)
def s1_Cov_I1I2_f(f, df, *P): return V_f(setup_1.PSD_I1I2(f, *P), df)
def s1_V_Ip_f(f, df, *P):     return V_f(setup_1.PSD_Ip(f, *P), df)
def s1_V_Im_f(f, df, *P):     return V_f(setup_1.PSD_Im(f, *P), df)

def s1_VdB_I1_f(f, *P): return V_dB(setup_1.PSD_I1(f, *P), setup_1.PSD_I1(f, *_r0(P, _R0_s1)))
def s1_VdB_I2_f(f, *P): return V_dB(setup_1.PSD_I2(f, *P), setup_1.PSD_I2(f, *_r0(P, _R0_s1)))
def s1_VdB_Ip_f(f, *P): return V_dB(setup_1.PSD_Ip(f, *P), setup_1.PSD_Ip(f, *_r0(P, _R0_s1)))
def s1_VdB_Im_f(f, *P): return V_dB(setup_1.PSD_Im(f, *P), setup_1.PSD_Im(f, *_r0(P, _R0_s1)))


### Setup 2   P = (alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0)

_R0_s2 = (8, 10)
def _s2_port(j, P):  return _port(setup_2, f"AS{j}p", f"AS{j}dp", P[4 + j], P[12]/np.pi, P)
def _s2_cross(P):    return _cross(setup_2, "AS1p", "AS1dp", "AS2p", "AS2dp", P[5], P[6], P)

def s2_expt_I1(*P):  return _expt(_s2_port, 1, P)
def s2_expt_I2(*P):  return _expt(_s2_port, 2, P)
def s2_expt_Ip(*P):  return s2_expt_I1(*P) + s2_expt_I2(*P)
def s2_expt_Im(*P):  return s2_expt_I1(*P) - s2_expt_I2(*P)

def s2_V_I1(*P):     return _V(_s2_port, 1, P)
def s2_V_I2(*P):     return _V(_s2_port, 2, P)
def s2_Cov_I1I2(*P): return _Cov(_s2_cross, P)
def s2_V_Ip(*P):     return _Vpm(_s2_port, _s2_cross, +1, P)
def s2_V_Im(*P):     return _Vpm(_s2_port, _s2_cross, -1, P)

def s2_VdB_I1(*P):   return V_dB(s2_V_I1(*P), s2_V_I1(*_r0(P, _R0_s2)))
def s2_VdB_I2(*P):   return V_dB(s2_V_I2(*P), s2_V_I2(*_r0(P, _R0_s2)))
def s2_VdB_Ip(*P):   return V_dB(s2_V_Ip(*P), s2_V_Ip(*_r0(P, _R0_s2)))
def s2_VdB_Im(*P):   return V_dB(s2_V_Im(*P), s2_V_Im(*_r0(P, _R0_s2)))

def s2_V_I1_f(f, df, *P):     return V_f(setup_2.PSD_I1(f, *P), df)
def s2_V_I2_f(f, df, *P):     return V_f(setup_2.PSD_I2(f, *P), df)
def s2_Cov_I1I2_f(f, df, *P): return V_f(setup_2.PSD_I1I2(f, *P), df)
def s2_V_Ip_f(f, df, *P):     return V_f(setup_2.PSD_Ip(f, *P), df)
def s2_V_Im_f(f, df, *P):     return V_f(setup_2.PSD_Im(f, *P), df)

def s2_VdB_I1_f(f, *P): return V_dB(setup_2.PSD_I1(f, *P), setup_2.PSD_I1(f, *_r0(P, _R0_s2)))
def s2_VdB_I2_f(f, *P): return V_dB(setup_2.PSD_I2(f, *P), setup_2.PSD_I2(f, *_r0(P, _R0_s2)))
def s2_VdB_Ip_f(f, *P): return V_dB(setup_2.PSD_Ip(f, *P), setup_2.PSD_Ip(f, *_r0(P, _R0_s2)))
def s2_VdB_Im_f(f, *P): return V_dB(setup_2.PSD_Im(f, *P), setup_2.PSD_Im(f, *_r0(P, _R0_s2)))


### Setup 3   P = (alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0)

_R0_s3 = (9,)
def _s3_port(j, P):  return _port(setup_3, f"A{j}pp", f"A{j}dpp", P[4 + j], P[11]/np.pi, P)
def _s3_cross(P):    return _cross(setup_3, "A1pp", "A1dpp", "A2pp", "A2dpp", P[5], P[6], P)

def s3_expt_I1(*P):  return _expt(_s3_port, 1, P)
def s3_expt_I2(*P):  return _expt(_s3_port, 2, P)
def s3_expt_Ip(*P):  return s3_expt_I1(*P) + s3_expt_I2(*P)
def s3_expt_Im(*P):  return s3_expt_I1(*P) - s3_expt_I2(*P)

def s3_V_I1(*P):     return _V(_s3_port, 1, P)
def s3_V_I2(*P):     return _V(_s3_port, 2, P)
def s3_Cov_I1I2(*P): return _Cov(_s3_cross, P)
def s3_V_Ip(*P):     return _Vpm(_s3_port, _s3_cross, +1, P)
def s3_V_Im(*P):     return _Vpm(_s3_port, _s3_cross, -1, P)

def s3_VdB_I1(*P):   return V_dB(s3_V_I1(*P), s3_V_I1(*_r0(P, _R0_s3)))
def s3_VdB_I2(*P):   return V_dB(s3_V_I2(*P), s3_V_I2(*_r0(P, _R0_s3)))
def s3_VdB_Ip(*P):   return V_dB(s3_V_Ip(*P), s3_V_Ip(*_r0(P, _R0_s3)))
def s3_VdB_Im(*P):   return V_dB(s3_V_Im(*P), s3_V_Im(*_r0(P, _R0_s3)))

def s3_V_I1_f(f, df, *P):     return V_f(setup_3.PSD_I1(f, *P), df)
def s3_V_I2_f(f, df, *P):     return V_f(setup_3.PSD_I2(f, *P), df)
def s3_Cov_I1I2_f(f, df, *P): return V_f(setup_3.PSD_I1I2(f, *P), df)
def s3_V_Ip_f(f, df, *P):     return V_f(setup_3.PSD_Ip(f, *P), df)
def s3_V_Im_f(f, df, *P):     return V_f(setup_3.PSD_Im(f, *P), df)

def s3_VdB_I1_f(f, *P): return V_dB(setup_3.PSD_I1(f, *P), setup_3.PSD_I1(f, *_r0(P, _R0_s3)))
def s3_VdB_I2_f(f, *P): return V_dB(setup_3.PSD_I2(f, *P), setup_3.PSD_I2(f, *_r0(P, _R0_s3)))
def s3_VdB_Ip_f(f, *P): return V_dB(setup_3.PSD_Ip(f, *P), setup_3.PSD_Ip(f, *_r0(P, _R0_s3)))
def s3_VdB_Im_f(f, *P): return V_dB(setup_3.PSD_Im(f, *P), setup_3.PSD_Im(f, *_r0(P, _R0_s3)))