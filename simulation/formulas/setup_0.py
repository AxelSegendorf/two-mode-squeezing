import numpy as np

### Help functions and setup
#
# Parameters (same order in every function):
#   alpha, phi       LO amplitude and phase
#   eta_FC           fibre coupler transmission
#   eta_PD1, eta_PD2 photodiode quantum efficiencies
#   r, theta         squeezing parameter and angle
#   Bsqz             squeezing (integration) bandwidth, Bsqz >= B + domega
#   domega           frequency shift of the LO, Delta omega
#   t                time (the constant field components rotate at domega)
#
# Notation: dA1D = delta A_1^Delta, dA1dD = (delta A_1^Delta)^dagger, bA1D = bar A_1^Delta, bA1dD = bar A_1^Delta*.
# The PSD functions are time averaged and symmetrised in f, so they do not depend on t.

q_e = 1.602176634e-19

def _aux(eta_FC):
    """sqrt(eta_FC), sqrt(1 - eta_FC), sqrt(eta_FC (1 - eta_FC))."""
    return np.sqrt(eta_FC), np.sqrt(1 - eta_FC), np.sqrt(eta_FC*(1 - eta_FC))


def _Pi(x, Bsqz):
    # Pi_Bsqz(x) = 1 for |x| < Bsqz, 0 otherwise
    return (np.abs(np.asarray(x)) < Bsqz).astype(float)


### Expectation values


def expt_dA1dD_dA1D(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # <dA1^d dA1>
    return eta_FC*Bsqz/np.pi*np.sinh(r)**2


def expt_dA1D_dA1dD(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # <dA1 dA1^d> = Bsqz/pi + <dA1^d dA1>
    return Bsqz/np.pi + expt_dA1dD_dA1D(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t)


def expt_dA1D_dA1D(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # <dA1 dA1>
    return eta_FC*Bsqz/np.pi*np.exp(1j*theta)*np.cosh(r)*np.sinh(r)


def expt_dA1dD_dA1dD(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # <dA1^d dA1^d> = <dA1 dA1>^*
    return np.conj(expt_dA1D_dA1D(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t))


def expt_dA2dD_dA2D(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # <dA2^d dA2>  (dA1 with eta_FC -> 1 - eta_FC)
    return (1 - eta_FC)*Bsqz/np.pi*np.sinh(r)**2


def expt_dA2D_dA2dD(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # <dA2 dA2^d> = Bsqz/pi + <dA2^d dA2>
    return Bsqz/np.pi + expt_dA2dD_dA2D(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t)


def expt_dA2D_dA2D(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # <dA2 dA2>
    return (1 - eta_FC)*Bsqz/np.pi*np.exp(1j*theta)*np.cosh(r)*np.sinh(r)


def expt_dA2dD_dA2dD(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # <dA2^d dA2^d> = <dA2 dA2>^*
    return np.conj(expt_dA2D_dA2D(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t))


def expt_dA1dD_dA2D(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # <dA1^d dA2>
    s1, s2, s12 = _aux(eta_FC)
    return Bsqz/np.pi*s12*np.sinh(r)**2


def expt_dA1D_dA2dD(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # <dA1 dA2^d>   (the vacuum term enters with the minus sign of the coupler: cosh^2 - 1 = sinh^2)
    s1, s2, s12 = _aux(eta_FC)
    return Bsqz/np.pi*s12*(np.cosh(r)**2 - 1)


def expt_dA1D_dA2D(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # <dA1 dA2>
    s1, s2, s12 = _aux(eta_FC)
    return Bsqz/np.pi*s12*np.exp(1j*theta)*np.cosh(r)*np.sinh(r)


def expt_dA1dD_dA2dD(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # <dA1^d dA2^d> = <dA1 dA2>^*
    return np.conj(expt_dA1D_dA2D(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t))


def expt_dA2dD_dA1D(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # <dA2^d dA1> = <dA1^d dA2>^*  (real)
    return np.conj(expt_dA1dD_dA2D(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t))


def expt_dA2D_dA1dD(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # <dA2 dA1^d> = <dA1 dA2^d>^*  (real)
    return np.conj(expt_dA1D_dA2dD(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t))


def expt_dA2D_dA1D(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # <dA2 dA1> = <dA1 dA2>
    return expt_dA1D_dA2D(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t)


def expt_dA2dD_dA1dD(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # <dA2^d dA1^d> = <dA1 dA2>^*
    return np.conj(expt_dA1D_dA2D(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t))


### Constant field components

def bA1D(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # bA1 = sqrt(1 - eta_FC) alpha e^{-i(domega t - phi)}
    s1, s2, s12 = _aux(eta_FC)
    return s2*alpha*np.exp(-1j*(domega*t - phi))


def bA1dD(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # bA1^*
    return np.conj(bA1D(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t))


def bA2D(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # bA2 = -sqrt(eta_FC) alpha e^{-i(domega t - phi)}
    s1, s2, s12 = _aux(eta_FC)
    return -s1*alpha*np.exp(-1j*(domega*t - phi))


def bA2dD(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # bA2^*
    return np.conj(bA2D(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t))


def bA1dD_bA1D(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # bA1^* bA1
    return (1 - eta_FC)*alpha**2


def bA1D_bA1dD(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # bA1 bA1^* = bA1^* bA1
    return bA1dD_bA1D(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t)


def bA1D_bA1D(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # bA1 bA1
    return (1 - eta_FC)*alpha**2*np.exp(-2j*(domega*t - phi))


def bA1dD_bA1dD(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # bA1^* bA1^* = (bA1 bA1)^*
    return np.conj(bA1D_bA1D(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t))


def bA2dD_bA2D(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # bA2^* bA2
    return eta_FC*alpha**2


def bA2D_bA2dD(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # bA2 bA2^* = bA2^* bA2
    return bA2dD_bA2D(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t)


def bA2D_bA2D(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # bA2 bA2
    return eta_FC*alpha**2*np.exp(-2j*(domega*t - phi))


def bA2dD_bA2dD(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # bA2^* bA2^* = (bA2 bA2)^*
    return np.conj(bA2D_bA2D(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t))


def bA1dD_bA2D(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # bA1^* bA2
    s1, s2, s12 = _aux(eta_FC)
    return -s12*alpha**2


def bA1D_bA2dD(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # bA1 bA2^* = (bA1^* bA2)^*
    return np.conj(bA1dD_bA2D(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t))


def bA1D_bA2D(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # bA1 bA2
    s1, s2, s12 = _aux(eta_FC)
    return -s12*alpha**2*np.exp(-2j*(domega*t - phi))


def bA1dD_bA2dD(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # bA1^* bA2^* = (bA1 bA2)^*
    return np.conj(bA1D_bA2D(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t))


def bA2dD_bA1D(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # bA2^* bA1 = (bA1^* bA2)^*
    return np.conj(bA1dD_bA2D(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t))


def bA2D_bA1dD(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # bA2 bA1^* = bA1^* bA2
    return bA1dD_bA2D(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t)


def bA2D_bA1D(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # bA2 bA1 = bA1 bA2
    return bA1D_bA2D(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t)


def bA2dD_bA1dD(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # bA2^* bA1^* = (bA1 bA2)^*
    return np.conj(bA1D_bA2D(alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t))




### PSD  (time averaged and symmetrised in f; general f, with the Pi_Bsqz windows)

def PSD_I1(f, alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # PSD_I1(f)
    P = (alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t)
    n, w = eta_PD1, 2*np.pi*f
    Pi0 = _Pi(w, Bsqz)
    Pis = _Pi(w - domega, Bsqz) + _Pi(w + domega, Bsqz)
    return n*q_e**2*bA1dD_bA1D(*P)*((1 - n*eta_FC)*Pi0 + n*eta_FC/2*np.cosh(2*r)*Pis)


def PSD_I2(f, alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # PSD_I2(f)
    P = (alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t)
    n, w = eta_PD2, 2*np.pi*f
    Pi0 = _Pi(w, Bsqz)
    Pis = _Pi(w - domega, Bsqz) + _Pi(w + domega, Bsqz)
    return n*q_e**2*bA2dD_bA2D(*P)*((1 - n*(1 - eta_FC))*Pi0 + n*(1 - eta_FC)/2*np.cosh(2*r)*Pis)


def PSD_I1I2(f, alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # PSD_I1I2(f) = PSD_I2I1(f)
    w = 2*np.pi*f
    Pi0 = _Pi(w, Bsqz)
    Pis = _Pi(w - domega, Bsqz) + _Pi(w + domega, Bsqz)
    return eta_PD1*eta_PD2*q_e**2*eta_FC*(1 - eta_FC)*alpha**2*(Pi0 - np.cosh(2*r)/2*Pis)


def PSD_Ip(f, alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # PSD_I+(f) = PSD_I1 + PSD_I2 + 2 PSD_I1I2
    P = (f, alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t)
    return PSD_I1(*P) + PSD_I2(*P) + 2*PSD_I1I2(*P)


def PSD_Im(f, alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t):
    # PSD_I-(f) = PSD_I1 + PSD_I2 - 2 PSD_I1I2
    P = (f, alpha, phi, eta_FC, eta_PD1, eta_PD2, r, theta, Bsqz, domega, t)
    return PSD_I1(*P) + PSD_I2(*P) - 2*PSD_I1I2(*P)