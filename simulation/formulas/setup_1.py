import numpy as np

### Help functions and setup
#
# Parameters (same order in every function):
#   alpha, phi       LO amplitude and phase
#   eta_FC1, eta_FC2 fibre coupler transmissions
#   eta_PD1, eta_PD2 photodiode quantum efficiencies
#   tau              FFR delay
#   r, theta         squeezing parameter and angle
#   B, omega0        measurement bandwidth and carrier frequency

q_e = 1.602176634e-19

def _aux(B, omega0, eta_FC1, eta_FC2):
    """zeta_1..zeta_4, mu, nu, S(x) = sin(Bx)/x, C(x) = cos(omega0 x), E(x) = exp(i omega0 x)."""
    a, b = eta_FC1, eta_FC2
    z1, z2, z3, z4 = np.sqrt([a*b, a*(1-b), (1-a)*b, (1-a)*(1-b)])
    mu = z1*z2 - z3*z4  # = (2 eta_FC1 - 1) sqrt(eta_FC2 (1 - eta_FC2))
    nu = z1*z3 - z2*z4  # = (2 eta_FC2 - 1) sqrt(eta_FC1 (1 - eta_FC1))
    S = lambda x: B*np.sinc(B*x/np.pi)  # sin(Bx)/x, -> B for x -> 0
    C = lambda x: np.cos(omega0*x)
    E = lambda x: np.exp(1j*omega0*x)
    return z1, z2, z3, z4, mu, nu, S, C, E


### Expectation values


def expt_dA1dp_dA1p(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # <dA1'^d dA1'>
    z1, z2, z3, z4, mu, nu, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2)
    t = tau
    return np.sinh(r)**2*(B*(z1**2 + z4**2) + 2*z1*z4*C(t)*S(t))/np.pi


def expt_dA1p_dA1dp(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # <dA1' dA1'^d> = B/pi + <dA1'^d dA1'>
    return B/np.pi + expt_dA1dp_dA1p(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0)


def expt_dA1p_dA1p(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # <dA1' dA1'>
    z1, z2, z3, z4, mu, nu, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2)
    t = tau
    return np.exp(1j*theta)*np.cosh(r)*np.sinh(r)*(B*(z1**2 + z4**2*E(2*t)) + 2*z1*z4*E(t)*S(t))/np.pi


def expt_dA1dp_dA1dp(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # <dA1'^d dA1'^d> = <dA1' dA1'>^*
    return np.conj(expt_dA1p_dA1p(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0))


def expt_dA2dp_dA2p(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # <dA2'^d dA2'>
    z1, z2, z3, z4, mu, nu, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2)
    t = tau
    return np.sinh(r)**2*(B*(z2**2 + z3**2) - 2*z2*z3*C(t)*S(t))/np.pi


def expt_dA2p_dA2dp(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # <dA2' dA2'^d> = B/pi + <dA2'^d dA2'>
    return B/np.pi + expt_dA2dp_dA2p(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0)


def expt_dA2p_dA2p(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # <dA2' dA2'>
    z1, z2, z3, z4, mu, nu, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2)
    t = tau
    return np.exp(1j*theta)*np.cosh(r)*np.sinh(r)*(B*(z2**2 + z3**2*E(2*t)) - 2*z2*z3*E(t)*S(t))/np.pi


def expt_dA2dp_dA2dp(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # <dA2'^d dA2'^d> = <dA2' dA2'>^*
    return np.conj(expt_dA2p_dA2p(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0))


def expt_dA1dp_dA2p(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # <dA1'^d dA2'>
    z1, z2, z3, z4, mu, nu, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2)
    t = tau
    return np.sinh(r)**2/np.pi*(B*mu - (z1*z3*E(t) - z2*z4*E(-t))*S(t))


def expt_dA1p_dA2dp(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # <dA1' dA2'^d> = <dA1'^d dA2'>^*   (the ports commute, no B/pi term)
    return np.conj(expt_dA1dp_dA2p(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0))


def expt_dA1p_dA2p(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # <dA1' dA2'>
    z1, z2, z3, z4, mu, nu, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2)
    t = tau
    return np.exp(1j*theta)*np.cosh(r)*np.sinh(r)*(B*(z1*z2 - z3*z4*E(2*t)) - nu*E(t)*S(t))/np.pi


def expt_dA1dp_dA2dp(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # <dA1'^d dA2'^d> = <dA1' dA2'>^*
    return np.conj(expt_dA1p_dA2p(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0))


def expt_dA2dp_dA1p(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # <dA2'^d dA1'> = <dA1'^d dA2'>^*
    return np.conj(expt_dA1dp_dA2p(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0))


def expt_dA2p_dA1dp(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # <dA2' dA1'^d> = <dA1' dA2'^d>^*
    return np.conj(expt_dA1p_dA2dp(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0))


def expt_dA2p_dA1p(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # <dA2' dA1'> = <dA1' dA2'>
    return expt_dA1p_dA2p(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0)


def expt_dA2dp_dA1dp(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # <dA2'^d dA1'^d> = <dA1' dA2'>^*
    return np.conj(expt_dA1p_dA2p(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0))


### Constant field components

def bA1p(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # bA1'
    z1, z2, z3, z4, mu, nu, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2)
    return (z3 - z2*E(tau))*alpha*np.exp(1j*phi)


def bA1dp(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # bA1'^*
    return np.conj(bA1p(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0))


def bA2p(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # bA2'
    z1, z2, z3, z4, mu, nu, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2)
    return (z4 + z1*E(tau))*alpha*np.exp(1j*phi)


def bA2dp(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # bA2'^*
    return np.conj(bA2p(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0))


def bA1dp_bA1p(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # bA1'^* bA1'
    z1, z2, z3, z4, mu, nu, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2)
    return alpha**2*(z2**2 + z3**2 - 2*z2*z3*C(tau))


def bA1p_bA1dp(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # bA1' bA1'^* = bA1'^* bA1'
    return bA1dp_bA1p(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0)


def bA1p_bA1p(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # bA1' bA1'
    z1, z2, z3, z4, mu, nu, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2)
    t = tau
    return alpha**2*np.exp(2j*phi)*(z3**2 + z2**2*E(2*t) - 2*z2*z3*E(t))


def bA1dp_bA1dp(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # bA1'^* bA1'^* = (bA1' bA1')^*
    return np.conj(bA1p_bA1p(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0))


def bA2dp_bA2p(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # bA2'^* bA2'
    z1, z2, z3, z4, mu, nu, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2)
    return alpha**2*(z1**2 + z4**2 + 2*z1*z4*C(tau))


def bA2p_bA2dp(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # bA2' bA2'^* = bA2'^* bA2'
    return bA2dp_bA2p(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0)


def bA2p_bA2p(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # bA2' bA2'
    z1, z2, z3, z4, mu, nu, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2)
    t = tau
    return alpha**2*np.exp(2j*phi)*(z4**2 + z1**2*E(2*t) + 2*z1*z4*E(t))


def bA2dp_bA2dp(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # bA2'^* bA2'^* = (bA2' bA2')^*
    return np.conj(bA2p_bA2p(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0))


def bA1dp_bA2p(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # bA1'^* bA2'
    z1, z2, z3, z4, mu, nu, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2)
    t = tau
    return alpha**2*(z3*z4 - z1*z2 + z1*z3*E(t) - z2*z4*E(-t))


def bA1p_bA2dp(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # bA1' bA2'^* = (bA1'^* bA2')^*
    return np.conj(bA1dp_bA2p(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0))


def bA1p_bA2p(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # bA1' bA2'
    z1, z2, z3, z4, mu, nu, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2)
    t = tau
    return alpha**2*np.exp(2j*phi)*(z3*z4 + nu*E(t) - z1*z2*E(2*t))


def bA1dp_bA2dp(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # bA1'^* bA2'^* = (bA1' bA2')^*
    return np.conj(bA1p_bA2p(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0))


def bA2dp_bA1p(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # bA2'^* bA1' = (bA1'^* bA2')^*
    return np.conj(bA1dp_bA2p(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0))


def bA2p_bA1dp(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # bA2' bA1'^* = bA1'^* bA2'
    return bA1dp_bA2p(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0)


def bA2p_bA1p(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # bA2' bA1' = bA1' bA2'
    return bA1p_bA2p(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0)


def bA2dp_bA1dp(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # bA2'^* bA1'^* = (bA1' bA2')^*
    return np.conj(bA1p_bA2p(alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0))




### PSD

def PSD_I1(f, alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # PSD_I1(f)
    P = (alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0)
    z1, z2, z3, z4, mu, nu, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2)
    t, n = tau, eta_PD1
    F = lambda x: np.cos(2*np.pi*f*x)
    return n*q_e**2*(bA1dp_bA1p(*P)*(1 + 2*n*np.sinh(r)**2*(z1**2 + z4**2 + 2*z1*z4*C(t)*F(t)))
        + n*np.sinh(2*r)*np.real(np.exp(1j*theta)*bA1dp_bA1dp(*P)*(
            z1**2 + z4**2*E(2*t) + 2*z1*z4*E(t)*F(t))))


def PSD_I2(f, alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # PSD_I2(f)
    P = (alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0)
    z1, z2, z3, z4, mu, nu, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2)
    t, n = tau, eta_PD2
    F = lambda x: np.cos(2*np.pi*f*x)
    return n*q_e**2*(bA2dp_bA2p(*P)*(1 + 2*n*np.sinh(r)**2*(z2**2 + z3**2 - 2*z2*z3*C(t)*F(t)))
        + n*np.sinh(2*r)*np.real(np.exp(1j*theta)*bA2dp_bA2dp(*P)*(
            z2**2 + z3**2*E(2*t) - 2*z2*z3*E(t)*F(t))))


def PSD_I1I2(f, alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # PSD_I1I2(f) = PSD_I2I1(f)
    P = (alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0)
    z1, z2, z3, z4, mu, nu, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2)
    t = tau
    F = lambda x: np.cos(2*np.pi*f*x)
    return eta_PD1*eta_PD2*q_e**2*(
        np.sinh(2*r)*np.real(np.exp(1j*theta)*bA1dp_bA2dp(*P)*(
            z1*z2 - z3*z4*E(2*t) - nu*E(t)*F(t)))
        + 2*np.sinh(r)**2*np.real(bA1dp_bA2p(*P)*(
            mu + (z2*z4*E(t) - z1*z3*E(-t))*F(t))))


def PSD_Ip(f, alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # PSD_I+(f) = PSD_I1 + PSD_I2 + 2 PSD_I1I2
    P = (f, alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0)
    return PSD_I1(*P) + PSD_I2(*P) + 2*PSD_I1I2(*P)


def PSD_Im(f, alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0):
    # PSD_I-(f) = PSD_I1 + PSD_I2 - 2 PSD_I1I2
    P = (f, alpha, phi, eta_FC1, eta_FC2, eta_PD1, eta_PD2, tau, r, theta, B, omega0)
    return PSD_I1(*P) + PSD_I2(*P) - 2*PSD_I1I2(*P)