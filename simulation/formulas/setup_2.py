import numpy as np

### Help functions and setup
#
# Parameters (same order in every function):
#   alpha, phi       LO amplitude and phase (before the injection beam splitter)
#   eta_FC1, eta_FC2 fibre coupler transmissions
#   eta_inj          injection transmission of the second squeezer into the LO arm
#   eta_PD1, eta_PD2 photodiode quantum efficiencies
#   tau              FFR delay
#   r, theta         squeezing parameter and angle of the main squeezer (SQZ)
#   rp, thetap       squeezing parameter and angle of the squeezer injected into the LO (r', theta')
#   B, omega0        measurement bandwidth and carrier frequency
#
# The LO amplitude after injection is beta = sqrt(1 - eta_inj) * alpha.

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


def _beta(alpha, eta_inj):
    # beta = sqrt(1 - eta_inj) alpha
    return np.sqrt(1 - eta_inj)*alpha


### Expectation values


def expt_dAS1dp_dAS1p(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # <dAS1'^d dAS1'>
    z1, z2, z3, z4, mu, nu, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2)
    t = tau
    return (np.sinh(r)**2*(B*(z1**2 + z4**2) + 2*z1*z4*C(t)*S(t))
        + eta_inj*np.sinh(rp)**2*(B*(z2**2 + z3**2) - 2*z2*z3*C(t)*S(t)))/np.pi


def expt_dAS1p_dAS1dp(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # <dAS1' dAS1'^d> = B/pi + <dAS1'^d dAS1'>
    return B/np.pi + expt_dAS1dp_dAS1p(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2,
                                       tau, r, theta, rp, thetap, B, omega0)


def expt_dAS1p_dAS1p(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # <dAS1' dAS1'>
    z1, z2, z3, z4, mu, nu, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2)
    t = tau
    return (np.exp(1j*theta)*np.cosh(r)*np.sinh(r)*(B*(z1**2 + z4**2*E(2*t)) + 2*z1*z4*E(t)*S(t))
        + eta_inj*np.exp(1j*thetap)*np.cosh(rp)*np.sinh(rp)*(B*(z3**2 + z2**2*E(2*t)) - 2*z2*z3*E(t)*S(t)))/np.pi


def expt_dAS1dp_dAS1dp(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # <dAS1'^d dAS1'^d> = <dAS1' dAS1'>^*
    return np.conj(expt_dAS1p_dAS1p(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2,
                                    tau, r, theta, rp, thetap, B, omega0))


def expt_dAS2dp_dAS2p(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # <dAS2'^d dAS2'>
    z1, z2, z3, z4, mu, nu, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2)
    t = tau
    return (np.sinh(r)**2*(B*(z2**2 + z3**2) - 2*z2*z3*C(t)*S(t))
        + eta_inj*np.sinh(rp)**2*(B*(z1**2 + z4**2) + 2*z1*z4*C(t)*S(t)))/np.pi


def expt_dAS2p_dAS2dp(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # <dAS2' dAS2'^d> = B/pi + <dAS2'^d dAS2'>
    return B/np.pi + expt_dAS2dp_dAS2p(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2,
                                       tau, r, theta, rp, thetap, B, omega0)


def expt_dAS2p_dAS2p(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # <dAS2' dAS2'>
    z1, z2, z3, z4, mu, nu, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2)
    t = tau
    return (np.exp(1j*theta)*np.cosh(r)*np.sinh(r)*(B*(z2**2 + z3**2*E(2*t)) - 2*z2*z3*E(t)*S(t))
        + eta_inj*np.exp(1j*thetap)*np.cosh(rp)*np.sinh(rp)*(B*(z4**2 + z1**2*E(2*t)) + 2*z1*z4*E(t)*S(t)))/np.pi


def expt_dAS2dp_dAS2dp(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # <dAS2'^d dAS2'^d> = <dAS2' dAS2'>^*
    return np.conj(expt_dAS2p_dAS2p(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2,
                                    tau, r, theta, rp, thetap, B, omega0))


def expt_dAS1dp_dAS2p(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # <dAS1'^d dAS2'>
    z1, z2, z3, z4, mu, nu, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2)
    t = tau
    return (np.sinh(r)**2 - eta_inj*np.sinh(rp)**2)/np.pi*(B*mu - (z1*z3*E(t) - z2*z4*E(-t))*S(t))


def expt_dAS1p_dAS2dp(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # <dAS1' dAS2'^d> = <dAS1'^d dAS2'>^*   (the 1's from <b b^d> cancel)
    return np.conj(expt_dAS1dp_dAS2p(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2,
                                     tau, r, theta, rp, thetap, B, omega0))


def expt_dAS1p_dAS2p(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # <dAS1' dAS2'>
    z1, z2, z3, z4, mu, nu, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2)
    t = tau
    return (np.exp(1j*theta)*np.cosh(r)*np.sinh(r)*(B*(z1*z2 - z3*z4*E(2*t)) - nu*E(t)*S(t))
        + eta_inj*np.exp(1j*thetap)*np.cosh(rp)*np.sinh(rp)*(B*(z3*z4 - z1*z2*E(2*t)) + nu*E(t)*S(t)))/np.pi


def expt_dAS1dp_dAS2dp(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # <dAS1'^d dAS2'^d> = <dAS1' dAS2'>^*
    return np.conj(expt_dAS1p_dAS2p(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2,
                                    tau, r, theta, rp, thetap, B, omega0))


def expt_dAS2dp_dAS1p(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # <dAS2'^d dAS1'> = <dAS1'^d dAS2'>^*
    return np.conj(expt_dAS1dp_dAS2p(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2,
                                     tau, r, theta, rp, thetap, B, omega0))


def expt_dAS2p_dAS1dp(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # <dAS2' dAS1'^d> = <dAS1' dAS2'^d>^*
    return np.conj(expt_dAS1p_dAS2dp(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2,
                                     tau, r, theta, rp, thetap, B, omega0))


def expt_dAS2p_dAS1p(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # <dAS2' dAS1'> = <dAS1' dAS2'>
    return expt_dAS1p_dAS2p(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2,
                            tau, r, theta, rp, thetap, B, omega0)


def expt_dAS2dp_dAS1dp(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # <dAS2'^d dAS1'^d> = <dAS1' dAS2'>^*
    return np.conj(expt_dAS1p_dAS2p(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2,
                                    tau, r, theta, rp, thetap, B, omega0))


### Constant field components

def bAS1p(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # bAS1'
    z1, z2, z3, z4, mu, nu, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2)
    return (z3 - z2*E(tau))*_beta(alpha, eta_inj)*np.exp(1j*phi)


def bAS1dp(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # bAS1'^*
    return np.conj(bAS1p(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0))


def bAS2p(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # bAS2'
    z1, z2, z3, z4, mu, nu, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2)
    return (z4 + z1*E(tau))*_beta(alpha, eta_inj)*np.exp(1j*phi)


def bAS2dp(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # bAS2'^*
    return np.conj(bAS2p(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0))


def bAS1dp_bAS1p(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # bAS1'^* bAS1'
    z1, z2, z3, z4, mu, nu, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2)
    return _beta(alpha, eta_inj)**2*(z2**2 + z3**2 - 2*z2*z3*C(tau))


def bAS1p_bAS1dp(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # bAS1' bAS1'^* = bAS1'^* bAS1'
    return bAS1dp_bAS1p(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0)


def bAS1p_bAS1p(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # bAS1' bAS1'
    z1, z2, z3, z4, mu, nu, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2)
    t = tau
    return _beta(alpha, eta_inj)**2*np.exp(2j*phi)*(z3**2 + z2**2*E(2*t) - 2*z2*z3*E(t))


def bAS1dp_bAS1dp(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # bAS1'^* bAS1'^* = (bAS1' bAS1')^*
    return np.conj(bAS1p_bAS1p(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0))


def bAS2dp_bAS2p(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # bAS2'^* bAS2'
    z1, z2, z3, z4, mu, nu, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2)
    return _beta(alpha, eta_inj)**2*(z1**2 + z4**2 + 2*z1*z4*C(tau))


def bAS2p_bAS2dp(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # bAS2' bAS2'^* = bAS2'^* bAS2'
    return bAS2dp_bAS2p(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0)


def bAS2p_bAS2p(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # bAS2' bAS2'
    z1, z2, z3, z4, mu, nu, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2)
    t = tau
    return _beta(alpha, eta_inj)**2*np.exp(2j*phi)*(z4**2 + z1**2*E(2*t) + 2*z1*z4*E(t))


def bAS2dp_bAS2dp(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # bAS2'^* bAS2'^* = (bAS2' bAS2')^*
    return np.conj(bAS2p_bAS2p(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0))


def bAS1dp_bAS2p(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # bAS1'^* bAS2'
    z1, z2, z3, z4, mu, nu, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2)
    t = tau
    return _beta(alpha, eta_inj)**2*(z3*z4 - z1*z2 + z1*z3*E(t) - z2*z4*E(-t))


def bAS1p_bAS2dp(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # bAS1' bAS2'^* = (bAS1'^* bAS2')^*
    return np.conj(bAS1dp_bAS2p(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0))


def bAS1p_bAS2p(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # bAS1' bAS2'
    z1, z2, z3, z4, mu, nu, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2)
    t = tau
    return _beta(alpha, eta_inj)**2*np.exp(2j*phi)*(z3*z4 + nu*E(t) - z1*z2*E(2*t))


def bAS1dp_bAS2dp(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # bAS1'^* bAS2'^* = (bAS1' bAS2')^*
    return np.conj(bAS1p_bAS2p(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0))


def bAS2dp_bAS1p(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # bAS2'^* bAS1' = (bAS1'^* bAS2')^*
    return np.conj(bAS1dp_bAS2p(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0))


def bAS2p_bAS1dp(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # bAS2' bAS1'^* = bAS1'^* bAS2'
    return bAS1dp_bAS2p(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0)


def bAS2p_bAS1p(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # bAS2' bAS1' = bAS1' bAS2'
    return bAS1p_bAS2p(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0)


def bAS2dp_bAS1dp(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # bAS2'^* bAS1'^* = (bAS1' bAS2')^*
    return np.conj(bAS1p_bAS2p(alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0))




### PSD

def PSD_I1(f, alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # PSD_I1(f)
    P = (alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0)
    z1, z2, z3, z4, mu, nu, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2)
    t, n = tau, eta_PD1
    F = lambda x: np.cos(2*np.pi*f*x)
    return n*q_e**2*(bAS1dp_bAS1p(*P)*(1 + 2*n*np.sinh(r)**2*(z1**2 + z4**2 + 2*z1*z4*C(t)*F(t))
            + 2*n*eta_inj*np.sinh(rp)**2*(z2**2 + z3**2 - 2*z2*z3*C(t)*F(t)))
        + n*np.real(bAS1dp_bAS1dp(*P)*(
            np.exp(1j*theta)*np.sinh(2*r)*(z1**2 + z4**2*E(2*t) + 2*z1*z4*E(t)*F(t))
            + eta_inj*np.exp(1j*thetap)*np.sinh(2*rp)*(z3**2 + z2**2*E(2*t) - 2*z2*z3*E(t)*F(t)))))


def PSD_I2(f, alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # PSD_I2(f)
    P = (alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0)
    z1, z2, z3, z4, mu, nu, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2)
    t, n = tau, eta_PD2
    F = lambda x: np.cos(2*np.pi*f*x)
    return n*q_e**2*(bAS2dp_bAS2p(*P)*(1 + 2*n*np.sinh(r)**2*(z2**2 + z3**2 - 2*z2*z3*C(t)*F(t))
            + 2*n*eta_inj*np.sinh(rp)**2*(z1**2 + z4**2 + 2*z1*z4*C(t)*F(t)))
        + n*np.real(bAS2dp_bAS2dp(*P)*(
            np.exp(1j*theta)*np.sinh(2*r)*(z2**2 + z3**2*E(2*t) - 2*z2*z3*E(t)*F(t))
            + eta_inj*np.exp(1j*thetap)*np.sinh(2*rp)*(z4**2 + z1**2*E(2*t) + 2*z1*z4*E(t)*F(t)))))


def PSD_I1I2(f, alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # PSD_I1I2(f) = PSD_I2I1(f)
    P = (alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0)
    z1, z2, z3, z4, mu, nu, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2)
    t = tau
    F = lambda x: np.cos(2*np.pi*f*x)
    return eta_PD1*eta_PD2*q_e**2*(
        np.sinh(2*r)*np.real(np.exp(1j*theta)*bAS1dp_bAS2dp(*P)*(
            z1*z2 - z3*z4*E(2*t) - nu*E(t)*F(t)))
        + eta_inj*np.sinh(2*rp)*np.real(np.exp(1j*thetap)*bAS1dp_bAS2dp(*P)*(
            z3*z4 - z1*z2*E(2*t) + nu*E(t)*F(t)))
        + 2*(np.sinh(r)**2 - eta_inj*np.sinh(rp)**2)*np.real(bAS1dp_bAS2p(*P)*(
            mu + (z2*z4*E(t) - z1*z3*E(-t))*F(t))))


def PSD_Ip(f, alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # PSD_I+(f) = PSD_I1 + PSD_I2 + 2 PSD_I1I2
    P = (f, alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0)
    return PSD_I1(*P) + PSD_I2(*P) + 2*PSD_I1I2(*P)


def PSD_Im(f, alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0):
    # PSD_I-(f) = PSD_I1 + PSD_I2 - 2 PSD_I1I2
    P = (f, alpha, phi, eta_FC1, eta_FC2, eta_inj, eta_PD1, eta_PD2, tau, r, theta, rp, thetap, B, omega0)
    return PSD_I1(*P) + PSD_I2(*P) - 2*PSD_I1I2(*P)