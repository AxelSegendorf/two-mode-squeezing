import numpy as np

### Help functions and setup

q_e = 1.602176634e-19 

def _aux(B, omega0, eta_FC1, eta_FC2, eta_FC3):
    """zeta'_1..zeta'_8, S(x) = sin(Bx)/x, C(x) = cos(omega0 x), E(x) = exp(i omega0 x)."""
    a, b, c = eta_FC1, eta_FC2, eta_FC3
    z = np.sqrt([a*b*c, a*(1-b)*c, (1-a)*b*c, (1-a)*(1-b)*c,
                 a*b*(1-c), a*(1-b)*(1-c), (1-a)*b*(1-c), (1-a)*(1-b)*(1-c)])
    S = lambda x: B*np.sinc(B*x/np.pi)  # sin(Bx)/x, -> B for x -> 0 (t.ex. tau = taup)
    C = lambda x: np.cos(omega0*x)
    E = lambda x: np.exp(1j*omega0*x)
    return (*z, S, C, E)


### Expectation values


def expt_dA1dpp_dA1pp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # <dA1''^d dA1''>
    z1, z2, z3, z4, z5, z6, z7, z8, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2, eta_FC3)
    t, tp = tau, taup
    return np.sinh(r)**2/np.pi*(B*(z1**2 + z4**2 + z6**2 + z7**2)
        + 2*S(t)*C(t)*(z1*z4 - z6*z7) + 2*S(tp)*C(tp)*(z1*z6 - z4*z7)
        - 2*S(t+tp)*C(t+tp)*z1*z7 + 2*S(t-tp)*C(t-tp)*z4*z6)


def expt_dA1pp_dA1dpp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # <dA1'' dA1''^d>
    z1, z2, z3, z4, z5, z6, z7, z8, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2, eta_FC3)
    t, tp, ch2 = tau, taup, np.cosh(r)**2
    return (B*(ch2*(z1**2 + z4**2 + z6**2 + z7**2) + z2**2 + z3**2 + z5**2 + z8**2)
        + 2*S(t)*C(t)*(ch2*(z1*z4 - z6*z7) - z2*z3 + z5*z8)
        + 2*S(tp)*C(tp)*(ch2*(z1*z6 - z4*z7) - z2*z5 + z3*z8)
        + 2*S(t+tp)*C(t+tp)*(-ch2*z1*z7 + z3*z5)
        + 2*S(t-tp)*C(t-tp)*(ch2*z4*z6 - z2*z8))/np.pi


def expt_dA1pp_dA1pp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # <dA1'' dA1''>
    z1, z2, z3, z4, z5, z6, z7, z8, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2, eta_FC3)
    t, tp = tau, taup
    return np.exp(1j*theta)*np.cosh(r)*np.sinh(r)/np.pi*(
        B*(z1**2 + E(2*t)*z4**2 + E(2*tp)*z6**2 + E(2*(t+tp))*z7**2)
        + 2*S(t)*(E(t)*z1*z4 - E(t+2*tp)*z6*z7)
        + 2*S(tp)*(E(tp)*z1*z6 - E(2*t+tp)*z4*z7)
        - 2*S(t+tp)*E(t+tp)*z1*z7 + 2*S(t-tp)*E(t+tp)*z4*z6)


def expt_dA1dpp_dA1dpp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # <dA1''^d dA1''^d> = <dA1'' dA1''>^*
    return np.conj(expt_dA1pp_dA1pp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2,
                                    tau, taup, r, theta, B, omega0))


def expt_dA2dpp_dA2pp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # <dA2''^d dA2''>
    z1, z2, z3, z4, z5, z6, z7, z8, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2, eta_FC3)
    t, tp = tau, taup
    return np.sinh(r)**2/np.pi*(B*(z2**2 + z3**2 + z5**2 + z8**2)
        + 2*S(t)*C(t)*(-z2*z3 + z5*z8) + 2*S(tp)*C(tp)*(-z2*z5 + z3*z8)
        + 2*S(t+tp)*C(t+tp)*z3*z5 - 2*S(t-tp)*C(t-tp)*z2*z8)


def expt_dA2pp_dA2dpp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # <dA2'' dA2''^d>
    z1, z2, z3, z4, z5, z6, z7, z8, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2, eta_FC3)
    t, tp, ch2 = tau, taup, np.cosh(r)**2
    return (B*(ch2*(z2**2 + z3**2 + z5**2 + z8**2) + z1**2 + z4**2 + z6**2 + z7**2)
        + 2*S(t)*C(t)*(ch2*(-z2*z3 + z5*z8) + z1*z4 - z6*z7)
        + 2*S(tp)*C(tp)*(ch2*(-z2*z5 + z3*z8) + z1*z6 - z4*z7)
        + 2*S(t+tp)*C(t+tp)*(ch2*z3*z5 - z1*z7)
        + 2*S(t-tp)*C(t-tp)*(-ch2*z2*z8 + z4*z6))/np.pi


def expt_dA2pp_dA2pp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # <dA2'' dA2''>
    z1, z2, z3, z4, z5, z6, z7, z8, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2, eta_FC3)
    t, tp = tau, taup
    return np.exp(1j*theta)*np.cosh(r)*np.sinh(r)/np.pi*(
        B*(z5**2 + E(2*t)*z8**2 + E(2*tp)*z2**2 + E(2*(t+tp))*z3**2)
        + 2*S(t)*(E(t)*z5*z8 - E(t+2*tp)*z2*z3)
        + 2*S(tp)*(-E(tp)*z2*z5 + E(2*t+tp)*z3*z8)
        + 2*S(t+tp)*E(t+tp)*z3*z5 - 2*S(t-tp)*E(t+tp)*z2*z8)


def expt_dA2dpp_dA2dpp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # <dA2''^d dA2''^d> = <dA2'' dA2''>^*
    return np.conj(expt_dA2pp_dA2pp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2,
                                    tau, taup, r, theta, B, omega0))


def expt_dA1dpp_dA2pp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # <dA1''^d dA2''>
    z1, z2, z3, z4, z5, z6, z7, z8, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2, eta_FC3)
    t, tp = tau, taup
    return np.sinh(r)**2/np.pi*(B*(z1*z5 - z2*z6 - z3*z7 + z4*z8)
        + S(t)*(E(t)*(z1*z8 + z3*z6) + E(-t)*(z2*z7 + z4*z5))
        + S(tp)*(E(tp)*(-z1*z2 + z3*z4) + E(-tp)*(z5*z6 - z7*z8))
        + S(t+tp)*(E(t+tp)*z1*z3 - E(-t-tp)*z5*z7)
        + S(t-tp)*(E(t-tp)*z6*z8 - E(tp-t)*z2*z4))


def expt_dA1pp_dA2dpp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # <dA1'' dA2''^d> = <dA1''^d dA2''>^*   (cosh^2(r) - 1 = sinh^2(r))
    return np.conj(expt_dA1dpp_dA2pp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2,
                                     tau, taup, r, theta, B, omega0))


def expt_dA1pp_dA2pp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # <dA1'' dA2''>
    z1, z2, z3, z4, z5, z6, z7, z8, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2, eta_FC3)
    t, tp = tau, taup
    return np.exp(1j*theta)*np.cosh(r)*np.sinh(r)/np.pi*(
        B*(z1*z5 + E(2*t)*z4*z8 - E(2*tp)*z2*z6 - E(2*(t+tp))*z3*z7)
        + S(t)*(E(t)*(z1*z8 + z4*z5) + E(t+2*tp)*(z2*z7 + z3*z6))
        + S(tp)*(E(tp)*(-z1*z2 + z5*z6) + E(2*t+tp)*(z3*z4 - z7*z8))
        + S(t+tp)*E(t+tp)*(z1*z3 - z5*z7) + S(t-tp)*E(t+tp)*(-z2*z4 + z6*z8))


def expt_dA1dpp_dA2dpp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # <dA1''^d dA2''^d> = <dA1'' dA2''>^*
    return np.conj(expt_dA1pp_dA2pp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2,
                                    tau, taup, r, theta, B, omega0))


def expt_dA2dpp_dA1pp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # <dA2''^d dA1''> = <dA1''^d dA2''>^*
    return np.conj(expt_dA1dpp_dA2pp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2,
                                     tau, taup, r, theta, B, omega0))


def expt_dA2pp_dA1dpp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # <dA2'' dA1''^d> = <dA1'' dA2''^d>^*
    return np.conj(expt_dA1pp_dA2dpp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2,
                                     tau, taup, r, theta, B, omega0))


def expt_dA2pp_dA1pp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # <dA2'' dA1''> = <dA1'' dA2''>
    return expt_dA1pp_dA2pp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2,
                            tau, taup, r, theta, B, omega0)


def expt_dA2dpp_dA1dpp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # <dA2''^d dA1''^d> = <dA1'' dA2''>^*
    return np.conj(expt_dA1pp_dA2pp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2,
                                    tau, taup, r, theta, B, omega0))


### Constant field components 

def bA1pp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # bA1''
    z1, z2, z3, z4, z5, z6, z7, z8, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2, eta_FC3)
    return (z3 - z2*E(tau) + E(taup)*(z8 + z5*E(tau)))*alpha*np.exp(1j*phi)


def bA1dpp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # bA1''^*
    return np.conj(bA1pp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0))


def bA2pp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # bA2''
    z1, z2, z3, z4, z5, z6, z7, z8, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2, eta_FC3)
    return (z7 - z6*E(tau) - E(taup)*(z4 + z1*E(tau)))*alpha*np.exp(1j*phi)


def bA2dpp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # bA2''^*
    return np.conj(bA2pp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0))


def bA1dpp_bA1pp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # bA1''^* bA1''
    z1, z2, z3, z4, z5, z6, z7, z8, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2, eta_FC3)
    t, tp = tau, taup
    return alpha**2*(z2**2 + z3**2 + z5**2 + z8**2 + 2*(z5*z8 - z2*z3)*C(t) + 2*(z3*z8 - z2*z5)*C(tp)
        + 2*z3*z5*C(t+tp) - 2*z2*z8*C(t-tp))


def bA1pp_bA1dpp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # bA1'' bA1''^* = bA1''^* bA1''
    return bA1dpp_bA1pp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0)


def bA1pp_bA1pp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # bA1'' bA1''
    z1, z2, z3, z4, z5, z6, z7, z8, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2, eta_FC3)
    t, tp = tau, taup
    return alpha**2*np.exp(2j*phi)*(z3**2 + z2**2*E(2*t) + z8**2*E(2*tp) + z5**2*E(2*(t+tp))
        - 2*z2*z3*E(t) + 2*z3*z8*E(tp) + 2*(z3*z5 - z2*z8)*E(t+tp)
        - 2*z2*z5*E(2*t+tp) + 2*z5*z8*E(t+2*tp))


def bA1dpp_bA1dpp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # bA1''^* bA1''^* = (bA1'' bA1'')^*
    return np.conj(bA1pp_bA1pp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0))


def bA2dpp_bA2pp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # bA2''^* bA2''
    z1, z2, z3, z4, z5, z6, z7, z8, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2, eta_FC3)
    t, tp = tau, taup
    return alpha**2*(z1**2 + z4**2 + z6**2 + z7**2 + 2*(z1*z4 - z6*z7)*C(t) + 2*(z1*z6 - z4*z7)*C(tp)
        - 2*z1*z7*C(t+tp) + 2*z4*z6*C(t-tp))


def bA2pp_bA2dpp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # bA2'' bA2''^* = bA2''^* bA2''
    return bA2dpp_bA2pp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0)


def bA2pp_bA2pp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # bA2'' bA2''
    z1, z2, z3, z4, z5, z6, z7, z8, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2, eta_FC3)
    t, tp = tau, taup
    return alpha**2*np.exp(2j*phi)*(z7**2 + z6**2*E(2*t) + z4**2*E(2*tp) + z1**2*E(2*(t+tp))
        - 2*z6*z7*E(t) - 2*z4*z7*E(tp) + 2*(z4*z6 - z1*z7)*E(t+tp)
        + 2*z1*z6*E(2*t+tp) + 2*z1*z4*E(t+2*tp))


def bA2dpp_bA2dpp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # bA2''^* bA2''^* = (bA2'' bA2'')^*
    return np.conj(bA2pp_bA2pp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0))


def bA1dpp_bA2pp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # bA1''^* bA2''
    z1, z2, z3, z4, z5, z6, z7, z8, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2, eta_FC3)
    t, tp = tau, taup
    return alpha**2*(z2*z6 + z3*z7 - z1*z5 - z4*z8
        - (z1*z8 + z3*z6)*E(t) - (z2*z7 + z4*z5)*E(-t)
        + (z1*z2 - z3*z4)*E(tp) + (z7*z8 - z5*z6)*E(-tp)
        - z1*z3*E(t+tp) + z5*z7*E(-t-tp) - z6*z8*E(t-tp) + z2*z4*E(tp-t))


def bA1pp_bA2dpp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # bA1'' bA2''^* = (bA1''^* bA2'')^*
    return np.conj(bA1dpp_bA2pp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0))


def bA1pp_bA2pp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # bA1'' bA2''
    z1, z2, z3, z4, z5, z6, z7, z8, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2, eta_FC3)
    t, tp = tau, taup
    return alpha**2*np.exp(2j*phi)*(z3*z7 + z2*z6*E(2*t) - z4*z8*E(2*tp) - z1*z5*E(2*(t+tp))
        - (z2*z7 + z3*z6)*E(t) + (z7*z8 - z3*z4)*E(tp)
        + (z2*z4 + z5*z7 - z1*z3 - z6*z8)*E(t+tp)
        + (z1*z2 - z5*z6)*E(2*t+tp) - (z1*z8 + z4*z5)*E(t+2*tp))


def bA1dpp_bA2dpp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # bA1''^* bA2''^* = (bA1'' bA2'')^*
    return np.conj(bA1pp_bA2pp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0))


def bA2dpp_bA1pp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # bA2''^* bA1'' = (bA1''^* bA2'')^*
    return np.conj(bA1dpp_bA2pp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0))


def bA2pp_bA1dpp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # bA2'' bA1''^* = bA1''^* bA2''
    return bA1dpp_bA2pp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0)


def bA2pp_bA1pp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # bA2'' bA1'' = bA1'' bA2''
    return bA1pp_bA2pp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0)


def bA2dpp_bA1dpp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # bA2''^* bA1''^* = (bA1'' bA2'')^*
    return np.conj(bA1pp_bA2pp(alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0))




### PSD 
 
def PSD_I1(f, alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # PSD_I1(f)
    P = (alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0)
    z1, z2, z3, z4, z5, z6, z7, z8, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2, eta_FC3)
    t, tp, h, n = tau, taup, np.cosh(2*r), eta_PD1
    F = lambda x: np.cos(2*np.pi*f*x)
    return n*q_e**2*(bA1dpp_bA1pp(*P)*(1 - n + n*(h*(z1**2 + z4**2 + z6**2 + z7**2) + z2**2 + z3**2 + z5**2 + z8**2)
            + 2*n*(h*(z1*z4 - z6*z7) - z2*z3 + z5*z8)*C(t)*F(t)
            + 2*n*(h*(z1*z6 - z4*z7) - z2*z5 + z3*z8)*C(tp)*F(tp)
            + 2*n*(-h*z1*z7 + z3*z5)*C(t+tp)*F(t+tp)
            + 2*n*(h*z4*z6 - z2*z8)*C(t-tp)*F(t-tp))
        + n*np.sinh(2*r)*np.real(np.exp(1j*theta)*bA1dpp_bA1dpp(*P)*(
            z1**2 + z4**2*E(2*t) + z6**2*E(2*tp) + z7**2*E(2*(t+tp))
            + 2*(z1*z4*E(t) - z6*z7*E(t+2*tp))*F(t) + 2*(z1*z6*E(tp) - z4*z7*E(2*t+tp))*F(tp)
            - 2*z1*z7*E(t+tp)*F(t+tp) + 2*z4*z6*E(t+tp)*F(t-tp))))
 
 
def PSD_I2(f, alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # PSD_I2(f)
    P = (alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0)
    z1, z2, z3, z4, z5, z6, z7, z8, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2, eta_FC3)
    t, tp, h, n = tau, taup, np.cosh(2*r), eta_PD2
    F = lambda x: np.cos(2*np.pi*f*x)
    return n*q_e**2*(bA2dpp_bA2pp(*P)*(1 - n + n*(h*(z2**2 + z3**2 + z5**2 + z8**2) + z1**2 + z4**2 + z6**2 + z7**2)
            + 2*n*(h*(-z2*z3 + z5*z8) + z1*z4 - z6*z7)*C(t)*F(t)
            + 2*n*(h*(-z2*z5 + z3*z8) + z1*z6 - z4*z7)*C(tp)*F(tp)
            + 2*n*(h*z3*z5 - z1*z7)*C(t+tp)*F(t+tp)
            + 2*n*(-h*z2*z8 + z4*z6)*C(t-tp)*F(t-tp))
        + n*np.sinh(2*r)*np.real(np.exp(1j*theta)*bA2dpp_bA2dpp(*P)*(
            z5**2 + z8**2*E(2*t) + z2**2*E(2*tp) + z3**2*E(2*(t+tp))
            + 2*(z5*z8*E(t) - z2*z3*E(t+2*tp))*F(t) + 2*(-z2*z5*E(tp) + z3*z8*E(2*t+tp))*F(tp)
            + 2*z3*z5*E(t+tp)*F(t+tp) - 2*z2*z8*E(t+tp)*F(t-tp))))
 
 
def PSD_I1I2(f, alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # PSD_I1I2(f) = PSD_I2I1(f)
    P = (alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0)
    z1, z2, z3, z4, z5, z6, z7, z8, S, C, E = _aux(B, omega0, eta_FC1, eta_FC2, eta_FC3)
    t, tp = tau, taup
    F = lambda x: np.cos(2*np.pi*f*x)
    return eta_PD1*eta_PD2*q_e**2*(
        np.sinh(2*r)*np.real(np.exp(1j*theta)*bA1dpp_bA2dpp(*P)*(
            z1*z5 + z4*z8*E(2*t) - z2*z6*E(2*tp) - z3*z7*E(2*(t+tp))
            + (E(t)*(z1*z8 + z4*z5) + E(t+2*tp)*(z2*z7 + z3*z6))*F(t)
            + (E(tp)*(-z1*z2 + z5*z6) + E(2*t+tp)*(z3*z4 - z7*z8))*F(tp)
            + E(t+tp)*(z1*z3 - z5*z7)*F(t+tp) + E(t+tp)*(-z2*z4 + z6*z8)*F(t-tp)))
        + 2*np.sinh(r)**2*np.real(bA1dpp_bA2pp(*P)*(
            z1*z5 - z2*z6 - z3*z7 + z4*z8
            + (E(t)*(z2*z7 + z4*z5) + E(-t)*(z1*z8 + z3*z6))*F(t)
            + (E(tp)*(z5*z6 - z7*z8) + E(-tp)*(-z1*z2 + z3*z4))*F(tp)
            + (-E(t+tp)*z5*z7 + E(-t-tp)*z1*z3)*F(t+tp)
            + (-E(t-tp)*z2*z4 + E(tp-t)*z6*z8)*F(t-tp))))
 
 
def PSD_Ip(f, alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # PSD_I+(f) = PSD_I1 + PSD_I2 + 2 PSD_I1I2
    P = (f, alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0)
    return PSD_I1(*P) + PSD_I2(*P) + 2*PSD_I1I2(*P)
 
 
def PSD_Im(f, alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0):
    # PSD_I-(f) = PSD_I1 + PSD_I2 - 2 PSD_I1I2
    P = (f, alpha, phi, eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, tau, taup, r, theta, B, omega0)
    return PSD_I1(*P) + PSD_I2(*P) - 2*PSD_I1I2(*P)