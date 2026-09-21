"""Functions implementing formulas for photocurrent signals of measurement setup 2."""

import numpy as np


def unnormalized_sinc(x):
    """sin(x)/x with sinc(0) = 1."""
    return np.sinc(x / np.pi)


## Prefactors

def zeta_1(eta_FC1, eta_FC2):
    return np.sqrt(eta_FC1 * eta_FC2)


def zeta_2(eta_FC1, eta_FC2):
    return np.sqrt(eta_FC1 * (1 - eta_FC2))


def zeta_3(eta_FC1, eta_FC2):
    return np.sqrt((1 - eta_FC1) * eta_FC2)


def zeta_4(eta_FC1, eta_FC2):
    return np.sqrt((1 - eta_FC1) * (1 - eta_FC2))


def _zeta_all(eta_FC1, eta_FC2):
    return (
        zeta_1(eta_FC1, eta_FC2),
        zeta_2(eta_FC1, eta_FC2),
        zeta_3(eta_FC1, eta_FC2),
        zeta_4(eta_FC1, eta_FC2),
    )


def _mu(eta_FC1, eta_FC2):
    z1, z2, z3, z4 = _zeta_all(eta_FC1, eta_FC2)
    return z1 * z2 - z3 * z4


def _nu(eta_FC1, eta_FC2):
    z1, z2, z3, z4 = _zeta_all(eta_FC1, eta_FC2)
    return z1 * z3 - z2 * z4


## Mean fields

def A_bar_1(eta_FC1, eta_FC2, alpha, phi, omega_0, tau):
    z1, z2, z3, _ = _zeta_all(eta_FC1, eta_FC2)
    return (z3 - z2 * np.exp(1j * omega_0 * tau)) * alpha * np.exp(1j * phi)


def A_bar_2(eta_FC1, eta_FC2, alpha, phi, omega_0, tau):
    z1, _, _, z4 = _zeta_all(eta_FC1, eta_FC2)
    return (z4 + z1 * np.exp(1j * omega_0 * tau)) * alpha * np.exp(1j * phi)


## Fluctuation correlators

def expt_dA1d_dA1(eta_FC1, eta_FC2, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau):
    """Expectation value of delta A_1^dagger with delta A_1."""
    z1, z2, z3, z4 = _zeta_all(eta_FC1, eta_FC2)
    Bp = B / np.pi
    s = unnormalized_sinc(B * tau)
    sh2 = np.sinh(r) ** 2
    sh2p = np.sinh(r_prime) ** 2
    a_bracket = z1**2 + z4**2 + 2 * z1 * z4 * np.cos(omega_0 * tau) * s
    b_bracket = z2**2 + z3**2 - 2 * z2 * z3 * np.cos(omega_0 * tau) * s
    return Bp * (a_bracket * sh2 + b_bracket * eta_inj * sh2p)


def expt_dA1_dA1d(eta_FC1, eta_FC2, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau):
    """Expectation value of delta A_1 with delta A_1^dagger."""
    return B / np.pi + expt_dA1d_dA1(
        eta_FC1, eta_FC2, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau
    )


def expt_dA1_dA1(eta_FC1, eta_FC2, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau):
    """Expectation value of delta A_1 with delta A_1."""
    z1, z2, z3, z4 = _zeta_all(eta_FC1, eta_FC2)
    Bp = B / np.pi
    s = unnormalized_sinc(B * tau)
    chsh = np.cosh(r) * np.sinh(r)
    chshp = np.cosh(r_prime) * np.sinh(r_prime)
    a_bracket = z1**2 + z4**2 * np.exp(2j * omega_0 * tau) + 2 * z1 * z4 * np.exp(1j * omega_0 * tau) * s
    b_bracket = z3**2 + z2**2 * np.exp(2j * omega_0 * tau) - 2 * z2 * z3 * np.exp(1j * omega_0 * tau) * s
    return Bp * (
        np.exp(1j * theta) * chsh * a_bracket
        + eta_inj * np.exp(1j * theta_prime) * chshp * b_bracket
    )


def expt_dA1d_dA1d(eta_FC1, eta_FC2, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau):
    """Expectation value of delta A_1^dagger with delta A_1^dagger."""
    return np.conj(
        expt_dA1_dA1(eta_FC1, eta_FC2, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau)
    )


def expt_dA2d_dA2(eta_FC1, eta_FC2, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau):
    """Expectation value of delta A_2^dagger with delta A_2."""
    z1, z2, z3, z4 = _zeta_all(eta_FC1, eta_FC2)
    Bp = B / np.pi
    s = unnormalized_sinc(B * tau)
    sh2 = np.sinh(r) ** 2
    sh2p = np.sinh(r_prime) ** 2
    a_bracket = z2**2 + z3**2 - 2 * z2 * z3 * np.cos(omega_0 * tau) * s
    b_bracket = z1**2 + z4**2 + 2 * z1 * z4 * np.cos(omega_0 * tau) * s
    return Bp * (a_bracket * sh2 + b_bracket * eta_inj * sh2p)


def expt_dA2_dA2d(eta_FC1, eta_FC2, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau):
    """Expectation value of delta A_2 with delta A_2^dagger."""
    return B / np.pi + expt_dA2d_dA2(
        eta_FC1, eta_FC2, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau
    )


def expt_dA2_dA2(eta_FC1, eta_FC2, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau):
    """Expectation value of delta A_2 with delta A_2."""
    z1, z2, z3, z4 = _zeta_all(eta_FC1, eta_FC2)
    Bp = B / np.pi
    s = unnormalized_sinc(B * tau)
    chsh = np.cosh(r) * np.sinh(r)
    chshp = np.cosh(r_prime) * np.sinh(r_prime)
    a_bracket = z2**2 + z3**2 * np.exp(2j * omega_0 * tau) - 2 * z2 * z3 * np.exp(1j * omega_0 * tau) * s
    b_bracket = z4**2 + z1**2 * np.exp(2j * omega_0 * tau) + 2 * z1 * z4 * np.exp(1j * omega_0 * tau) * s
    return Bp * (
        np.exp(1j * theta) * chsh * a_bracket
        + eta_inj * np.exp(1j * theta_prime) * chshp * b_bracket
    )


def expt_dA2d_dA2d(eta_FC1, eta_FC2, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau):
    """Expectation value of delta A_2^dagger with delta A_2^dagger."""
    return np.conj(
        expt_dA2_dA2(eta_FC1, eta_FC2, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau)
    )


def expt_dA1d_dA2(eta_FC1, eta_FC2, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau):
    """Expectation value of delta A_1^dagger with delta A_2."""
    z1, z2, z3, z4 = _zeta_all(eta_FC1, eta_FC2)
    Bp = B / np.pi
    s = unnormalized_sinc(B * tau)
    mu = _mu(eta_FC1, eta_FC2)
    sh2 = np.sinh(r) ** 2
    sh2p = np.sinh(r_prime) ** 2
    bracket = mu - (z1 * z3 * np.exp(1j * omega_0 * tau) - z2 * z4 * np.exp(-1j * omega_0 * tau)) * s
    return Bp * bracket * (sh2 - eta_inj * sh2p)


def expt_dA1_dA2d(eta_FC1, eta_FC2, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau):
    """Expectation value of delta A_1 with delta A_2^dagger."""
    return np.conj(
        expt_dA1d_dA2(eta_FC1, eta_FC2, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau)
    )


def expt_dA1_dA2(eta_FC1, eta_FC2, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau):
    """Expectation value of delta A_1 with delta A_2."""
    z1, z2, z3, z4 = _zeta_all(eta_FC1, eta_FC2)
    Bp = B / np.pi
    s = unnormalized_sinc(B * tau)
    nu = _nu(eta_FC1, eta_FC2)
    chsh = np.cosh(r) * np.sinh(r)
    chshp = np.cosh(r_prime) * np.sinh(r_prime)
    a_bracket = z1 * z2 - z3 * z4 * np.exp(2j * omega_0 * tau) - nu * np.exp(1j * omega_0 * tau) * s
    b_bracket = z3 * z4 - z1 * z2 * np.exp(2j * omega_0 * tau) + nu * np.exp(1j * omega_0 * tau) * s
    return Bp * (
        np.exp(1j * theta) * chsh * a_bracket
        + eta_inj * np.exp(1j * theta_prime) * chshp * b_bracket
    )


def expt_dA1d_dA2d(eta_FC1, eta_FC2, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau):
    """Expectation value of delta A_1^dagger with delta A_2^dagger."""
    return np.conj(
        expt_dA1_dA2(eta_FC1, eta_FC2, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau)
    )


def expt_dA2d_dA1(eta_FC1, eta_FC2, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau):
    """Expectation value of delta A_2^dagger with delta A_1."""
    return expt_dA1_dA2d(
        eta_FC1, eta_FC2, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau
    )


def expt_dA2_dA1d(eta_FC1, eta_FC2, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau):
    """Expectation value of delta A_2 with delta A_1^dagger."""
    return np.conj(
        expt_dA1_dA2d(eta_FC1, eta_FC2, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau)
    )


def expt_dA2_dA1(eta_FC1, eta_FC2, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau):
    """Expectation value of delta A_2 with delta A_1."""
    return expt_dA1_dA2(
        eta_FC1, eta_FC2, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau
    )


def expt_dA2d_dA1d(eta_FC1, eta_FC2, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau):
    """Expectation value of delta A_2^dagger with delta A_1^dagger."""
    return np.conj(
        expt_dA1_dA2(eta_FC1, eta_FC2, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau)
    )


## Main signals

def expt_I_1(eta_FC1, eta_FC2, eta_PD1, eta_PD2, e, alpha, phi, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau):
    """Expectation value of signal 1."""
    a_bar = A_bar_1(eta_FC1, eta_FC2, alpha, phi, omega_0, tau)
    c_d_d = expt_dA1d_dA1(eta_FC1, eta_FC2, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau)
    return np.real(eta_PD1 * e * (np.conj(a_bar) * a_bar + c_d_d))


def expt_I_2(eta_FC1, eta_FC2, eta_PD1, eta_PD2, e, alpha, phi, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau):
    """Expectation value of signal 2."""
    a_bar = A_bar_2(eta_FC1, eta_FC2, alpha, phi, omega_0, tau)
    c_d_d = expt_dA2d_dA2(eta_FC1, eta_FC2, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau)
    return np.real(eta_PD2 * e * (np.conj(a_bar) * a_bar + c_d_d))


def var_I_1(eta_FC1, eta_FC2, eta_PD1, eta_PD2, e, alpha, phi, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau):
    """Variance of signal 1."""
    pref = (eta_PD1 * e) ** 2
    a_bar = A_bar_1(eta_FC1, eta_FC2, alpha, phi, omega_0, tau)
    c_d_d = expt_dA1d_dA1(eta_FC1, eta_FC2, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau)
    c_d_dag = expt_dA1_dA1d(eta_FC1, eta_FC2, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau)
    c_dd = expt_dA1_dA1(eta_FC1, eta_FC2, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau)
    term1 = np.conj(a_bar) * a_bar * (c_d_d + c_d_dag)
    term2 = 2 * np.real(np.conj(a_bar) ** 2 * c_dd)
    term3 = np.abs(c_dd) ** 2
    term4 = c_d_d * c_d_dag
    return np.real(pref * (term1 + term2 + term3 + term4))


def var_I_2(eta_FC1, eta_FC2, eta_PD1, eta_PD2, e, alpha, phi, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau):
    """Variance of signal 2."""
    pref = (eta_PD2 * e) ** 2
    a_bar = A_bar_2(eta_FC1, eta_FC2, alpha, phi, omega_0, tau)
    c_d_d = expt_dA2d_dA2(eta_FC1, eta_FC2, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau)
    c_d_dag = expt_dA2_dA2d(eta_FC1, eta_FC2, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau)
    c_dd = expt_dA2_dA2(eta_FC1, eta_FC2, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau)
    term1 = np.conj(a_bar) * a_bar * (c_d_d + c_d_dag)
    term2 = 2 * np.real(np.conj(a_bar) ** 2 * c_dd)
    term3 = np.abs(c_dd) ** 2
    term4 = c_d_d * c_d_dag
    return np.real(pref * (term1 + term2 + term3 + term4))


def cov_I_1_I_2(eta_FC1, eta_FC2, eta_PD1, eta_PD2, e, alpha, phi, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau):
    """Covariance of signals 1 and 2."""
    pref = eta_PD1 * eta_PD2 * e**2
    a_bar_1 = A_bar_1(eta_FC1, eta_FC2, alpha, phi, omega_0, tau)
    a_bar_2 = A_bar_2(eta_FC1, eta_FC2, alpha, phi, omega_0, tau)
    c_dA1_dA2d = expt_dA1_dA2d(eta_FC1, eta_FC2, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau)
    c_dA1_dA2 = expt_dA1_dA2(eta_FC1, eta_FC2, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau)
    c_dA1d_dA2 = expt_dA1d_dA2(eta_FC1, eta_FC2, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau)
    c_dA1d_dA2d = expt_dA1d_dA2d(eta_FC1, eta_FC2, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau)
    value = pref * (
        2 * np.real(np.conj(a_bar_1) * a_bar_2 * c_dA1_dA2d)
        + 2 * np.real(np.conj(a_bar_1) * np.conj(a_bar_2) * c_dA1_dA2)
        + c_dA1d_dA2d * c_dA1_dA2
        + c_dA1d_dA2 * c_dA1_dA2d
    )
    return np.real(value)


def expt_I_diff(eta_FC1, eta_FC2, eta_PD1, eta_PD2, e, alpha, phi, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau):
    """Expectation value of difference signal (-).

    Not assuming equal quantum efficiencies of detectors.
    """
    args = (eta_FC1, eta_FC2, eta_PD1, eta_PD2, e, alpha, phi, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau)
    return expt_I_1(*args) - expt_I_2(*args)


def expt_I_comb(eta_FC1, eta_FC2, eta_PD1, eta_PD2, e, alpha, phi, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau):
    """Expectation value of combination signal (+).

    Not assuming equal quantum efficiencies of detectors.
    """
    args = (eta_FC1, eta_FC2, eta_PD1, eta_PD2, e, alpha, phi, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau)
    return expt_I_1(*args) + expt_I_2(*args)


def var_I_diff(eta_FC1, eta_FC2, eta_PD1, eta_PD2, e, alpha, phi, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau):
    """Variance of difference signal (-).

    Not assuming equal quantum efficiencies of detectors.
    """
    args = (eta_FC1, eta_FC2, eta_PD1, eta_PD2, e, alpha, phi, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau)
    return var_I_1(*args) + var_I_2(*args) - 2 * cov_I_1_I_2(*args)


def var_I_comb(eta_FC1, eta_FC2, eta_PD1, eta_PD2, e, alpha, phi, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau):
    """Variance of combination signal (+).

    Not assuming equal quantum efficiencies of detectors.
    """
    args = (eta_FC1, eta_FC2, eta_PD1, eta_PD2, e, alpha, phi, eta_inj, r, theta, r_prime, theta_prime, B, omega_0, tau)
    return var_I_1(*args) + var_I_2(*args) + 2 * cov_I_1_I_2(*args)
