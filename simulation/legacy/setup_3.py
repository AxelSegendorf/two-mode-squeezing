"""Functions implementing formulas for photocurrent signals of measurement setup 3."""

import numpy as np


def unnormalized_sinc(x):
    """sin(x)/x with sinc(0) = 1."""
    return np.sinc(x / np.pi)


def _kernel(B, delay):
    """sin(B delay)/delay = B sinc(B delay), with sinc(0) = 1."""
    return B * unnormalized_sinc(B * delay)


## Prefactors

def zeta_p1(eta_FC1, eta_FC2, eta_FC3):
    return np.sqrt(eta_FC1 * eta_FC2 * eta_FC3)


def zeta_p2(eta_FC1, eta_FC2, eta_FC3):
    return np.sqrt(eta_FC1 * (1 - eta_FC2) * eta_FC3)


def zeta_p3(eta_FC1, eta_FC2, eta_FC3):
    return np.sqrt((1 - eta_FC1) * eta_FC2 * eta_FC3)


def zeta_p4(eta_FC1, eta_FC2, eta_FC3):
    return np.sqrt((1 - eta_FC1) * (1 - eta_FC2) * eta_FC3)


def zeta_p5(eta_FC1, eta_FC2, eta_FC3):
    return np.sqrt(eta_FC1 * eta_FC2 * (1 - eta_FC3))


def zeta_p6(eta_FC1, eta_FC2, eta_FC3):
    return np.sqrt(eta_FC1 * (1 - eta_FC2) * (1 - eta_FC3))


def zeta_p7(eta_FC1, eta_FC2, eta_FC3):
    return np.sqrt((1 - eta_FC1) * eta_FC2 * (1 - eta_FC3))


def zeta_p8(eta_FC1, eta_FC2, eta_FC3):
    return np.sqrt((1 - eta_FC1) * (1 - eta_FC2) * (1 - eta_FC3))


def _zeta_p_all(eta_FC1, eta_FC2, eta_FC3):
    return (
        zeta_p1(eta_FC1, eta_FC2, eta_FC3),
        zeta_p2(eta_FC1, eta_FC2, eta_FC3),
        zeta_p3(eta_FC1, eta_FC2, eta_FC3),
        zeta_p4(eta_FC1, eta_FC2, eta_FC3),
        zeta_p5(eta_FC1, eta_FC2, eta_FC3),
        zeta_p6(eta_FC1, eta_FC2, eta_FC3),
        zeta_p7(eta_FC1, eta_FC2, eta_FC3),
        zeta_p8(eta_FC1, eta_FC2, eta_FC3),
    )


## Mean fields

def A_bar_1(eta_FC1, eta_FC2, eta_FC3, alpha, phi, omega_0, tau, tau_prime):
    _, z2, z3, _, z5, _, _, z8 = _zeta_p_all(eta_FC1, eta_FC2, eta_FC3)
    return (
        z3
        - z2 * np.exp(1j * omega_0 * tau)
        + z8 * np.exp(1j * omega_0 * tau_prime)
        + z5 * np.exp(1j * omega_0 * (tau + tau_prime))
    ) * alpha * np.exp(1j * phi)


def A_bar_2(eta_FC1, eta_FC2, eta_FC3, alpha, phi, omega_0, tau, tau_prime):
    z1, _, _, z4, _, z6, z7, _ = _zeta_p_all(eta_FC1, eta_FC2, eta_FC3)
    return (
        z7
        - z6 * np.exp(1j * omega_0 * tau)
        - z4 * np.exp(1j * omega_0 * tau_prime)
        - z1 * np.exp(1j * omega_0 * (tau + tau_prime))
    ) * alpha * np.exp(1j * phi)


## Fluctuation correlators

def expt_dA1d_dA1(eta_FC1, eta_FC2, eta_FC3, r, theta, B, omega_0, tau, tau_prime):
    """Expectation value of delta A_1^dagger with delta A_1."""
    z1, _, _, z4, _, z6, z7, _ = _zeta_p_all(eta_FC1, eta_FC2, eta_FC3)
    k_t = _kernel(B, tau)
    k_tp = _kernel(B, tau_prime)
    k_sum = _kernel(B, tau + tau_prime)
    k_diff = _kernel(B, tau - tau_prime)
    sh2 = np.sinh(r) ** 2
    bracket = (
        B * (z1**2 + z4**2 + z6**2 + z7**2)
        + 2 * k_t * np.cos(omega_0 * tau) * (z1 * z4 - z6 * z7)
        + 2 * k_tp * np.cos(omega_0 * tau_prime) * (z1 * z6 - z4 * z7)
        - 2 * k_sum * np.cos(omega_0 * (tau + tau_prime)) * z1 * z7
        + 2 * k_diff * np.cos(omega_0 * (tau - tau_prime)) * z4 * z6
    )
    return sh2 / np.pi * bracket


def expt_dA1_dA1d(eta_FC1, eta_FC2, eta_FC3, r, theta, B, omega_0, tau, tau_prime):
    """Expectation value of delta A_1 with delta A_1^dagger."""
    z1, z2, z3, z4, z5, z6, z7, z8 = _zeta_p_all(eta_FC1, eta_FC2, eta_FC3)
    k_t = _kernel(B, tau)
    k_tp = _kernel(B, tau_prime)
    k_sum = _kernel(B, tau + tau_prime)
    k_diff = _kernel(B, tau - tau_prime)
    ch2 = np.cosh(r) ** 2
    bracket = (
        B * (ch2 * (z1**2 + z4**2 + z6**2 + z7**2) + (z2**2 + z3**2 + z5**2 + z8**2))
        + 2 * k_t * np.cos(omega_0 * tau) * (ch2 * (z1 * z4 - z6 * z7) - z2 * z3 + z5 * z8)
        + 2 * k_tp * np.cos(omega_0 * tau_prime) * (ch2 * (z1 * z6 - z4 * z7) - z2 * z5 + z3 * z8)
        + 2 * k_sum * np.cos(omega_0 * (tau + tau_prime)) * (-ch2 * z1 * z7 + z3 * z5)
        + 2 * k_diff * np.cos(omega_0 * (tau - tau_prime)) * (ch2 * z4 * z6 - z2 * z8)
    )
    return bracket / np.pi


def expt_dA1_dA1(eta_FC1, eta_FC2, eta_FC3, r, theta, B, omega_0, tau, tau_prime):
    """Expectation value of delta A_1 with delta A_1."""
    z1, _, _, z4, _, z6, z7, _ = _zeta_p_all(eta_FC1, eta_FC2, eta_FC3)
    k_t = _kernel(B, tau)
    k_tp = _kernel(B, tau_prime)
    k_sum = _kernel(B, tau + tau_prime)
    k_diff = _kernel(B, tau - tau_prime)
    chsh = np.cosh(r) * np.sinh(r)
    bracket = (
        B * (
            z1**2
            + np.exp(2j * omega_0 * tau) * z4**2
            + np.exp(2j * omega_0 * tau_prime) * z6**2
            + np.exp(2j * omega_0 * (tau + tau_prime)) * z7**2
        )
        + 2 * k_t * (
            np.exp(1j * omega_0 * tau) * z1 * z4
            - np.exp(1j * omega_0 * (tau + 2 * tau_prime)) * z6 * z7
        )
        + 2 * k_tp * (
            np.exp(1j * omega_0 * tau_prime) * z1 * z6
            - np.exp(1j * omega_0 * (2 * tau + tau_prime)) * z4 * z7
        )
        - 2 * k_sum * np.exp(1j * omega_0 * (tau + tau_prime)) * z1 * z7
        + 2 * k_diff * np.exp(1j * omega_0 * (tau + tau_prime)) * z4 * z6
    )
    return np.exp(1j * theta) * chsh / np.pi * bracket


def expt_dA1d_dA1d(eta_FC1, eta_FC2, eta_FC3, r, theta, B, omega_0, tau, tau_prime):
    """Expectation value of delta A_1^dagger with delta A_1^dagger."""
    return np.conj(
        expt_dA1_dA1(eta_FC1, eta_FC2, eta_FC3, r, theta, B, omega_0, tau, tau_prime)
    )


def expt_dA2d_dA2(eta_FC1, eta_FC2, eta_FC3, r, theta, B, omega_0, tau, tau_prime):
    """Expectation value of delta A_2^dagger with delta A_2."""
    _, z2, z3, _, z5, _, _, z8 = _zeta_p_all(eta_FC1, eta_FC2, eta_FC3)
    k_t = _kernel(B, tau)
    k_tp = _kernel(B, tau_prime)
    k_sum = _kernel(B, tau + tau_prime)
    k_diff = _kernel(B, tau - tau_prime)
    sh2 = np.sinh(r) ** 2
    bracket = (
        B * (z2**2 + z3**2 + z5**2 + z8**2)
        + 2 * k_t * np.cos(omega_0 * tau) * (-z2 * z3 + z5 * z8)
        + 2 * k_tp * np.cos(omega_0 * tau_prime) * (-z2 * z5 + z3 * z8)
        + 2 * k_sum * np.cos(omega_0 * (tau + tau_prime)) * z3 * z5
        - 2 * k_diff * np.cos(omega_0 * (tau - tau_prime)) * z2 * z8
    )
    return sh2 / np.pi * bracket


def expt_dA2_dA2d(eta_FC1, eta_FC2, eta_FC3, r, theta, B, omega_0, tau, tau_prime):
    """Expectation value of delta A_2 with delta A_2^dagger."""
    z1, z2, z3, z4, z5, z6, z7, z8 = _zeta_p_all(eta_FC1, eta_FC2, eta_FC3)
    k_t = _kernel(B, tau)
    k_tp = _kernel(B, tau_prime)
    k_sum = _kernel(B, tau + tau_prime)
    k_diff = _kernel(B, tau - tau_prime)
    ch2 = np.cosh(r) ** 2
    bracket = (
        B * (ch2 * (z2**2 + z3**2 + z5**2 + z8**2) + (z1**2 + z4**2 + z6**2 + z7**2))
        + 2 * k_t * np.cos(omega_0 * tau) * (ch2 * (-z2 * z3 + z5 * z8) + z1 * z4 - z6 * z7)
        + 2 * k_tp * np.cos(omega_0 * tau_prime) * (ch2 * (-z2 * z5 + z3 * z8) + z1 * z6 - z4 * z7)
        + 2 * k_sum * np.cos(omega_0 * (tau + tau_prime)) * (ch2 * z3 * z5 - z1 * z7)
        + 2 * k_diff * np.cos(omega_0 * (tau - tau_prime)) * (-ch2 * z2 * z8 + z4 * z6)
    )
    return bracket / np.pi


def expt_dA2_dA2(eta_FC1, eta_FC2, eta_FC3, r, theta, B, omega_0, tau, tau_prime):
    """Expectation value of delta A_2 with delta A_2."""
    _, z2, z3, _, z5, _, _, z8 = _zeta_p_all(eta_FC1, eta_FC2, eta_FC3)
    k_t = _kernel(B, tau)
    k_tp = _kernel(B, tau_prime)
    k_sum = _kernel(B, tau + tau_prime)
    k_diff = _kernel(B, tau - tau_prime)
    chsh = np.cosh(r) * np.sinh(r)
    bracket = (
        B * (
            z5**2
            + np.exp(2j * omega_0 * tau) * z8**2
            + np.exp(2j * omega_0 * tau_prime) * z2**2
            + np.exp(2j * omega_0 * (tau + tau_prime)) * z3**2
        )
        + 2 * k_t * (
            np.exp(1j * omega_0 * tau) * z5 * z8
            - np.exp(1j * omega_0 * (tau + 2 * tau_prime)) * z2 * z3
        )
        + 2 * k_tp * (
            -np.exp(1j * omega_0 * tau_prime) * z2 * z5
            + np.exp(1j * omega_0 * (2 * tau + tau_prime)) * z3 * z8
        )
        + 2 * k_sum * np.exp(1j * omega_0 * (tau + tau_prime)) * z3 * z5
        - 2 * k_diff * np.exp(1j * omega_0 * (tau + tau_prime)) * z2 * z8
    )
    return np.exp(1j * theta) * chsh / np.pi * bracket


def expt_dA2d_dA2d(eta_FC1, eta_FC2, eta_FC3, r, theta, B, omega_0, tau, tau_prime):
    """Expectation value of delta A_2^dagger with delta A_2^dagger."""
    return np.conj(
        expt_dA2_dA2(eta_FC1, eta_FC2, eta_FC3, r, theta, B, omega_0, tau, tau_prime)
    )


def expt_dA1d_dA2(eta_FC1, eta_FC2, eta_FC3, r, theta, B, omega_0, tau, tau_prime):
    """Expectation value of delta A_1^dagger with delta A_2."""
    z1, z2, z3, z4, z5, z6, z7, z8 = _zeta_p_all(eta_FC1, eta_FC2, eta_FC3)
    k_t = _kernel(B, tau)
    k_tp = _kernel(B, tau_prime)
    k_sum = _kernel(B, tau + tau_prime)
    k_diff = _kernel(B, tau - tau_prime)
    sh2 = np.sinh(r) ** 2
    bracket = (
        B * (z1 * z5 - z2 * z6 - z3 * z7 + z4 * z8)
        + k_t * (
            np.exp(1j * omega_0 * tau) * (z1 * z8 + z3 * z6)
            + np.exp(-1j * omega_0 * tau) * (z2 * z7 + z4 * z5)
        )
        + k_tp * (
            np.exp(1j * omega_0 * tau_prime) * (-z1 * z2 + z3 * z4)
            + np.exp(-1j * omega_0 * tau_prime) * (z5 * z6 - z7 * z8)
        )
        + k_sum * (
            np.exp(1j * omega_0 * (tau + tau_prime)) * z1 * z3
            - np.exp(-1j * omega_0 * (tau + tau_prime)) * z5 * z7
        )
        + k_diff * (
            np.exp(1j * omega_0 * (tau - tau_prime)) * z6 * z8
            - np.exp(-1j * omega_0 * (tau - tau_prime)) * z2 * z4
        )
    )
    return sh2 / np.pi * bracket


def expt_dA1_dA2d(eta_FC1, eta_FC2, eta_FC3, r, theta, B, omega_0, tau, tau_prime):
    """Expectation value of delta A_1 with delta A_2^dagger."""
    z1, z2, z3, z4, z5, z6, z7, z8 = _zeta_p_all(eta_FC1, eta_FC2, eta_FC3)
    k_t = _kernel(B, tau)
    k_tp = _kernel(B, tau_prime)
    k_sum = _kernel(B, tau + tau_prime)
    k_diff = _kernel(B, tau - tau_prime)
    ch2 = np.cosh(r) ** 2
    mu_p = z1 * z5 - z2 * z6 - z3 * z7 + z4 * z8
    bracket = (
        B * (ch2 * mu_p + (-z1 * z5 + z2 * z6 + z3 * z7 - z4 * z8))
        + k_t * (
            np.exp(1j * omega_0 * tau) * (ch2 * (z2 * z7 + z4 * z5) - z2 * z7 - z4 * z5)
            + np.exp(-1j * omega_0 * tau) * (ch2 * (z1 * z8 + z3 * z6) - z1 * z8 - z3 * z6)
        )
        + k_tp * (
            np.exp(1j * omega_0 * tau_prime) * (ch2 * (z5 * z6 - z7 * z8) - z5 * z6 + z7 * z8)
            + np.exp(-1j * omega_0 * tau_prime) * (ch2 * (-z1 * z2 + z3 * z4) + z1 * z2 - z3 * z4)
        )
        + k_sum * (
            np.exp(1j * omega_0 * (tau + tau_prime)) * (-ch2 * z5 * z7 + z5 * z7)
            + np.exp(-1j * omega_0 * (tau + tau_prime)) * (ch2 * z1 * z3 - z1 * z3)
        )
        + k_diff * (
            np.exp(1j * omega_0 * (tau - tau_prime)) * (-ch2 * z2 * z4 + z2 * z4)
            + np.exp(-1j * omega_0 * (tau - tau_prime)) * (ch2 * z6 * z8 - z6 * z8)
        )
    )
    return bracket / np.pi


def expt_dA1_dA2(eta_FC1, eta_FC2, eta_FC3, r, theta, B, omega_0, tau, tau_prime):
    """Expectation value of delta A_1 with delta A_2."""
    z1, z2, z3, z4, z5, z6, z7, z8 = _zeta_p_all(eta_FC1, eta_FC2, eta_FC3)
    k_t = _kernel(B, tau)
    k_tp = _kernel(B, tau_prime)
    k_sum = _kernel(B, tau + tau_prime)
    k_diff = _kernel(B, tau - tau_prime)
    chsh = np.cosh(r) * np.sinh(r)
    bracket = (
        B * (
            z1 * z5
            + np.exp(2j * omega_0 * tau) * z4 * z8
            - np.exp(2j * omega_0 * tau_prime) * z2 * z6
            - np.exp(2j * omega_0 * (tau + tau_prime)) * z3 * z7
        )
        + k_t * (
            np.exp(1j * omega_0 * tau) * (z1 * z8 + z4 * z5)
            + np.exp(1j * omega_0 * (tau + 2 * tau_prime)) * (z2 * z7 + z3 * z6)
        )
        + k_tp * (
            np.exp(1j * omega_0 * tau_prime) * (-z1 * z2 + z5 * z6)
            + np.exp(1j * omega_0 * (2 * tau + tau_prime)) * (z3 * z4 - z7 * z8)
        )
        + k_sum * np.exp(1j * omega_0 * (tau + tau_prime)) * (z1 * z3 - z5 * z7)
        + k_diff * np.exp(1j * omega_0 * (tau + tau_prime)) * (-z2 * z4 + z6 * z8)
    )
    return np.exp(1j * theta) * chsh / np.pi * bracket


def expt_dA1d_dA2d(eta_FC1, eta_FC2, eta_FC3, r, theta, B, omega_0, tau, tau_prime):
    """Expectation value of delta A_1^dagger with delta A_2^dagger."""
    return np.conj(
        expt_dA1_dA2(eta_FC1, eta_FC2, eta_FC3, r, theta, B, omega_0, tau, tau_prime)
    )


def expt_dA2d_dA1(eta_FC1, eta_FC2, eta_FC3, r, theta, B, omega_0, tau, tau_prime):
    """Expectation value of delta A_2^dagger with delta A_1."""
    return np.conj(
        expt_dA1d_dA2(eta_FC1, eta_FC2, eta_FC3, r, theta, B, omega_0, tau, tau_prime)
    )


def expt_dA2_dA1d(eta_FC1, eta_FC2, eta_FC3, r, theta, B, omega_0, tau, tau_prime):
    """Expectation value of delta A_2 with delta A_1^dagger."""
    return np.conj(
        expt_dA1_dA2d(eta_FC1, eta_FC2, eta_FC3, r, theta, B, omega_0, tau, tau_prime)
    )


def expt_dA2_dA1(eta_FC1, eta_FC2, eta_FC3, r, theta, B, omega_0, tau, tau_prime):
    """Expectation value of delta A_2 with delta A_1."""
    return expt_dA1_dA2(eta_FC1, eta_FC2, eta_FC3, r, theta, B, omega_0, tau, tau_prime)


def expt_dA2d_dA1d(eta_FC1, eta_FC2, eta_FC3, r, theta, B, omega_0, tau, tau_prime):
    """Expectation value of delta A_2^dagger with delta A_1^dagger."""
    return np.conj(
        expt_dA1_dA2(eta_FC1, eta_FC2, eta_FC3, r, theta, B, omega_0, tau, tau_prime)
    )


## Main signals

def expt_I_1(eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, e, alpha, phi, r, theta, B, omega_0, tau, tau_prime):
    """Expectation value of signal 1."""
    a_bar = A_bar_1(eta_FC1, eta_FC2, eta_FC3, alpha, phi, omega_0, tau, tau_prime)
    c_d_d = expt_dA1d_dA1(eta_FC1, eta_FC2, eta_FC3, r, theta, B, omega_0, tau, tau_prime)
    return np.real(eta_PD1 * e * (np.conj(a_bar) * a_bar + c_d_d))


def expt_I_2(eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, e, alpha, phi, r, theta, B, omega_0, tau, tau_prime):
    """Expectation value of signal 2."""
    a_bar = A_bar_2(eta_FC1, eta_FC2, eta_FC3, alpha, phi, omega_0, tau, tau_prime)
    c_d_d = expt_dA2d_dA2(eta_FC1, eta_FC2, eta_FC3, r, theta, B, omega_0, tau, tau_prime)
    return np.real(eta_PD2 * e * (np.conj(a_bar) * a_bar + c_d_d))


def var_I_1(eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, e, alpha, phi, r, theta, B, omega_0, tau, tau_prime):
    """Variance of signal 1."""
    pref = (eta_PD1 * e) ** 2
    a_bar = A_bar_1(eta_FC1, eta_FC2, eta_FC3, alpha, phi, omega_0, tau, tau_prime)
    c_d_d = expt_dA1d_dA1(eta_FC1, eta_FC2, eta_FC3, r, theta, B, omega_0, tau, tau_prime)
    c_d_dag = expt_dA1_dA1d(eta_FC1, eta_FC2, eta_FC3, r, theta, B, omega_0, tau, tau_prime)
    c_dd = expt_dA1_dA1(eta_FC1, eta_FC2, eta_FC3, r, theta, B, omega_0, tau, tau_prime)
    term1 = np.conj(a_bar) * a_bar * (c_d_d + c_d_dag)
    term2 = 2 * np.real(np.conj(a_bar) ** 2 * c_dd)
    term3 = np.abs(c_dd) ** 2
    term4 = c_d_d * c_d_dag
    return np.real(pref * (term1 + term2 + term3 + term4))


def var_I_2(eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, e, alpha, phi, r, theta, B, omega_0, tau, tau_prime):
    """Variance of signal 2."""
    pref = (eta_PD2 * e) ** 2
    a_bar = A_bar_2(eta_FC1, eta_FC2, eta_FC3, alpha, phi, omega_0, tau, tau_prime)
    c_d_d = expt_dA2d_dA2(eta_FC1, eta_FC2, eta_FC3, r, theta, B, omega_0, tau, tau_prime)
    c_d_dag = expt_dA2_dA2d(eta_FC1, eta_FC2, eta_FC3, r, theta, B, omega_0, tau, tau_prime)
    c_dd = expt_dA2_dA2(eta_FC1, eta_FC2, eta_FC3, r, theta, B, omega_0, tau, tau_prime)
    term1 = np.conj(a_bar) * a_bar * (c_d_d + c_d_dag)
    term2 = 2 * np.real(np.conj(a_bar) ** 2 * c_dd)
    term3 = np.abs(c_dd) ** 2
    term4 = c_d_d * c_d_dag
    return np.real(pref * (term1 + term2 + term3 + term4))


def cov_I_1_I_2(eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, e, alpha, phi, r, theta, B, omega_0, tau, tau_prime):
    """Covariance of signals 1 and 2."""
    pref = eta_PD1 * eta_PD2 * e**2
    a_bar_1 = A_bar_1(eta_FC1, eta_FC2, eta_FC3, alpha, phi, omega_0, tau, tau_prime)
    a_bar_2 = A_bar_2(eta_FC1, eta_FC2, eta_FC3, alpha, phi, omega_0, tau, tau_prime)
    c_dA1_dA2d = expt_dA1_dA2d(eta_FC1, eta_FC2, eta_FC3, r, theta, B, omega_0, tau, tau_prime)
    c_dA1_dA2 = expt_dA1_dA2(eta_FC1, eta_FC2, eta_FC3, r, theta, B, omega_0, tau, tau_prime)
    c_dA1d_dA2 = expt_dA1d_dA2(eta_FC1, eta_FC2, eta_FC3, r, theta, B, omega_0, tau, tau_prime)
    c_dA1d_dA2d = expt_dA1d_dA2d(eta_FC1, eta_FC2, eta_FC3, r, theta, B, omega_0, tau, tau_prime)
    value = pref * (
        2 * np.real(np.conj(a_bar_1) * a_bar_2 * c_dA1_dA2d)
        + 2 * np.real(np.conj(a_bar_1) * np.conj(a_bar_2) * c_dA1_dA2)
        + c_dA1d_dA2d * c_dA1_dA2
        + c_dA1d_dA2 * c_dA1_dA2d
    )
    return np.real(value)


def expt_I_diff(eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, e, alpha, phi, r, theta, B, omega_0, tau, tau_prime):
    """Expectation value of difference signal (-).

    Not assuming equal quantum efficiencies of detectors.
    """
    args = (eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, e, alpha, phi, r, theta, B, omega_0, tau, tau_prime)
    return expt_I_1(*args) - expt_I_2(*args)


def expt_I_comb(eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, e, alpha, phi, r, theta, B, omega_0, tau, tau_prime):
    """Expectation value of combination signal (+).

    Not assuming equal quantum efficiencies of detectors.
    """
    args = (eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, e, alpha, phi, r, theta, B, omega_0, tau, tau_prime)
    return expt_I_1(*args) + expt_I_2(*args)


def var_I_diff(eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, e, alpha, phi, r, theta, B, omega_0, tau, tau_prime):
    """Variance of difference signal (-).

    Not assuming equal quantum efficiencies of detectors.
    """
    args = (eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, e, alpha, phi, r, theta, B, omega_0, tau, tau_prime)
    return var_I_1(*args) + var_I_2(*args) - 2 * cov_I_1_I_2(*args)


def var_I_comb(eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, e, alpha, phi, r, theta, B, omega_0, tau, tau_prime):
    """Variance of combination signal (+).

    Not assuming equal quantum efficiencies of detectors.
    """
    args = (eta_FC1, eta_FC2, eta_FC3, eta_PD1, eta_PD2, e, alpha, phi, r, theta, B, omega_0, tau, tau_prime)
    return var_I_1(*args) + var_I_2(*args) + 2 * cov_I_1_I_2(*args)
