import numpy as np
import pytest
from scipy.integrate import solve_ivp
from brakesim.models.pitch import PitchParams, pitch_rhs


@pytest.fixture
def pitch_params():
    # Example-ish values. Replace with your car once you have them.
    m = 300.0
    h_cg = 0.30
    Iyy = 60.0  # kg*m^2 (order-of-magnitude)
    k_theta = 25000.0  # N*m/rad
    # Pick damping near moderate (doesn't need to be perfect for tests)
    # c = 2*zeta*sqrt(k*I)
    zeta = 0.3
    c_theta = 2.0 * zeta * np.sqrt(k_theta * Iyy)
    return PitchParams(m=m, h_cg=h_cg, Iyy=Iyy, k_theta=k_theta, c_theta=c_theta)


def test_pitch_steady_state_constant_ax(pitch_params):
    ax = -10.0  # braking
    # steady state: theta_dot=0, theta_ddot=0 => theta_ss = (m*ax*h)/k
    theta_ss_expected = (pitch_params.m * ax * pitch_params.h_cg) / pitch_params.k_theta

    def rhs(t, y):
        theta, theta_dot = y
        dtheta, dtheta_dot = pitch_rhs(theta, theta_dot, ax, pitch_params)
        return [dtheta, dtheta_dot]

    y0 = [0.0, 0.0]
    sol = solve_ivp(rhs, (0.0, 3.0), y0, method="Radau", rtol=1e-8, atol=1e-10)

    theta_end = sol.y[0, -1]
    # Should approach steady state within a small tolerance after 3s
    assert np.isclose(theta_end, theta_ss_expected, rtol=0.02, atol=1e-4)
    # Braking should produce nose-down (theta negative)
    assert theta_end < 0.0


def test_pitch_energy_decays_with_damping(pitch_params):
    # Free response: ax=0, initial pitch deflection -> should decay (damping > 0)
    ax = 0.0

    def rhs(t, y):
        theta, theta_dot = y
        dtheta, dtheta_dot = pitch_rhs(theta, theta_dot, ax, pitch_params)
        return [dtheta, dtheta_dot]

    y0 = [-0.05, 0.0]  # initial dive
    sol = solve_ivp(rhs, (0.0, 5.0), y0, method="Radau", rtol=1e-8, atol=1e-10)

    theta = sol.y[0]
    theta_dot = sol.y[1]

    # "Energy-like" quantity for linear oscillator:
    E = 0.5 * pitch_params.Iyy * theta_dot**2 + 0.5 * pitch_params.k_theta * theta**2

    # With damping, energy should trend downward overall.
    # Use final < initial as a robust check.
    assert E[-1] < 0.1 * E[0]
