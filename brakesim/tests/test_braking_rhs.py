import numpy as np
from scipy.integrate import solve_ivp

from brakesim.models import BrakingParams, EnvironmentParams, PitchParams, braking_rhs


def test_full_braking_rhs_slows_vehicle():
    m = 300.0
    h_cg = 0.30
    Iyy = 60.0
    k_theta = 25000.0
    zeta = 0.3
    c_theta = 2.0 * zeta * np.sqrt(k_theta * Iyy)

    pitch = PitchParams(m=m, h_cg=h_cg, Iyy=Iyy, k_theta=k_theta, c_theta=c_theta)
    params = BrakingParams(
        pitch=pitch,
        g=9.81,
        a=0.9,
        b=0.9,
        k_wf=30000.0,
        k_wr=30000.0,
        Re=0.25,
        Iw=1.2,
        C_kappa=25.0,
        Cl=1.6,
        Cd=1.2,
        A_ref=1.1,
        aero_balance=0.5,
        pedal_ratio=5.0,
        mc_area_f=2.0e-4,
        mc_area_r=2.0e-4,
        caliper_piston_area=3.0e-4,
        caliper_piston_count=4,
        pad_mu_static=0.45,
        pad_mu_kinetic=0.38,
        r_brake=0.12,
        balance_bar=0.6,
    )
    env = EnvironmentParams(mu=1.2, rho=1.225)
    pedal_force = 500.0

    def rhs(t, y):
        return braking_rhs(t, y, params, env, pedal_force)

    v0 = 20.0
    y0 = np.array([v0, 0.0, 0.0, 80.0, 80.0, 80.0, 80.0], dtype=float)
    sol = solve_ivp(rhs, (0.0, 2.0), y0, method="Radau", rtol=1e-8, atol=1e-10)

    v_end = sol.y[0, -1]
    theta_end = sol.y[1, -1]

    assert v_end < 0.5 * v0
    assert theta_end < 0.0
