from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from brakesim.models.loads import normal_loads_from_pitch
from brakesim.models.pitch import PitchParams, pitch_rhs
from brakesim.models.tire import slip_ratio, tire_fx
from brakesim.models.pacejka import TirData
from brakesim.models.wheel import wheel_rhs_locking


@dataclass(frozen=True)
class BrakingParams:
    pitch: PitchParams
    g: float
    a: float
    b: float
    k_wf: float
    k_wr: float
    Re: float | Mapping[str, float]
    Iw: float | Mapping[str, float]
    C_kappa: float
    Cl: float
    Cd: float
    A_ref: float
    aero_balance: float
    pedal_ratio: float
    mc_area_f: float
    mc_area_r: float
    caliper_piston_area: float | Mapping[str, float]
    caliper_piston_count: int | Mapping[str, int]
    pad_mu_static: float | Mapping[str, float]
    pad_mu_kinetic: float | Mapping[str, float]
    r_brake: float | Mapping[str, float]
    balance_bar: float
    tir_params: "TirData | None" = None
    k_tf: float | None = None
    k_tr: float | None = None


@dataclass(frozen=True)
class EnvironmentParams:
    mu: float
    rho: float


def _wheel_value(value: float | Mapping[str, float], key: str) -> float:
    if isinstance(value, Mapping):
        return value[key]
    return value


def _wheel_values(
    value: float | Mapping[str, float],
) -> dict[str, float]:
    if isinstance(value, Mapping):
        return dict(value)
    return {"FL": value, "FR": value, "RL": value, "RR": value}


def _wheel_side(key: str) -> str:
    if key in ("FL", "FR"):
        return "F"
    return "R"


def brake_torque_from_pedal(
    pedal_force: float,
    p: BrakingParams,
    omegas: Mapping[str, float],
    omega_eps: float = 1e-3,
) -> dict[str, float]:
    if p.mc_area_f <= 0.0 or p.mc_area_r <= 0.0:
        raise ValueError("Master cylinder area must be positive.")
    if p.pedal_ratio <= 0.0:
        raise ValueError("Pedal ratio must be positive.")

    balance = min(max(p.balance_bar, 0.0), 1.0)
    pedal_force_f = pedal_force * balance
    pedal_force_r = pedal_force * (1.0 - balance)
    pressure_f = pedal_force_f * p.pedal_ratio / p.mc_area_f
    pressure_r = pedal_force_r * p.pedal_ratio / p.mc_area_r

    T = {}
    for key, omega in omegas.items():
        side = _wheel_side(key)
        pressure = pressure_f if side == "F" else pressure_r
        piston_area = _wheel_value(p.caliper_piston_area, key)
        piston_count = _wheel_value(p.caliper_piston_count, key)
        mu_pad = _wheel_value(p.pad_mu_static, key)
        if omega <= omega_eps:
            mu_pad = _wheel_value(p.pad_mu_kinetic, key)
        r_brake = _wheel_value(p.r_brake, key)
        T[key] = pressure * piston_area * piston_count * mu_pad * r_brake
    return T


def braking_rhs(
    t: float,
    y: np.ndarray,
    p: BrakingParams,
    env: EnvironmentParams,
    pedal_force: float,
) -> np.ndarray:
    """
    Full straight-line braking RHS.

    State vector: [v, theta, theta_dot, omega_FL, omega_FR, omega_RL, omega_RR]
    """
    v, theta, theta_dot, omega_fl, omega_fr, omega_rl, omega_rr = y
    omegas = {"FL": omega_fl, "FR": omega_fr, "RL": omega_rl, "RR": omega_rr}

    downforce = 0.5 * env.rho * p.Cl * p.A_ref * v * v
    aero_f = p.aero_balance * downforce
    aero_r = (1.0 - p.aero_balance) * downforce

    Fz = normal_loads_from_pitch(
        theta=theta,
        m=p.pitch.m,
        g=p.g,
        a=p.a,
        b=p.b,
        k_wf=p.k_wf,
        k_wr=p.k_wr,
        aero_f=aero_f,
        aero_r=aero_r,
        k_tf=p.k_tf,
        k_tr=p.k_tr,
    )

    Fx = {}
    for key, omega in omegas.items():
        kappa = slip_ratio(v, omega, _wheel_value(p.Re, key))
        Fx[key] = tire_fx(kappa, Fz[key], env.mu, p.C_kappa, p.tir_params)

    F_drag = -0.5 * env.rho * p.Cd * p.A_ref * v * abs(v)
    ax = (sum(Fx.values()) + F_drag) / p.pitch.m
    dtheta, dtheta_dot = pitch_rhs(theta, theta_dot, ax, p.pitch)
    dv = ax

    brake = brake_torque_from_pedal(pedal_force, p, omegas)
    domega = {}
    for key, omega in omegas.items():
        domega[key] = wheel_rhs_locking(
            omega,
            brake[key],
            Fx[key],
            _wheel_value(p.Re, key),
            _wheel_value(p.Iw, key),
        )

    return np.array(
        [
            dv,
            dtheta,
            dtheta_dot,
            domega["FL"],
            domega["FR"],
            domega["RL"],
            domega["RR"],
        ],
        dtype=float,
    )


def zero_speed_event(t: float, y: np.ndarray) -> float:
    """Event function to stop integration when vehicle speed reaches zero."""
    _ = t
    return y[0]


zero_speed_event.terminal = True
zero_speed_event.direction = -1
