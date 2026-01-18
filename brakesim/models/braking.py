from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from brakesim.models.loads import normal_loads_from_pitch
from brakesim.models.pitch import PitchParams, pitch_rhs
from brakesim.models.tire import slip_ratio, tire_fx
from brakesim.models.wheel import wheel_rhs


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
    mu: float
    C_kappa: float
    aero_f: float = 0.0
    aero_r: float = 0.0
    k_tf: float | None = None
    k_tr: float | None = None


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


def braking_rhs(
    t: float,
    y: np.ndarray,
    p: BrakingParams,
    T_brake: float | Mapping[str, float],
) -> np.ndarray:
    """
    Full straight-line braking RHS.

    State vector: [v, theta, theta_dot, omega_FL, omega_FR, omega_RL, omega_RR]
    """
    v, theta, theta_dot, omega_fl, omega_fr, omega_rl, omega_rr = y
    omegas = {"FL": omega_fl, "FR": omega_fr, "RL": omega_rl, "RR": omega_rr}

    Fz = normal_loads_from_pitch(
        theta=theta,
        m=p.pitch.m,
        g=p.g,
        a=p.a,
        b=p.b,
        k_wf=p.k_wf,
        k_wr=p.k_wr,
        aero_f=p.aero_f,
        aero_r=p.aero_r,
        k_tf=p.k_tf,
        k_tr=p.k_tr,
    )

    Fx = {}
    for key, omega in omegas.items():
        kappa = slip_ratio(v, omega, _wheel_value(p.Re, key))
        Fx[key] = tire_fx(kappa, Fz[key], p.mu, p.C_kappa)

    ax = sum(Fx.values()) / p.pitch.m
    dtheta, dtheta_dot = pitch_rhs(theta, theta_dot, ax, p.pitch)
    dv = ax

    brake = _wheel_values(T_brake)
    domega = {}
    for key, omega in omegas.items():
        domega[key] = wheel_rhs(
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
