from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PitchParams:
    m: float        # kg
    h_cg: float     # m
    Iyy: float      # kg*m^2
    k_theta: float  # N*m/rad
    c_theta: float  # N*m*s/rad


def pitch_rhs(theta: float, theta_dot: float, ax: float, p: PitchParams) -> tuple[float, float]:
    """
    Pitch DOF dynamics:
      theta_ddot = (m*ax*h_cg - k_theta*theta - c_theta*theta_dot)/Iyy

    Conventions:
      ax < 0 is braking
      theta < 0 is nose-down (dive)
    """
    theta_ddot = (p.m * ax * p.h_cg - p.k_theta * theta - p.c_theta * theta_dot) / p.Iyy
    return theta_dot, theta_ddot
