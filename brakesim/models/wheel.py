from __future__ import annotations


def wheel_rhs(
    omega: float,
    T_brake: float,
    Fx: float,
    Re: float,
    Iw: float,
) -> float:
    """
    Wheel rotational dynamics.

    Iw * domega = -T_brake - Re * Fx

    Conventions:
      T_brake > 0 resists forward rotation
      Fx < 0 under braking
    """
    return (-T_brake - Re * Fx) / Iw


def wheel_rhs_locking(
    omega: float,
    T_brake: float,
    Fx: float,
    Re: float,
    Iw: float,
    omega_eps: float = 1e-3,
) -> float:
    """
    Wheel dynamics with a simple lock/unlock regime.

    If the wheel is at (or below) zero speed and net torque would further
    decelerate it, hold it locked. If net torque accelerates it, allow unlock.
    """
    domega = (-T_brake - Re * Fx) / Iw
    if omega <= omega_eps and domega < 0.0:
        return 0.0
    return domega
