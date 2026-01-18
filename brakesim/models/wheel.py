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
