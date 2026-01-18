from __future__ import annotations
import numpy as np


def slip_ratio(v: float, omega: float, Re: float, v_eps: float = 0.5) -> float:
    """
    Longitudinal slip ratio.

    Conventions:
      v > 0 : vehicle moving forward
      omega > 0 : wheel rotating forward
      slip < 0 : braking
    """
    denom = max(abs(v), v_eps)
    return (Re * omega - v) / denom


def tire_fx(
    kappa: float,
    Fz: float,
    mu: float,
    C_kappa: float,
) -> float:
    """
    Longitudinal tire force model.

    Parameters
    ----------
    kappa : slip ratio
    Fz : normal load [N]
    mu : peak friction coefficient
    C_kappa : longitudinal stiffness parameter (1/slip)

    Returns
    -------
    Fx : longitudinal force [N] (negative under braking)
    """
    # Saturates smoothly at ±mu*Fz
    return mu * Fz * np.tanh(C_kappa * kappa)
