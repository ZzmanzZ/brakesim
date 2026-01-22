from __future__ import annotations
import numpy as np

from brakesim.models.pacejka import TirData


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
    tir: "TirData | None" = None,
) -> float:
    """
    Longitudinal tire force model (Pacejka-style).

    Parameters
    ----------
    kappa : slip ratio
    Fz : normal load [N]
    mu : peak friction coefficient
    C_kappa : stiffness factor (used as Pacejka B)
    tir : optional .tir dataset; if provided, uses Magic Formula coefficients

    Returns
    -------
    Fx : longitudinal force [N] (negative under braking)
    """
    if tir is None:
        # Simple Pacejka form with fixed C and E; B provided by C_kappa.
        D = mu * Fz
        B = C_kappa
        C = 1.3
        E = 0.0
        return D * np.sin(C * np.arctan(B * kappa - E * (B * kappa - np.arctan(B * kappa))))

    return tire_fx_pacejka(kappa, Fz, tir)


def tire_fx_pacejka(kappa: float, Fz: float, tir: "TirData") -> float:
    """
    Pure longitudinal Magic Formula (MF5.2-style) using .tir coefficients.
    """
    Fz0 = tir.get("VERTICAL", "FNOMIN", 1.0) or 1.0
    LFZO = tir.get("SCALING_COEFFICIENTS", "LFZO", 1.0) or 1.0
    dfz = (Fz - Fz0 * LFZO) / (Fz0 * LFZO)

    LCX = tir.get("SCALING_COEFFICIENTS", "LCX", 1.0) or 1.0
    LMUX = tir.get("SCALING_COEFFICIENTS", "LMUX", 1.0) or 1.0
    LEX = tir.get("SCALING_COEFFICIENTS", "LEX", 1.0) or 1.0
    LKX = tir.get("SCALING_COEFFICIENTS", "LKX", 1.0) or 1.0
    LHX = tir.get("SCALING_COEFFICIENTS", "LHX", 1.0) or 1.0
    LVX = tir.get("SCALING_COEFFICIENTS", "LVX", 1.0) or 1.0

    PCX1 = tir.get("LONGITUDINAL_COEFFICIENTS", "PCX1", 1.0) or 1.0
    PDX1 = tir.get("LONGITUDINAL_COEFFICIENTS", "PDX1", 1.0) or 1.0
    PDX2 = tir.get("LONGITUDINAL_COEFFICIENTS", "PDX2", 0.0) or 0.0
    PDX3 = tir.get("LONGITUDINAL_COEFFICIENTS", "PDX3", 0.0) or 0.0
    PEX1 = tir.get("LONGITUDINAL_COEFFICIENTS", "PEX1", 0.0) or 0.0
    PEX2 = tir.get("LONGITUDINAL_COEFFICIENTS", "PEX2", 0.0) or 0.0
    PEX3 = tir.get("LONGITUDINAL_COEFFICIENTS", "PEX3", 0.0) or 0.0
    PEX4 = tir.get("LONGITUDINAL_COEFFICIENTS", "PEX4", 0.0) or 0.0
    PKX1 = tir.get("LONGITUDINAL_COEFFICIENTS", "PKX1", 0.0) or 0.0
    PKX2 = tir.get("LONGITUDINAL_COEFFICIENTS", "PKX2", 0.0) or 0.0
    PKX3 = tir.get("LONGITUDINAL_COEFFICIENTS", "PKX3", 0.0) or 0.0
    PHX1 = tir.get("LONGITUDINAL_COEFFICIENTS", "PHX1", 0.0) or 0.0
    PHX2 = tir.get("LONGITUDINAL_COEFFICIENTS", "PHX2", 0.0) or 0.0
    PVX1 = tir.get("LONGITUDINAL_COEFFICIENTS", "PVX1", 0.0) or 0.0
    PVX2 = tir.get("LONGITUDINAL_COEFFICIENTS", "PVX2", 0.0) or 0.0

    Cx = PCX1 * LCX
    Dx = (PDX1 + PDX2 * dfz) * (1.0 - PDX3 * 0.0) * Fz * LMUX
    Ex = (PEX1 + PEX2 * dfz + PEX3 * dfz * dfz) * LEX
    Kxk = Fz * (PKX1 + PKX2 * dfz) * np.exp(PKX3 * dfz) * LKX
    Bx = Kxk / max(Cx * Dx, 1e-6)
    Shx = (PHX1 + PHX2 * dfz) * LHX
    Svx = Fz * (PVX1 + PVX2 * dfz) * LVX * LMUX

    kx = kappa + Shx
    Ex = Ex * (1.0 - PEX4 * np.sign(kx))
    return Dx * np.sin(Cx * np.arctan(Bx * kx - Ex * (Bx * kx - np.arctan(Bx * kx)))) + Svx
