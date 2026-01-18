from __future__ import annotations

def series_stiffness(k1: float, k2: float) -> float:
    """Two springs in series. Handles large k2 (tire very stiff) gracefully."""
    if k1 <= 0.0 or k2 <= 0.0:
        raise ValueError("Stiffness must be positive.")
    return 1.0 / (1.0 / k1 + 1.0 / k2)


def normal_loads_from_pitch(
    theta: float,
    m: float,
    g: float,
    a: float,
    b: float,
    k_wf: float,
    k_wr: float,
    aero_f: float = 0.0,
    aero_r: float = 0.0,
    k_tf: float | None = None,  # tire vertical stiffness per front wheel (N/m)
    k_tr: float | None = None,  # tire vertical stiffness per rear wheel (N/m)
):
    """
    Compute tire normal loads from pitch angle.

    Conventions:
      theta < 0 : nose down (braking)
      ax < 0 : braking
      Suspension compression increases tire normal load.

    k_wf/k_wr are wheel rates at the contact patch *excluding* tire stiffness.
    If k_tf/k_tr provided, uses series combination (suspension + tire in series).
    """
    L = a + b

    # Static axle loads
    Fzf0 = m * g * b / L
    Fzr0 = m * g * a / L

    # Pitch-induced vertical displacements (at axles)
    dz_f = -a * theta   # front compression positive under braking
    dz_r =  b * theta   # rear extension negative under braking

    # Effective vertical stiffness per wheel at each axle
    if k_tf is not None:
        k_wf_eff = series_stiffness(k_wf, k_tf)
    else:
        k_wf_eff = k_wf

    if k_tr is not None:
        k_wr_eff = series_stiffness(k_wr, k_tr)
    else:
        k_wr_eff = k_wr

    # Axle normal loads (sum left+right)
    Fzf = Fzf0 + 2.0 * k_wf_eff * dz_f + aero_f
    Fzr = Fzr0 + 2.0 * k_wr_eff * dz_r + aero_r

    return {
        "FL": 0.5 * Fzf,
        "FR": 0.5 * Fzf,
        "RL": 0.5 * Fzr,
        "RR": 0.5 * Fzr,
    }
