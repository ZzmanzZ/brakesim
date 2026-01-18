from __future__ import annotations

from pathlib import Path
import sys
import json
import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# Ensure local package imports work when running the script directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brakesim.models import (
    VehicleParams,
    braking_rhs,
    normal_loads_from_pitch,
    slip_ratio,
    tire_fx,
    zero_speed_event,
)


def default_vehicle() -> VehicleParams:
    # Example-ish values; adjust for your car.
    zeta = 0.3
    m = 300.0
    k_theta = 25000.0
    Iyy = 60.0
    c_theta = 2.0 * zeta * np.sqrt(k_theta * Iyy)
    return VehicleParams(
        m=m,
        g=9.81,
        a=0.9,
        b=0.9,
        h_cg=0.30,
        Iyy=Iyy,
        k_theta=k_theta,
        c_theta=c_theta,
        k_wf=30000.0,
        k_wr=30000.0,
        Re=0.25,
        Iw=1.2,
        mu=1.2,
        C_kappa=25.0,
    )


def load_vehicle(path: Path) -> VehicleParams:
    data = json.loads(path.read_text())
    return VehicleParams.from_dict(data)


def default_sim_config() -> dict:
    return {
        "t_end": 5.0,
        "t_eval_points": 800,
        "T_brake": {"FL": 200.0, "FR": 200.0, "RL": 200.0, "RR": 200.0},
        "y0": {
            "v": 20.0,
            "theta": 0.0,
            "theta_dot": 0.0,
            "omega_fl": 80.0,
            "omega_fr": 80.0,
            "omega_rl": 80.0,
            "omega_rr": 80.0,
        },
    }


def load_sim_config(path: Path) -> dict:
    data = json.loads(path.read_text())
    defaults = default_sim_config()
    defaults.update(data)
    return defaults


def initial_state_from_config(config: dict) -> np.ndarray:
    y0_cfg = config.get("y0", {})
    keys = ["v", "theta", "theta_dot", "omega_fl", "omega_fr", "omega_rl", "omega_rr"]
    if not all(k in y0_cfg for k in keys):
        missing = [k for k in keys if k not in y0_cfg]
        raise ValueError(f"Missing initial state keys: {missing}")
    return np.array(
        [
            y0_cfg["v"],
            y0_cfg["theta"],
            y0_cfg["theta_dot"],
            y0_cfg["omega_fl"],
            y0_cfg["omega_fr"],
            y0_cfg["omega_rl"],
            y0_cfg["omega_rr"],
        ],
        dtype=float,
    )


def run_sim(vehicle: VehicleParams, config: dict):
    params = vehicle.braking_params()
    T_brake = config["T_brake"]

    def rhs(t, y):
        return braking_rhs(t, y, params, T_brake)

    y0 = initial_state_from_config(config)
    t_end = config["t_end"]
    t_eval = np.linspace(0.0, t_end, int(config["t_eval_points"]))
    sol = solve_ivp(
        rhs,
        (0.0, t_end),
        y0,
        t_eval=t_eval,
        events=zero_speed_event,
        method="Radau",
        rtol=1e-8,
        atol=1e-10,
    )
    return sol, vehicle, params, T_brake


def compute_wheel_series(sol, params, T_brake):
    t = sol.t
    v = sol.y[0]
    theta = sol.y[1]
    omega = {
        "FL": sol.y[3],
        "FR": sol.y[4],
        "RL": sol.y[5],
        "RR": sol.y[6],
    }

    Fz = {k: np.zeros_like(t) for k in omega}
    Fx = {k: np.zeros_like(t) for k in omega}
    kappa = {k: np.zeros_like(t) for k in omega}
    Tbr = {k: np.zeros_like(t) for k in omega}

    def wheel_value(value, key):
        if isinstance(value, dict):
            return value[key]
        return value

    for i in range(t.size):
        loads = normal_loads_from_pitch(
            theta=theta[i],
            m=params.pitch.m,
            g=params.g,
            a=params.a,
            b=params.b,
            k_wf=params.k_wf,
            k_wr=params.k_wr,
            aero_f=params.aero_f,
            aero_r=params.aero_r,
            k_tf=params.k_tf,
            k_tr=params.k_tr,
        )
        for key in omega:
            Re = wheel_value(params.Re, key)
            kappa[key][i] = slip_ratio(v[i], omega[key][i], Re)
            Fz[key][i] = loads[key]
            Fx[key][i] = tire_fx(kappa[key][i], Fz[key][i], params.mu, params.C_kappa)
            Tbr[key][i] = wheel_value(T_brake, key)

    return t, v, theta, omega, kappa, Fx, Fz, Tbr


def save_plots(out_dir: Path, series, show: bool = False):
    t, v, theta, omega, kappa, Fx, Fz, _ = series
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, v, label="v")
    plt.ylabel("Speed [m/s]")
    plt.grid(True)
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(t, theta, label="theta")
    plt.ylabel("Pitch [rad]")
    plt.xlabel("Time [s]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "speed_pitch.png", dpi=150)
    if not show:
        plt.close()

    plt.figure(figsize=(9, 6))
    for key in omega:
        plt.plot(t, omega[key], label=f"omega {key}")
    plt.ylabel("Wheel speed [rad/s]")
    plt.xlabel("Time [s]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "wheel_speeds.png", dpi=150)
    if not show:
        plt.close()

    plt.figure(figsize=(9, 6))
    for key in kappa:
        plt.plot(t, kappa[key], label=f"kappa {key}")
    plt.ylabel("Slip ratio [-]")
    plt.xlabel("Time [s]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "slip_ratios.png", dpi=150)
    if not show:
        plt.close()

    plt.figure(figsize=(9, 6))
    for key in Fx:
        plt.plot(t, Fx[key], label=f"Fx {key}")
    plt.ylabel("Longitudinal force [N]")
    plt.xlabel("Time [s]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "tire_forces.png", dpi=150)
    if not show:
        plt.close()

    plt.figure(figsize=(9, 6))
    for key in Fz:
        plt.plot(t, Fz[key], label=f"Fz {key}")
    plt.ylabel("Normal load [N]")
    plt.xlabel("Time [s]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "normal_loads.png", dpi=150)
    if not show:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run braking sim and generate plots.")
    parser.add_argument("--show", action="store_true", help="Show plots in an interactive window.")
    parser.add_argument("--params", type=str, help="Path to JSON vehicle params.")
    parser.add_argument("--init", type=str, help="Path to JSON simulation config.")
    args = parser.parse_args()

    if args.init:
        config = load_sim_config(Path(args.init))
    else:
        config = default_sim_config()

    if args.params:
        vehicle = load_vehicle(Path(args.params))
        sol, vehicle, params, T_brake = run_sim(vehicle, config)
    else:
        sol, vehicle, params, T_brake = run_sim(default_vehicle(), config)
    series = compute_wheel_series(sol, params, T_brake)
    out_dir = Path("brakesim/plots")
    save_plots(out_dir, series, show=args.show)
    if args.show:
        plt.show()
    print(f"Saved plots to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
