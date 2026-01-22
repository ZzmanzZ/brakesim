from __future__ import annotations

from pathlib import Path
import sys
import argparse
import yaml

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# Ensure local package imports work when running the script directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brakesim.models import (
    EnvironmentParams,
    VehicleParams,
    brake_torque_from_pedal,
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


def load_vehicle(path: Path) -> VehicleParams:
    data = yaml.safe_load(path.read_text()) or {}
    return VehicleParams.from_dict(data)


def default_sim_config() -> dict:
    return {
        "t_end": 5.0,
        "t_eval_points": 800,
        "rho": 1.225,
        "mu": 1.2,
        "pedal_force": 500.0,
        "pedal_profile": None,
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
    data = yaml.safe_load(path.read_text()) or {}
    defaults = default_sim_config()
    defaults.update(data)
    return defaults


def pedal_force_at(config: dict):
    profile = config.get("pedal_profile")
    if not profile:
        force = float(config["pedal_force"])

        def constant(_t: float) -> float:
            return force

        return constant

    points = [(float(p["t"]), float(p["force"])) for p in profile]
    points.sort(key=lambda item: item[0])
    times = np.array([p[0] for p in points], dtype=float)
    forces = np.array([p[1] for p in points], dtype=float)

    def piecewise(t: float) -> float:
        return float(np.interp(t, times, forces))

    return piecewise


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
    env = EnvironmentParams(mu=config["mu"], rho=config["rho"])
    pedal_force = pedal_force_at(config)

    def rhs(t, y):
        return braking_rhs(t, y, params, env, pedal_force(t))

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
    return sol, vehicle, params, pedal_force


def compute_wheel_series(sol, params, env, pedal_force):
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
    Fx_total = np.zeros_like(t)
    F_drag = np.zeros_like(t)
    ax = np.zeros_like(t)
    downforce = np.zeros_like(t)
    pedal_force_series = np.zeros_like(t)

    def wheel_value(value, key):
        if isinstance(value, dict):
            return value[key]
        return value

    for i in range(t.size):
        df = 0.5 * env.rho * params.Cl * params.A_ref * v[i] * v[i]
        downforce[i] = df
        aero_f = params.aero_balance * df
        aero_r = (1.0 - params.aero_balance) * df
        loads = normal_loads_from_pitch(
            theta=theta[i],
            m=params.pitch.m,
            g=params.g,
            a=params.a,
            b=params.b,
            k_wf=params.k_wf,
            k_wr=params.k_wr,
            aero_f=aero_f,
            aero_r=aero_r,
            k_tf=params.k_tf,
            k_tr=params.k_tr,
        )
        pedal_force_series[i] = pedal_force(t[i])
        torque = brake_torque_from_pedal(pedal_force_series[i], params, {k: omega[k][i] for k in omega})
        for key in omega:
            Re = wheel_value(params.Re, key)
            kappa[key][i] = slip_ratio(v[i], omega[key][i], Re)
            Fz[key][i] = loads[key]
            Fx[key][i] = tire_fx(
                kappa[key][i],
                Fz[key][i],
                env.mu,
                params.C_kappa,
                params.tir_params,
            )
            Tbr[key][i] = torque[key]
        Fx_total[i] = sum(Fx[key][i] for key in omega)
        F_drag[i] = -0.5 * env.rho * params.Cd * params.A_ref * v[i] * abs(v[i])
        ax[i] = (Fx_total[i] + F_drag[i]) / params.pitch.m

    return t, v, theta, omega, kappa, Fx, Fz, Tbr, Fx_total, F_drag, ax, downforce, pedal_force_series


def save_plots(out_dir: Path, series, show: bool = False):
    t, v, theta, omega, kappa, Fx, Fz, Tbr, Fx_total, F_drag, ax, downforce, pedal_force_series = series
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

    plt.figure(figsize=(9, 6))
    plt.subplot(3, 1, 1)
    plt.plot(t, Fx_total, label="Fx_total")
    plt.plot(t, F_drag, label="F_drag")
    plt.ylabel("Force [N]")
    plt.grid(True)
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(t, ax, label="ax")
    plt.ylabel("Accel [m/s^2]")
    plt.grid(True)
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(t, downforce, label="Downforce")
    plt.ylabel("Downforce [N]")
    plt.xlabel("Time [s]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "longitudinal_forces.png", dpi=150)
    if not show:
        plt.close()

    plt.figure(figsize=(9, 6))
    for key in Tbr:
        plt.plot(t, Tbr[key], label=f"T_brake {key}")
    plt.ylabel("Brake torque [N*m]")
    plt.xlabel("Time [s]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "brake_torque.png", dpi=150)
    if not show:
        plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(t, pedal_force_series, label="Pedal force")
    plt.ylabel("Pedal force [N]")
    plt.xlabel("Time [s]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "pedal_force.png", dpi=150)
    if not show:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run braking sim and generate plots.")
    parser.add_argument("--show", action="store_true", help="Show plots in an interactive window.")
    parser.add_argument("--params", type=str, help="Path to YAML vehicle params.")
    parser.add_argument("--init", type=str, help="Path to YAML simulation config.")
    parser.add_argument("--dump-config", action="store_true", help="Print parsed vehicle and sim config.")
    args = parser.parse_args()

    if args.init:
        config = load_sim_config(Path(args.init))
    else:
        config = default_sim_config()

    if args.params:
        vehicle = load_vehicle(Path(args.params))
        sol, vehicle, params, pedal_force = run_sim(vehicle, config)
    else:
        sol, vehicle, params, pedal_force = run_sim(default_vehicle(), config)
    env = EnvironmentParams(mu=config["mu"], rho=config["rho"])
    if args.dump_config:
        print("Parsed init config:", config)
        print("Parsed vehicle params:", vehicle)
    series = compute_wheel_series(sol, params, env, pedal_force)
    out_dir = Path("brakesim/plots")
    save_plots(out_dir, series, show=args.show)
    if args.show:
        plt.show()
    print(f"Saved plots to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
