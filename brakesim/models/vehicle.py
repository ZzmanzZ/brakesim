from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from pathlib import Path

from brakesim.models.braking import BrakingParams
from brakesim.models.pacejka import parse_tir, TirData
from brakesim.models.pitch import PitchParams


@dataclass(frozen=True)
class VehicleParams:
    m: float
    g: float
    a: float
    b: float
    h_cg: float
    Iyy: float
    k_theta: float
    c_theta: float
    k_wf: float
    k_wr: float
    Re: float | Mapping[str, float]
    Iw: float | Mapping[str, float]
    C_kappa: float
    Cl: float
    Cd: float
    A_ref: float
    aero_balance: float
    mc_area_f: float | None = None
    mc_area_r: float | None = None
    caliper_piston_area: float | Mapping[str, float] | None = None
    caliper_piston_count: int | Mapping[str, int] | None = None
    pedal_ratio: float | None = None
    pad_mu_static: float | Mapping[str, float] | None = None
    pad_mu_kinetic: float | Mapping[str, float] | None = None
    r_brake: float | Mapping[str, float] | None = None
    balance_bar: float | None = None
    tir_file: str | None = None
    k_tf: float | None = None
    k_tr: float | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "VehicleParams":
        return cls(**data)

    def pitch_params(self) -> PitchParams:
        return PitchParams(
            m=self.m,
            h_cg=self.h_cg,
            Iyy=self.Iyy,
            k_theta=self.k_theta,
            c_theta=self.c_theta,
        )

    def braking_params(self) -> BrakingParams:
        tir_params: TirData | None = None
        if self.tir_file:
            tir_params = parse_tir(Path(self.tir_file))
        return BrakingParams(
            pitch=self.pitch_params(),
            g=self.g,
            a=self.a,
            b=self.b,
            k_wf=self.k_wf,
            k_wr=self.k_wr,
            Re=self.Re,
            Iw=self.Iw,
            C_kappa=self.C_kappa,
            Cl=self.Cl,
            Cd=self.Cd,
            A_ref=self.A_ref,
            aero_balance=self.aero_balance,
            pedal_ratio=self.pedal_ratio,
            mc_area_f=self.mc_area_f,
            mc_area_r=self.mc_area_r,
            caliper_piston_area=self.caliper_piston_area,
            caliper_piston_count=self.caliper_piston_count,
            pad_mu_static=self.pad_mu_static,
            pad_mu_kinetic=self.pad_mu_kinetic,
            r_brake=self.r_brake,
            balance_bar=self.balance_bar,
            tir_params=tir_params,
            k_tf=self.k_tf,
            k_tr=self.k_tr,
        )
