from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from brakesim.models.braking import BrakingParams
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
    mu: float
    C_kappa: float
    aero_f: float = 0.0
    aero_r: float = 0.0
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
        return BrakingParams(
            pitch=self.pitch_params(),
            g=self.g,
            a=self.a,
            b=self.b,
            k_wf=self.k_wf,
            k_wr=self.k_wr,
            Re=self.Re,
            Iw=self.Iw,
            mu=self.mu,
            C_kappa=self.C_kappa,
            aero_f=self.aero_f,
            aero_r=self.aero_r,
            k_tf=self.k_tf,
            k_tr=self.k_tr,
        )
