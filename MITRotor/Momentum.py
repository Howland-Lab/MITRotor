from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal
import numpy as np
from numpy.typing import ArrayLike

from UnifiedMomentumModel import Momentum as UMM

if TYPE_CHECKING:
    from .Geometry import BEMGeometry
    from .RotorDefinition import RotorDefinition
    from .Aerodynamics import AerodynamicProperties

__all__ = [
    "MomentumModel",
    "ConstantInduction",
    "ClassicalMomentum",
    "HeckMomentum",
    "UnifiedMomentum",
    "MadsenMomentum",
]


class MomentumModel(ABC):
    @abstractmethod
    def compute_induction(self, Cx: ArrayLike, yaw: float) -> ArrayLike:
        ...

    @abstractmethod
    def __call__(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
    ) -> ArrayLike:
        ...
    
    def _func_rotor(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
    ) -> ArrayLike:
        
        rotor_avg_axial_force = (
            geom.rotor_average(
                geom.annulus_average(
                    aero_props.C_x_corr
                    )
                    )
        )

        return self.compute_induction(rotor_avg_axial_force, yaw)



    def _func_annulus(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
    ) -> ArrayLike:
        
        annulus_avg_axial_force = (
            
                geom.annulus_average(
                    aero_props.C_x_corr
                    )
                    )[:, None] * np.ones(geom.shape)
        

        return self.compute_induction(annulus_avg_axial_force, yaw)

    def _func_sector(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
    ) -> ArrayLike:
        axial_force = aero_props.C_x_corr

        return self.compute_induction(axial_force, yaw)

    def __call__(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
    ) -> ArrayLike:
        an = self._func(aero_props, pitch, tsr, yaw, rotor, geom)
        return an


class ConstantInduction(MomentumModel):
    def __init__(self, a = 1/3):
        self.a = a

    def _func(self, aero_props, pitch, tsr, yaw, rotor, geom) -> ArrayLike:
        return self.a * np.ones_like(yaw)



class ClassicalMomentum(MomentumModel):
    def compute_induction(self, Cx, yaw):
        return 0.5 * (1 - np.sqrt(1 - Cx))



class HeckMomentum(MomentumModel):
    def __init__(
        self, averaging: Literal["sector", "annulus", "rotor"] = "rotor", ac: float = 1 / 3, v4_correction: float = 1.0
    ):
        self.v4_correction = v4_correction
        self.ac = ac
        if averaging == "rotor":
            self._func = self._func_rotor
        elif averaging == "annulus":
            self._func = self._func_annulus
        elif averaging == "sector":
            self._func = self._func_sector
        else:
            raise ValueError(f"Averaging method {averaging} not found for HeckMomentum model.")
        self.averaging = averaging

    def compute_induction(self, Cx: ArrayLike, yaw: float) -> ArrayLike:
        Ctc = 4 * self.ac * (1 - self.ac) / (1 + 0.25 * (1 - self.ac) ** 2 * np.sin(yaw) ** 2)
        slope = (16 * (1 - self.ac) ** 2 * np.sin(yaw) ** 2 - 128 * self.ac + 64) / (
            (1 - self.ac) ** 2 * np.sin(yaw) ** 2 + 4
        ) ** 2

        a = (2 * Cx - 4 + np.sqrt(-(Cx**2) * np.sin(yaw) ** 2 - 16 * Cx + 16)) / (
            -4 + np.sqrt(-(Cx**2) * np.sin(yaw) ** 2 - 16 * Cx + 16)
        )

        if np.iterable(Cx):
            mask = Cx > Ctc
            if np.any(mask):
                a[mask] = (Cx[mask] - Ctc) / slope + self.ac

        return a


class UnifiedMomentum(MomentumModel):
    def __init__(self, averaging: Literal["sector", "annulus", "rotor"] = "rotor", beta=0.1403):
        self.beta = beta

        if averaging == "rotor":
            self._func = self._func_rotor
        elif averaging == "annulus":
            self._func = self._func_annulus
        elif averaging == "sector":
            self._func = self._func_sector
        else:
            raise ValueError(f"Averaging method {averaging} not found for UnifiedMomentum model.")
        self.averaging = averaging

        self.model_Ct = UMM.ThrustBasedUnified(beta=beta)

    def compute_induction(self, Cx: ArrayLike, yaw: float) -> ArrayLike:
        return self.model_Ct(Cx, yaw).a


class MadsenMomentum(MomentumModel):
    """
    Madsen momentum model based on Madsen 2020 in Wind Energy Science.
    """
    def __init__(self, averaging: Literal["sector", "annulus", "rotor"] = "rotor"):
        if averaging == "rotor":
            self._func = self._func_rotor
        elif averaging == "annulus":
            self._func = self._func_annulus
        elif averaging == "sector":
            self._func = self._func_sector
        else:
            raise ValueError(f"Averaging method {averaging} not found for UnifiedMomentum model.")
        self.averaging = averaging


    def compute_induction(self, Cx: ArrayLike, yaw: float) -> ArrayLike:
        an = Cx**3 * 0.0883 + Cx**2 * 0.0586 + Cx * 0.2460
        return an
