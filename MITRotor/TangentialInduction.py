from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import ArrayLike

from .Aerodynamics import AerodynamicProperties
from .Geometry import BEMGeometry
from .RotorDefinition import RotorDefinition

__all__ = [
    "TangentialInductionModel",
    "NoTangentialInduction",
    "DefaultTangentialInduction",
]


class TangentialInductionModel(ABC):
    @abstractmethod
    def __call__(
        self,
        aero_props: AerodynamicProperties,
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: RotorDefinition,
        geom: BEMGeometry,
        tilt: float = 0.0,
    ) -> ArrayLike:
        ...


class NoTangentialInduction(TangentialInductionModel):
    def __call__(
        self,
        aero_props: AerodynamicProperties,
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: RotorDefinition,
        geom: BEMGeometry,
        tilt: float = 0.0,
    ) -> ArrayLike:
        return np.zeros_like(aero_props.an)


class DefaultTangentialInduction(TangentialInductionModel):
    def __call__(
        self,
        aero_props: AerodynamicProperties,
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: RotorDefinition,
        geom: BEMGeometry,
        tilt: float = 0.0,
    ) -> ArrayLike:
        

        aprime = (
            np.clip(aero_props.C_tau_corr, -2, 2)
            / (4 * np.maximum(geom.mu_mesh, 0.1) ** 2 * tsr * (1 - aero_props.an) * np.cos(yaw))
        )
        # TODO: add in effects of tilt

        return aprime
