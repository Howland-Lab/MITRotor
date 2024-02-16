import numpy as np
from numpy.typing import ArrayLike

from .Aerodynamics import AerodynamicProperties
from .Geometry import BEMGeometry
from .RotorDefinition import RotorDefinition

__all__ = ["calc_aprime"]


def no_aprime(
    aero_props: AerodynamicProperties,
    pitch: float,
    tsr: float,
    yaw: float,
    rotor: RotorDefinition,
    geom: BEMGeometry,
):
    return np.zeros_like(aero_props.an)


def calc_aprime(
    aero_props: AerodynamicProperties, pitch: float, tsr: float, yaw: float, rotor: RotorDefinition, geom: BEMGeometry
) -> ArrayLike:
    tangential_integral = geom.annulus_average(aero_props.W**2 * aero_props.Ctan)

    aprime = (
        aero_props.solidity
        / (4 * np.maximum(geom.mu, 0.1) ** 2 * tsr * (1 - aero_props.an) * np.cos(yaw))
        * tangential_integral
    )

    return aprime
