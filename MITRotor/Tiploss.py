from typing import Protocol
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .Geometry import BEMGeometry
    from .RotorDefinition import RotorDefinition
    from .BEM import AerodynamicProperties


def NoTiploss(
    aero_props: "AerodynamicProperties",
    pitch: float,
    tsr: float,
    yaw: float,
    rotor: "RotorDefinition",
    geometry: "BEMGeometry",
):
    return np.ones_like(geometry.mu)


def Prandtl(
    aero_props: "AerodynamicProperties",
    pitch: float,
    tsr: float,
    yaw: float,
    rotor: "RotorDefinition",
    geometry: "BEMGeometry",
):
    phi = geometry.annulus_average(aero_props.phi)
    f = rotor.N_blades / 2 * (1 - geometry.mu) / (geometry.mu * np.abs(np.sin(phi)))
    F = 2 / np.pi * np.arccos(np.clip(np.exp(-np.clip(f, -100, 100)), -1, 1))
    return np.maximum(F, 0.01)


def PrandtlRootTip(
    aero_props: "AerodynamicProperties",
    pitch: float,
    tsr: float,
    yaw: float,
    rotor: "RotorDefinition",
    geometry: "BEMGeometry",
):
    phi = geometry.annulus_average(aero_props.phi)
    R_hub = rotor.hub_radius / rotor.R
    f_tip = rotor.N_blades / 2 * (1 - geometry.mu) / (np.maximum(geometry.mu, 0.0001) * np.abs(np.sin(phi)))
    F_tip = 2 / np.pi * np.arccos(np.clip(np.exp(-np.clip(f_tip, -100, 100)), -1, 1))
    f_hub = rotor.N_blades / 2 * (geometry.mu - R_hub) / (np.maximum(geometry.mu, 0.0001) * np.abs(np.sin(phi)))
    F_hub = 2 / np.pi * np.arccos(np.clip(np.exp(-np.clip(f_hub, -100, 100)), -1, 1))

    return np.maximum(F_hub * F_tip, 0.00001)
