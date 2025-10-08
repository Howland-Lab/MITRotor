from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .Geometry import BEMGeometry
    from .RotorDefinition import RotorDefinition
    from .BEMSolver import AerodynamicProperties

__all__ = ["TipLossModel", "NoTipLoss", "PrandtlTipLoss"]


class TipLossModel(ABC):
    @abstractmethod
    def __call__(
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geometry: "BEMGeometry",
    ) -> ArrayLike:
        ...


class NoTipLoss(TipLossModel):
    def __call__(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geometry: "BEMGeometry",
    ):
        return np.ones_like(geometry.mu_mesh)


class PrandtlTipLoss(TipLossModel):
    def __init__(self, root_loss: bool = True):
        self.root_loss = root_loss

    def __call__(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geometry: "BEMGeometry",
    ):
        phi = aero_props.phi
        R_hub = rotor.hub_radius / rotor.R
        f_tip = (
            rotor.N_blades / 2 * (1 - geometry.mu_mesh) / (np.maximum(geometry.mu_mesh, 0.0001) * np.abs(np.sin(phi)))
        )
        F_tip = 2 / np.pi * np.arccos(np.clip(np.exp(-np.clip(f_tip, -100, 100)), -0.9999, 0.9999))

        if self.root_loss:
            f_hub = (
                rotor.N_blades
                / 2
                * (geometry.mu_mesh - R_hub)
                / (np.maximum(geometry.mu_mesh, 0.0001) * np.abs(np.sin(phi)))
            )
            F_hub = 2 / np.pi * np.arccos(np.clip(np.exp(-np.clip(f_hub, -100, 100)), -0.9999, 0.9999))

            return F_hub * F_tip

        else:
            return F_tip
        
class PrandtlTipLoss2(TipLossModel):
    def __init__(self, root_loss: bool = True):
        self.root_loss = root_loss

    def __call__(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geometry: "BEMGeometry",
    ):
        phi = aero_props.phi
        R_hub = rotor.hub_radius / rotor.R
        f_tip = (
             rotor.N_blades / 2 * (1 - geometry.mu_mesh)  * np.sqrt(1 + tsr**2)
        )
        F_tip = 2 / np.pi * np.arccos(np.clip(np.exp(-np.clip(f_tip, -100, 100)), -0.9999, 0.9999))

        if self.root_loss:
            f_hub = (
                rotor.N_blades
                / 2
                * (geometry.mu_mesh - R_hub)
                / (np.maximum(geometry.mu_mesh, 0.0001) * np.abs(np.sin(phi)))
            )
            F_hub = 2 / np.pi * np.arccos(np.clip(np.exp(-np.clip(f_hub, -100, 100)), -0.9999, 0.9999))

            return F_hub * F_tip

        else:
            return F_tip
        

class ShenTipLoss(TipLossModel):
    def __init__(self, root_loss: bool = True):
        self.root_loss = root_loss

    def __call__(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geometry: "BEMGeometry",
    ):
        phi = aero_props.phi
        R_hub = rotor.hub_radius / rotor.R
        f_tip = (
            rotor.N_blades / 2 * (1 - geometry.mu_mesh) / (np.maximum(geometry.mu_mesh, 0.0001) * np.abs(np.sin(phi)))
        )
        g = np.exp(-0.125 * (rotor.N_blades * tsr - 21)) + 0.1

        F_tip = 2 / np.pi * np.arccos(np.clip(np.exp(-np.clip(g*f_tip, -100, 100)), -0.9999, 0.9999))

        if self.root_loss:
            f_hub = (
                rotor.N_blades
                / 2
                * (geometry.mu_mesh - R_hub)
                / (np.maximum(geometry.mu_mesh, 0.0001) * np.abs(np.sin(phi)))
            )
            F_hub = 2 / np.pi * np.arccos(np.clip(np.exp(-np.clip(f_hub, -100, 100)), -0.9999, 0.9999))

            return F_hub * F_tip

        else:
            return F_tip
        

