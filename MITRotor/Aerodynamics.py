from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from .RotorDefinition import RotorDefinition
from .Geometry import BEMGeometry

__all__ = [
    "AerodynamicModel",
    "AerodynamicProperties",
    "DefaultAerodynamics",
    "KraghAerodynamics",
]


@dataclass
class AerodynamicProperties:
    """
    Data class representing aerodynamic properties.

    Attributes (on radial grid):
        an (ArrayLike): Axial induction
        aprime (ArrayLike): Tangential induction
        solidity (ArrayLike): Blade solidity

    Attributes (on polar grid):
        U (ArrayLike): Inflow velocity.
        wdir (ArrayLike): Inflow direction.
        Vax (ArrayLike): Blade element axial velocity.
        Vtan (ArrayLike): Blade element tangential velocity.
        aoa (ArrayLike): Blade element angle of attack.
        Cl (ArrayLike): Blade element lift coefficient.
        Cd (ArrayLike): Blade element drag coefficient.
        F (Optional[ArrayLike]): Blade element tip loss (optional).

    Properties:
        W: Blade element inflow magnitude.
        phi: Blade element inflow direction
        Ctan: Blade element tangengial force coefficient.
        Cax: Blade element axial force coefficient.
    """

    # Radial grid
    an: ArrayLike
    aprime: ArrayLike
    solidity: ArrayLike
    # Full grid
    U: ArrayLike
    wdir: ArrayLike
    Vax: ArrayLike
    Vtan: ArrayLike
    aoa: ArrayLike
    Cl: ArrayLike
    Cd: ArrayLike
    F: Optional[ArrayLike] = None

    def __post_init__(self):
        pass

    @cached_property
    def W(self):
        """
        Blade element inflow magnitude.
        """
        return np.sqrt(self.Vax**2 + self.Vtan**2)

    @cached_property
    def phi(self):
        """
        Blade element inflow direction.
        """
        return np.arctan2(self.Vax, self.Vtan)

    @cached_property
    def Ctan(self):
        """
        Blade element tangengial force coefficient.
        """
        return self.Cl * np.sin(self.phi) - self.Cd * np.cos(self.phi)

    @cached_property
    def Cax(self):
        """
        Blade element axial force coefficient.
        """
        return self.Cl * np.cos(self.phi) + self.Cd * np.sin(self.phi)


class AerodynamicModel(ABC):
    @abstractmethod
    def __call__(
        self,
        an: ArrayLike,
        aprime: ArrayLike,
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: RotorDefinition,
        geom: BEMGeometry,
        U: ArrayLike,
        wdir: ArrayLike,
    ) -> AerodynamicProperties:
        """
        Performs the aerodynamic calculations in a blade-element code using the
        method outlined in Howland et al. 2020. (Influence of atmospheric conditions
        on the power production of utility-scale wind turbines in yaw misalignment)

        Args:
            an (ArrayLike): Axial induction radial profile.
            aprime (ArrayLike): tangengial induction radial profile.
            pitch (float): blade pitch angle [rad].
            tsr (float): Rotor tip-speed ratio.
            yaw (float): Rotor yaw angle [rad].
            rotor (RotorDefinition): Turbine rotor definition object.
            geom (BEMGeometry): Blade element geometry object.
            U (ArrayLike): Inflow velocity on polar grid.
            wdir (ArrayLike): Inflow direction on polar grid.

        Returns:
            AerodynamicProperties: Calculated aerodynamic properties stored in AerodynamicProperties object.

        """
        ...


class KraghAerodynamics(AerodynamicModel):
    def __call__(
        self,
        an: ArrayLike,
        aprime: ArrayLike,
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: RotorDefinition,
        geom: BEMGeometry,
        U: ArrayLike,
        wdir: ArrayLike,
    ) -> AerodynamicProperties:
        """
        Performs the aerodynamic calculations in a blade-element code using the
        method outlined in Howland et al. 2020. (Influence of atmospheric conditions
        on the power production of utility-scale wind turbines in yaw misalignment)

        Args:
            an (ArrayLike): Axial induction radial profile.
            aprime (ArrayLike): tangengial induction radial profile.
            pitch (float): blade pitch angle [rad].
            tsr (float): Rotor tip-speed ratio.
            yaw (float): Rotor yaw angle [rad].
            rotor (RotorDefinition): Turbine rotor definition object.
            geom (BEMGeometry): Blade element geometry object.
            U (ArrayLike): Inflow velocity on polar grid.
            wdir (ArrayLike): Inflow direction on polar grid.

        Returns:
            AerodynamicProperties: Calculated aerodynamic properties stored in AerodynamicProperties object.

        """
        local_yaw = wdir - yaw

        Vax = U * ((1 - an) * np.cos(local_yaw * np.cos(geom.theta_mesh)) * np.cos(local_yaw * np.sin(geom.theta_mesh)))
        Vtan = (1 + aprime) * tsr * geom.mu_mesh - U * (1 - an) * np.cos(local_yaw * np.sin(geom.theta_mesh)) * np.sin(
            local_yaw * np.cos(geom.theta_mesh)
        )

        phi = np.arctan2(Vax, Vtan)
        aoa = phi - rotor.twist(geom.mu_mesh) - pitch
        aoa = np.clip(aoa, -np.pi / 2, np.pi / 2)

        Cl, Cd = rotor.clcd(geom.mu_mesh, aoa)

        solidity = rotor.solidity(geom.mu_mesh)

        aero_props = AerodynamicProperties(
            an, aprime, solidity, U * np.ones(geom.shape), wdir * np.ones(geom.shape), Vax, Vtan, aoa, Cl, Cd
        )

        return aero_props


class DefaultAerodynamics(AerodynamicModel):
    def __call__(
        self,
        an: ArrayLike,
        aprime: ArrayLike,
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: RotorDefinition,
        geom: BEMGeometry,
        U: ArrayLike,
        wdir: ArrayLike,
    ) -> AerodynamicProperties:
        """
        Performs the aerodynamic calculations in a blade-element code using the
        method outlined in Howland et al. 2020. (Influence of atmospheric conditions
        on the power production of utility-scale wind turbines in yaw misalignment)

        Args:
            an (ArrayLike): Axial induction radial profile.
            aprime (ArrayLike): tangengial induction radial profile.
            pitch (float): blade pitch angle [rad].
            tsr (float): Rotor tip-speed ratio.
            yaw (float): Rotor yaw angle [rad].
            rotor (RotorDefinition): Turbine rotor definition object.
            geom (BEMGeometry): Blade element geometry object.
            U (ArrayLike): Inflow velocity on polar grid.
            wdir (ArrayLike): Inflow direction on polar grid.

        Returns:
            AerodynamicProperties: Calculated aerodynamic properties stored in AerodynamicProperties object.

        """
        local_yaw = -yaw

        Vax = U * ((1 - an) * np.cos(local_yaw))
        Vtan = (1 + aprime) * tsr * geom.mu_mesh - U * np.cos(geom.theta_mesh) * np.sin(
            local_yaw
        )

        phi = np.arctan2(Vax, Vtan)
        aoa = phi - rotor.twist(geom.mu_mesh) - pitch
        aoa = np.clip(aoa, -np.pi / 2, np.pi / 2)

        Cl, Cd = rotor.clcd(geom.mu_mesh, aoa)

        solidity = rotor.solidity(geom.mu_mesh)

        aero_props = AerodynamicProperties(
            an, aprime, solidity, U * np.ones(geom.shape), wdir * np.ones(geom.shape), Vax, Vtan, aoa, Cl, Cd
        )

        return aero_props
