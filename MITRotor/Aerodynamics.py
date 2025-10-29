from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from .RotorDefinition import RotorDefinition
from .Geometry import BEMGeometry
from UnifiedMomentumModel.Utilities.Geometry import calc_eff_yaw

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
        F (ArrayLike): Blade element tip loss.

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
    F: ArrayLike = None

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
    def C_n(self):
        """
        Blade element axial blade force coefficient.
        """
        return self.Cl * np.cos(self.phi) + self.Cd * np.sin(self.phi)

    @cached_property
    def C_tan(self):
        """
        Blade element tangential blade force coefficient.
        """
        return self.Cl * np.sin(self.phi) - self.Cd * np.cos(self.phi)
    
    @cached_property
    def C_x(self):
        """
        Blade element axial area force coefficient.
        """
        return self.solidity * self.W**2 * self.C_n

    @cached_property
    def C_tau(self):
        """
        Blade element tangential area force coefficient.
        """
        return self.solidity * self.W**2 * self.C_tan
    
    @cached_property
    def C_x_corr(self):
        """
        Corrected blade element area axial force coefficient.
        """
        return self.C_x / self.F
    
    @cached_property
    def C_tau_corr(self):
        """
        Corrected blade element area tangential force coefficient.
        """
        return self.C_tau / self.F




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
        tilt: float = 0,
    ) -> AerodynamicProperties:
        """
        Performs the aerodynamic calculations in a blade-element code.

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
            tilt (float): Rotor tilt angle [rad].

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
        tilt: float = 0.0,
    ) -> AerodynamicProperties:
        """
        Performs the aerodynamic calculations in a blade-element code using the
        method outlined in Howland et al. 2020. (Influence of atmospheric conditions
        on the power production of utility-scale wind turbines in yaw misalignment),
        which builds on 2014 paper by Kragh and Hansen: https://doi.org/10.1002/we.1612.

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
            tilt (float): Rotor tilt angle [rad].

        Returns:
            AerodynamicProperties: Calculated aerodynamic properties stored in AerodynamicProperties object.

        """
        if tilt != 0:
            raise ValueError("Tilt not supported by the KraghAerodynamics model. Use DefaultAerodynamics.")
        local_yaw = wdir - yaw

        Vax = (
            U
            * (1 - an)
            * np.cos(local_yaw * np.cos(geom.theta_mesh))
            * np.cos(local_yaw * np.sin(geom.theta_mesh))
        )
        Vtan = (
            (1 + aprime) * tsr * geom.mu_mesh
            - U * (1 - an)
            * np.cos(local_yaw * np.sin(geom.theta_mesh))
            * np.sin(local_yaw * np.cos(geom.theta_mesh))
        )

        phi = np.arctan2(Vax, Vtan)
        aoa = phi - rotor.twist(geom.mu_mesh) - pitch
        aoa = np.clip(aoa, -np.pi / 2, np.pi / 2)

        Cl, Cd = rotor.clcd(geom.mu_mesh, aoa)

        solidity = rotor.solidity(geom.mu_mesh)

        aero_props = AerodynamicProperties(
            an = an,
            aprime = aprime,
            solidity = solidity,
            U = U * np.ones(geom.shape),
            wdir = wdir * np.ones(geom.shape),
            Vax = Vax,
            Vtan = Vtan,
            aoa = aoa,
            Cl = Cl,
            Cd = Cd,
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
        tilt: float = 0.0,
    ) -> AerodynamicProperties:
        """
        Performs the aerodynamic calculations in a blade-element code using the
        method outlined in the supplementary material in Liew et al., 2024:
        https://www.nature.com/articles/s41467-024-50756-5

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
        # calculate values in "yaw-only" frame
        local_yaw = -self.eff_yaw
        Vax = U * ((1 - an) * np.cos(local_yaw))
        Vtan = (
            (1 + aprime) * tsr * geom.mu_mesh
            - U * (1 - an)
            * np.cos(self.eff_theta_mesh)
            * np.sin(local_yaw)
        )

        phi = np.arctan2(Vax, Vtan)
        aoa = phi - rotor.twist(geom.mu_mesh) - pitch
        aoa = np.clip(aoa, -np.pi / 2, np.pi / 2)

        Cl, Cd = rotor.clcd(geom.mu_mesh, aoa)

        solidity = rotor.solidity(geom.mu_mesh)

        aero_props = AerodynamicProperties(
            an = an,
            aprime = aprime,
            solidity = solidity,
            U = U * np.ones(geom.shape),
            wdir = wdir * np.ones(geom.shape),
            Vax = Vax,
            Vtan = Vtan,
            aoa = aoa,
            Cl = Cl,
            Cd = Cd,
        )

        return aero_props