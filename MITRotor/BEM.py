from functools import wraps, cached_property
from typing import Tuple, Optional, Callable, Protocol
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from . import ThrustInduction, Tiploss
from .RotorDefinition import RotorDefinition
from .Geometry import BEMGeometry
from UnifiedMomentumModel.Utilities.FixedPointIteration import adaptivefixedpointiteration, FixedPointIterationResult
from UnifiedMomentumModel.Momentum import Heck


class TiplossModel(Protocol):
    def __call__(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geometry: "BEMGeometry",
    ):
        ...


def fullgrid(func):
    @wraps(func)
    def wrapper(self, grid=None, **kwargs):
        # Assuming the function returns a grid of values
        quantity = func(self, grid, **kwargs)

        if grid == "full":
            return quantity  # No averaging

        elif grid == "azim":
            raise NotImplementedError

        elif grid == "radial":
            return self.geom.annulus_average(quantity)

        elif grid is None:
            return self.geom.rotor_average(self.geom.annulus_average(quantity))

        else:
            raise ValueError(f"Unsupported average type: {grid}")

    return wrapper


def radialgrid(func):
    @wraps(func)
    def wrapper(self, grid=None, **kwargs):
        # Assuming the function returns a grid of values
        quantity = func(self, grid, **kwargs)

        if grid == "full":
            raise ValueError("Cannot return radial quantity on a full polar grid.")

        elif grid == "azim":
            raise NotImplementedError

        elif grid == "radial":
            return quantity

        elif grid is None:
            return self.geom.rotor_average(quantity)

        else:
            raise ValueError(f"Unsupported average type: {grid}")

    return wrapper


@dataclass
class AerodynamicProperties:
    # Radial grid
    an: ArrayLike
    aprime: ArrayLike
    solidity: ArrayLike
    # Full grid
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
        return np.sqrt(self.Vax**2 + self.Vtan**2)

    @cached_property
    def phi(self):
        return np.arctan2(self.Vax, self.Vtan)

    @cached_property
    def Ctan(self):
        return self.Cl * np.sin(self.phi) - self.Cd * np.cos(self.phi)

    @cached_property
    def Cax(self):
        return self.Cl * np.cos(self.phi) + self.Cd * np.sin(self.phi)


@dataclass
class BEMSolution:
    pitch: float
    tsr: float
    yaw: float
    aero_props: AerodynamicProperties = field(repr=False)
    geom: BEMGeometry = field(repr=False)
    converged: bool
    niter: int

    def __post_init__(self):
        sol = Heck()(self.Ctprime(), self.yaw)
        self._u4, self._v4 = sol.u4, sol.v4

    @radialgrid
    def a(self, grid=None):
        return self.aero_props.an

    @radialgrid
    def aprime(self, grid=None):
        return self.aero_props.aprime

    @radialgrid
    def solidity(self, grid=None):
        return self.aero_props.solidity

    @fullgrid
    def Vax(self, grid=None):
        return self.aero_props.Vax

    @fullgrid
    def Vtan(self, grid=None):
        return self.aero_props.Vtan

    @fullgrid
    def W(self, grid=None):
        return self.aero_props.W

    @fullgrid
    def phi(self, grid=None):
        return self.aero_props.phi

    @fullgrid
    def aoa(self, grid=None):
        return self.aero_props.aoa

    @fullgrid
    def Cl(self, grid=None):
        return self.aero_props.Cl

    @fullgrid
    def Cd(self, grid=None):
        return self.aero_props.Cd

    @fullgrid
    def Cax(self, grid=None):
        return self.aero_props.Cax

    @fullgrid
    def Ctan(self, grid=None):
        return self.aero_props.Ctan

    @fullgrid
    def F(self, grid=None):
        return self.aero_props.F

    def u4(self):
        return self._u4

    def v4(self):
        return self._v4

    @radialgrid
    def Cp(self, grid=None):
        dCp = self.tsr * self.solidity(grid="radial") * self.geom.mu * self.W(grid="radial") ** 2 * self.Ctan(grid="radial")
        return dCp

    @radialgrid
    def Ct(self, grid=None):
        _Ct = self.solidity(grid="radial") * self.W(grid="radial") ** 2 * self.Cax(grid="radial")
        return _Ct

    @radialgrid
    def Ctprime(self, grid=None):
        Ctprime = self.Ct(grid="radial") / ((1 - self.a(grid="radial")) ** 2 * np.cos(self.yaw) ** 2)
        return Ctprime

    @radialgrid
    def Cq(self, grid=None):
        return self.Cp(grid="radial") / self.tsr

    # @fullgrid
    # def Fax(self, U_inf: float, rho=1.293) -> ArrayLike:
    #     R = self.R
    #     dR = self.geometry.dmu * R
    #     A = self.geometry.mu_mesh * R * dR * self.geometry.dtheta

    #     return 0.5 * rho * U_inf**2 * self.Cax(doubledecomp=True) * A

    # @doubledecompose
    # def Ftan(self, U_inf: float, rho=1.293) -> ArrayLike:
    #     R = self.R
    #     dR = self.geometry.dmu * R
    #     A = self.geometry.mu_mesh * R * dR * self.geometry.dtheta

    #     return 0.5 * rho * U_inf**2 * self.Ctan(doubledecomp=True) * A

    # @decompose
    # def thrust(self, U_inf: float, rho=1.293) -> ArrayLike:
    #     R = self.R
    #     dR = self.geometry.dmu * R
    #     A = self.geometry.mu_mesh * R * dR * self.geometry.dtheta

    #     return 0.5 * rho * U_inf**2 * self.Ct(decomp=True) * A

    # @decompose
    # def power(self, U_inf: float, rho=1.293) -> ArrayLike:
    #     R = self.R
    #     dR = self.geometry.dmu * R
    #     A = self.geometry.mu_mesh * R * dR * self.geometry.dtheta

    #     return 0.5 * rho * U_inf**3 * self.Cp(decomp=True) * A

    # @decompose
    # def torque(self, U_inf: float, rho=1.293) -> ArrayLike:
    #     R = self.R
    #     rotor_speed = self.tsr * U_inf / R

    #     return self.power(U_inf, rho=rho, decomp=True) / rotor_speed

    # @doubledecompose
    # def REWS(self):
    #     return self.U


@adaptivefixedpointiteration(max_iter=500, relaxations=[0.25, 0.5, 0.96])
class BEM:
    """
    A generic BEM class which facilitates dependency injection for various models.
    Models which can be injected are:
    - rotor definition
    - BEM geometry
    - aerodynamic properties calculation method
    - tip loss method
    - axial induction calculation method
    - tangential induction calculation method
    """

    def __init__(
        self,
        rotor: RotorDefinition,
        geometry: Optional[BEMGeometry] = None,
        tiploss_model: Optional[TiplossModel] = None,
        Cta_method: Optional[ThrustInduction.ThrustInductionModel] = None,
        aprime_model: Optional[Callable] = None,
        aero_model: Optional[Callable] = None,
    ):
        self.rotor = rotor

        self.geometry: BEMGeometry = geometry or BEMGeometry(Nr=10, Ntheta=20)
        self.calc_aerodynamics = aero_model or update_aerodynamics
        self.calc_tiploss: TiplossModel = tiploss_model or Tiploss.PrandtlRootTip
        self.calc_an: ThrustInduction.ThrustInductionModel = Cta_method or ThrustInduction.RotorAveragedHeck()
        self.calc_aprime = aprime_model or calc_aprime

        # self._solidity = self.rotor.solidity(self.geometry.mu)

    def initial_guess(self, pitch: float, tsr: float, yaw: float = 0.0, windfield=None) -> Tuple[ArrayLike, ...]:
        a = 1 / 3 * np.ones(self.geometry.Nr)
        aprime = np.zeros(self.geometry.Nr)

        return a, aprime

    def residual(
        self,
        x: Tuple[ArrayLike, ...],
        pitch: ArrayLike,
        tsr: ArrayLike,
        yaw: ArrayLike = 0.0,
    ) -> Tuple[ArrayLike, ...]:
        an, aprime = x

        aero_props = self.calc_aerodynamics(an, aprime, pitch, tsr, yaw, self.rotor, self.geometry)
        aero_props.F = self.calc_tiploss(aero_props, pitch, tsr, yaw, self.rotor, self.geometry)
        e_an = self.calc_an(aero_props, pitch, tsr, yaw, self.rotor, self.geometry) - an
        e_aprime = self.calc_aprime(aero_props, pitch, tsr, yaw, self.rotor, self.geometry) - aprime

        return e_an, e_aprime

    def post_process(self, result: FixedPointIterationResult, pitch, tsr, yaw) -> BEMSolution:
        an, aprime = result.x
        aero_props = self.calc_aerodynamics(an, aprime, pitch, tsr, yaw, self.rotor, self.geometry)
        aero_props.F = self.calc_tiploss(aero_props, pitch, tsr, yaw, self.rotor, self.geometry)

        return BEMSolution(pitch, tsr, yaw, aero_props, self.geometry, result.converged, result.niter)


def update_aerodynamics(
    an: ArrayLike,
    aprime: ArrayLike,
    pitch: float,
    tsr: float,
    yaw: float,
    rotor: RotorDefinition,
    geom: BEMGeometry,
) -> AerodynamicProperties:
    wdir, U = 0, 1  # Todo: update this to take arbitrary wdir distributions
    local_yaw = wdir - yaw

    Vax = U * ((1 - np.expand_dims(an, axis=-1)) * np.cos(local_yaw * np.cos(geom.theta_mesh)) * np.cos(local_yaw * np.sin(geom.theta_mesh)))
    Vtan = (1 + np.expand_dims(aprime, axis=-1)) * tsr * geom.mu_mesh - U * (1 - np.expand_dims(an, axis=-1)) * np.cos(
        local_yaw * np.sin(geom.theta_mesh)
    ) * np.sin(local_yaw * np.cos(geom.theta_mesh))

    phi = np.arctan2(Vax, Vtan)
    aoa = phi - rotor.twist(geom.mu_mesh) - pitch
    aoa = np.clip(aoa, -np.pi / 2, np.pi / 2)

    Cl, Cd = rotor.clcd(geom.mu_mesh, aoa)

    solidity = rotor.solidity(geom.mu)

    aero_props = AerodynamicProperties(an, aprime, solidity, Vax, Vtan, aoa, Cl, Cd)

    return aero_props


def calc_aprime(
    aero_props: AerodynamicProperties, pitch: float, tsr: float, yaw: float, rotor: RotorDefinition, geom: BEMGeometry
) -> ArrayLike:
    tangential_integral = geom.annulus_average(aero_props.W**2 * aero_props.Ctan)

    aprime = (
        aero_props.solidity / (4 * np.maximum(geom.mu, 0.1) ** 2 * tsr * (1 - aero_props.an) * np.cos(yaw)) * tangential_integral
    )

    return aprime
