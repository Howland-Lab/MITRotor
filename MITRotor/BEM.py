from functools import wraps
from typing import Tuple, Optional, Callable, Protocol, Literal
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from . import ThrustInduction, Tiploss
from .RotorDefinition import RotorDefinition
from .Geometry import BEMGeometry
from .Aerodynamics import AerodynamicProperties, HowlandAerodynamics
from .TangentialInduction import calc_aprime
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
    def wrapper(self, grid: Literal["averaged", "full", "radial"] = "averaged", **kwargs):
        # Assuming the function returns a grid of values
        quantity = func(self, grid, **kwargs)

        if grid == "full":
            return quantity  # No averaging

        elif grid == "azim":
            raise NotImplementedError

        elif grid == "radial":
            return self.geom.annulus_average(quantity)

        elif grid == "averaged":
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
    def a(self, grid: Literal["averaged", "full", "radial"] = "averaged"):
        return self.aero_props.an

    @radialgrid
    def aprime(self, grid: Literal["averaged", "full", "radial"] = "averaged"):
        return self.aero_props.aprime

    @radialgrid
    def solidity(self, grid: Literal["averaged", "full", "radial"] = "averaged"):
        return self.aero_props.solidity

    @fullgrid
    def U(self, grid: Literal["averaged", "full", "radial"] = "averaged"):
        return self.aero_props.U

    @fullgrid
    def wdir(self, grid: Literal["averaged", "full", "radial"] = "averaged"):
        return self.aero_props.wdir

    @fullgrid
    def Vax(self, grid: Literal["averaged", "full", "radial"] = "averaged"):
        return self.aero_props.Vax

    @fullgrid
    def Vtan(self, grid: Literal["averaged", "full", "radial"] = "averaged"):
        return self.aero_props.Vtan

    @fullgrid
    def W(self, grid: Literal["averaged", "full", "radial"] = "averaged"):
        return self.aero_props.W

    @fullgrid
    def phi(self, grid: Literal["averaged", "full", "radial"] = "averaged"):
        return self.aero_props.phi

    @fullgrid
    def aoa(self, grid: Literal["averaged", "full", "radial"] = "averaged"):
        return self.aero_props.aoa

    @fullgrid
    def Cl(self, grid: Literal["averaged", "full", "radial"] = "averaged"):
        return self.aero_props.Cl

    @fullgrid
    def Cd(self, grid: Literal["averaged", "full", "radial"] = "averaged"):
        return self.aero_props.Cd

    @fullgrid
    def Cax(self, grid: Literal["averaged", "full", "radial"] = "averaged"):
        return self.aero_props.Cax

    @fullgrid
    def Ctan(self, grid: Literal["averaged", "full", "radial"] = "averaged"):
        return self.aero_props.Ctan

    @fullgrid
    def F(self, grid: Literal["averaged", "full", "radial"] = "averaged"):
        return self.aero_props.F

    def u4(self):
        return self._u4

    def v4(self):
        return self._v4

    @radialgrid
    def Cp(self, grid=None):
        dCp = (
            self.tsr
            * self.solidity(grid="radial")
            * self.geom.mu
            * self.W(grid="radial") ** 2
            * self.Ctan(grid="radial")
        )
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
        self.calc_aerodynamics = aero_model or HowlandAerodynamics
        self.calc_tiploss: TiplossModel = tiploss_model or Tiploss.PrandtlRootTip
        self.calc_an: ThrustInduction.ThrustInductionModel = Cta_method or ThrustInduction.RotorAveragedHeck()
        self.calc_aprime = aprime_model or calc_aprime

        # self._solidity = self.rotor.solidity(self.geometry.mu)

    def sample_points(self, yaw: float = 0.0) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        X, Y, Z = self.geometry.cartesian(yaw)
        return X, Y, Z

    def initial_guess(
        self, pitch: float, tsr: float, yaw: float = 0.0, U: ArrayLike = 1.0, wdir: ArrayLike = 0.0
    ) -> Tuple[ArrayLike, ...]:
        a = 1 / 3 * np.ones(self.geometry.Nr)
        aprime = np.zeros(self.geometry.Nr)

        return a, aprime

    def residual(
        self,
        x: Tuple[ArrayLike, ...],
        pitch: ArrayLike,
        tsr: ArrayLike,
        yaw: ArrayLike = 0.0,
        U: ArrayLike = 1.0,
        wdir: ArrayLike = 0.0,
    ) -> Tuple[ArrayLike, ...]:
        an, aprime = x

        aero_props = self.calc_aerodynamics(an, aprime, pitch, tsr, yaw, self.rotor, self.geometry, U, wdir)
        aero_props.F = self.calc_tiploss(aero_props, pitch, tsr, yaw, self.rotor, self.geometry)
        e_an = self.calc_an(aero_props, pitch, tsr, yaw, self.rotor, self.geometry) - an
        e_aprime = self.calc_aprime(aero_props, pitch, tsr, yaw, self.rotor, self.geometry) - aprime

        return e_an, e_aprime

    def post_process(self, result: FixedPointIterationResult, pitch, tsr, yaw, U=1.0, wdir=0.0) -> BEMSolution:
        an, aprime = result.x
        aero_props = self.calc_aerodynamics(an, aprime, pitch, tsr, yaw, self.rotor, self.geometry, U, wdir)
        aero_props.F = self.calc_tiploss(aero_props, pitch, tsr, yaw, self.rotor, self.geometry)

        return BEMSolution(pitch, tsr, yaw, aero_props, self.geometry, result.converged, result.niter)
