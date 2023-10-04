from functools import wraps
from typing import Tuple, Union

import numpy as np
import numpy.typing as npt

from . import ThrustInduction, Tiploss
from .RotorDefinition import RotorDefinition
from .Utilities import adaptivefixedpointiteration, fixedpointiteration


class _BEMSolverBase:
    def __init__(
        self,
        rotor: RotorDefinition,
        Cta_method: Union[str, ThrustInduction.CtaModel] = "Unified",
        tiploss: Union[str, Tiploss.TiplossModel] = "PrandtlRootTip",
        Nr=20,
        Ntheta=21,
    ):
        self.rotor = rotor

        self.Nr, self.Ntheta = Nr, Ntheta

        self.mu, self.theta, self.mu_mesh, self.theta_mesh = self.calc_gridpoints(
            Nr, Ntheta
        )

        self.Cta_func = ThrustInduction.build_cta_model(Cta_method)
        self.tiploss_func = Tiploss.build_tiploss_model(tiploss, rotor)

        self._solidity = self.rotor.solidity(self.mu_mesh)

    def solve(self):
        ...

    @classmethod
    def calc_gridpoints(cls, Nr: int, Ntheta: int) -> Tuple[npt.ArrayLike, ...]:
        mu = np.linspace(0.0, 1.0, Nr)
        theta = np.linspace(0.0, 2 * np.pi, Ntheta)

        theta_mesh, mu_mesh = np.meshgrid(theta, mu)

        return mu, theta, mu_mesh, theta_mesh

    def gridpoints_cart(self, yaw: float) -> Tuple[npt.ArrayLike, ...]:
        """
        Returns the grid point locations in cartesian coordinates
        nondimensionialized by rotor radius. Origin is located at hub center.

        Note: effect of yaw angle on grid points is not yet implemented.
        """
        # Probable sign error here.
        X = np.zeros_like(self.mu_mesh)
        Y = self.mu_mesh * np.sin(self.theta_mesh)  # lateral
        Z = self.mu_mesh * np.cos(self.theta_mesh)  # vertical

        return X, Y, Z

    def _sample_windfield(self, windfield):
        yaw = 0  # To do: change grid points based on yaw angle.
        _X, _Y, _Z = self.gridpoints_cart(yaw)
        Y = _Y
        Z = self.rotor.hub_height / self.rotor.R + _Z

        U = windfield.wsp(Y, Z)
        wdir = windfield.wdir(Y, Z)

        return U, wdir


def calc_gridpoints(Nr, Ntheta):
    mu = np.linspace(0.0, 0.999, Nr)
    theta = np.linspace(0.0, 2 * np.pi, Ntheta)

    theta_mesh, mu_mesh = np.meshgrid(theta, mu)

    return mu, theta, mu_mesh, theta_mesh


def decompose(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        decomp = kwargs.pop("decomp", False)
        X = func(self, *args, **kwargs)
        if not decomp:
            return rotor_average(self.mu, X)
        else:
            return X

    return wrapper


def doubledecompose(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        decomp = kwargs.pop("decomp", False)
        doubledecomp = kwargs.pop("doubledecomp", False)
        X = func(self, *args, **kwargs)
        if doubledecomp:
            return X
        if decomp:
            return annulus_average(self.theta_mesh, X)
        else:
            return rotor_average(self.mu, annulus_average(self.theta_mesh, X))

    return wrapper


def rotor_average(mu, X):
    # Takes annulus average quantities and performs rotor average

    X_rotor = 2 * np.trapz(X * mu, mu)
    return X_rotor


def annulus_average(theta_mesh, X):
    X_azim = 1 / (2 * np.pi) * np.trapz(X, theta_mesh, axis=-1)

    return X_azim


class BEMSolution:
    def __init__(
        self,
        mu: npt.ArrayLike,
        theta: npt.ArrayLike,
        mu_mesh: npt.ArrayLike,
        theta_mesh: npt.ArrayLike,
        pitch: npt.ArrayLike,
        tsr: npt.ArrayLike,
        yaw: npt.ArrayLike,
        U: npt.ArrayLike,
        wdir: npt.ArrayLike,
        R: float,
    ):
        self.mu, self.theta = mu, theta
        self.mu_mesh, self.theta_mesh = mu_mesh, theta_mesh
        self.Nr, self.Ntheta = len(mu), len(theta)

        self.pitch, self.tsr, self.yaw = pitch, tsr, yaw
        self.U, self.wdir = U, wdir

        self.R = R

        self._a = 1 / 3 * np.ones((self.Nr))
        self._aprime = np.zeros((self.Nr))
        self._Ctprime = np.zeros((self.Nr))
        self._u4 = np.zeros((self.Nr))
        self._v4 = np.zeros((self.Nr))
        self._dp = np.zeros((self.Nr))

        self.solidity = np.zeros((self.Nr))

        # aerodynamic
        self._phi = np.zeros((self.Nr, self.Ntheta))
        self._aoa = np.zeros((self.Nr, self.Ntheta))
        self._Vax = np.zeros((self.Nr, self.Ntheta))
        self._Vtan = np.zeros((self.Nr, self.Ntheta))
        self._W = np.zeros((self.Nr, self.Ntheta))
        self._Cax = np.zeros((self.Nr, self.Ntheta))
        self._Ctan = np.zeros((self.Nr, self.Ntheta))
        self._tiploss = np.zeros((self.Nr, self.Ntheta))

        # Convergence information
        self.sampled_Ctprimes = []
        self.converged = False

    def zero_nans(self):
        self._a[np.isnan(self._a)] = 0
        self._aprime[np.isnan(self._aprime)] = 0
        self._Ctprime[np.isnan(self._Ctprime)] = 0
        self._u4[np.isnan(self._u4)] = 0
        self._v4[np.isnan(self._v4)] = 0
        self._dp[np.isnan(self._dp)] = 0
        self._phi[np.isnan(self._phi)] = 0
        self._aoa[np.isnan(self._aoa)] = 0
        self._Vax[np.isnan(self._Vax)] = 0
        self._Vtan[np.isnan(self._Vtan)] = 0
        self._W[np.isnan(self._W)] = 0
        self._Cax[np.isnan(self._Cax)] = 0
        self._Ctan[np.isnan(self._Ctan)] = 0
        self._tiploss[np.isnan(self._tiploss)] = 0

    @decompose
    def a(self):
        return self._a

    @decompose
    def u4(self):
        return self._u4

    @decompose
    def v4(self):
        return self._v4

    @decompose
    def dp(self):
        return self._dp

    @decompose
    def Ctprime(self):
        return self._Ctprime

    @decompose
    def aprime(self):
        return self._aprime

    @decompose
    def Cp(self):
        integral = annulus_average(self.theta_mesh, self._W**2 * self._Ctan)
        dCp = self.tsr * self.solidity * self.mu * integral
        return dCp

    @decompose
    def Ct(self):
        dCt = self._Ctprime * (1 - self._a) ** 2 * np.cos(self.yaw) ** 2
        return dCt

    @decompose
    def Cq(self):
        return self.Cp(decomp=True) / self.tsr

    @doubledecompose
    def phi(self):
        return self._phi

    @doubledecompose
    def aoa(self):
        return self._aoa

    @doubledecompose
    def Vax(self):
        return self._Vax

    @doubledecompose
    def Vtan(self):
        return self._Vtan

    @doubledecompose
    def W(self):
        return self._W

    @doubledecompose
    def Cax(self):
        return self._Cax

    @doubledecompose
    def tiploss(self):
        return self._tiploss

    @doubledecompose
    def Ctan(self):
        return self._Ctan

    @doubledecompose
    def Cl(self):
        return self._Cl

    @doubledecompose
    def Cd(self):
        return self._Cd

    @doubledecompose
    def Fax(self, U_inf: float, rho=1.293) -> npt.ArrayLike:
        R = self.R
        dR = np.diff(self.mu)[0] * R
        dtheta = np.diff(self.theta)[0]
        A = self.mu_mesh * R * dR * dtheta

        return 0.5 * rho * U_inf**2 * self.Cax(doubledecomp=True) * A

    @doubledecompose
    def Ftan(self, U_inf: float, rho=1.293) -> npt.ArrayLike:
        R = self.R
        dR = np.diff(self.mu)[0] * R
        dtheta = np.diff(self.theta)[0]

        A = self.mu_mesh * R * dR * dtheta

        return 0.5 * rho * U_inf**2 * self.Ctan(doubledecomp=True) * A

    @decompose
    def thrust(self, U_inf: float, rho=1.293) -> npt.ArrayLike:
        R = self.R
        dR = np.diff(self.mu)[0] * R
        dtheta = np.diff(self.theta)[0]

        A = self.mu * R * dR * dtheta

        return 0.5 * rho * U_inf**2 * self.Ct(decomp=True) * A

    @decompose
    def power(self, U_inf: float, rho=1.293) -> npt.ArrayLike:
        R = self.R
        dR = np.diff(self.mu)[0] * R
        dtheta = np.diff(self.theta)[0]

        A = self.mu * R * dR * dtheta

        return 0.5 * rho * U_inf**3 * self.Cp(decomp=True) * A

    @decompose
    def torque(self, U_inf: float, rho=1.293) -> npt.ArrayLike:
        R = self.R
        rotor_speed = self.tsr * U_inf / R

        return self.power(U_inf, rho=rho, decomp=True) / rotor_speed

    @doubledecompose
    def REWS(self):
        return self.U


class BEM(_BEMSolverBase):
    def __init__(
        self,
        rotor: RotorDefinition,
        tiploss: Union[str, Tiploss.TiplossModel] = "PrandtlRootTip",
        Cta_method: Union[str, ThrustInduction.CtaModel] = "Unified",
        beta=0.1403,
        Nr=20,
        Ntheta=21,
        inner_maxiter=400,
        inner_relax=0.25,
        outer_maxiter=300,
        outer_relax=0.25,
    ):
        self.rotor = rotor

        self.Nr, self.Ntheta = Nr, Ntheta

        self.mu, self.theta, self.mu_mesh, self.theta_mesh = calc_gridpoints(Nr, Ntheta)

        self.tiploss_func = Tiploss.build_tiploss_model(tiploss, rotor)
        self.momentum_model = ThrustInduction.build_cta_model(Cta_method)

        self._solidity = self.rotor.solidity(self.mu)

        self.beta = beta

        self.inner_maxiter = inner_maxiter
        self.inner_relax = inner_relax
        self.outer_maxiter = outer_maxiter
        self.outer_relax = outer_relax

    def solve(
        self, pitch: float, tsr: float, yaw: float = 0.0, windfield=None
    ) -> BEMSolution:
        if callable(windfield):
            self.U, self.wdir = self._sample_windfield(windfield)
        elif windfield:
            self.U, self.wdir = windfield
        else:
            self.U, self.wdir = np.ones_like(self.mu_mesh), np.zeros_like(self.mu_mesh)

        self.sol = BEMSolution(
            self.mu,
            self.theta,
            self.mu_mesh,
            self.theta_mesh,
            pitch,
            tsr,
            yaw,
            self.U,
            self.wdir,
            self.rotor.R,
        )
        self.sol.inner_niter = 0

        self.sol.solidity = self._solidity
        self.sol.beta = self.beta

        a0 = 1 / 3 * np.ones(self.Nr)
        aprime0 = np.zeros(self.Nr)

        x0 = np.vstack([a0, aprime0])

        FP_sol = fixedpointiteration(
            self.residual, x0=x0, maxiter=self.outer_maxiter, relax=self.outer_relax
        )

        self.sol.converged = FP_sol.converged
        self.sol.outer_niter = FP_sol.niter
        self.sol.outer_relax = FP_sol.relax

        if FP_sol.converged:
            (
                self.sol._a,
                self.sol._aprime,
            ) = FP_sol.x

        self.sol.zero_nans()
        return self.sol

    def residual(self, x: npt.ArrayLike) -> npt.ArrayLike:
        an, aprime = x

        self.update_aerodynamics(an, aprime)

        tangential_integral = annulus_average(
            self.sol.theta_mesh,
            self.sol._W**2 * self.sol._Ctan,
        )

        e_aprime = (
            self.sol.solidity
            / (
                4
                * np.maximum(self.mu, 0.1) ** 2
                * self.sol.tsr
                * (1 - an)
                * np.cos(self.sol.yaw)
            )
            * tangential_integral
        ) - aprime

        # x0 = np.vstack([self.sol._a, self.sol._u4, self.sol._v4, self.sol._dp])

        # self.sol._a = an
        self.sol = self.momentum_model(self.sol)
        # mom_sol = self.momentum_model.solve(
        #     self.sol._Ctprime,
        #     self.sol.yaw,
        #     x0,
        #     maxiter=self.inner_maxiter,
        #     relax=self.inner_relax,
        # )

        e_an = self.sol._a - an
        # self.sol._a = mom_sol.an
        self.sol._aprime = aprime
        # self.sol._u4 = mom_sol.u4
        # self.sol._v4 = mom_sol.v4
        # self.sol._dp = mom_sol.dp
        # self.sol.inner_niter = np.maximum(mom_sol.niter, self.sol.inner_niter)
        # self.sol.inner_relax = self.inner_relax
        self.sol.sampled_Ctprimes.extend(self.sol._Ctprime)

        residual = np.vstack([e_an, e_aprime])

        return residual

    def update_aerodynamics(
        self, an: npt.ArrayLike, aprime: npt.ArrayLike
    ) -> npt.ArrayLike:
        sol = self.sol

        local_yaw = sol.wdir - sol.yaw
        sol._Vax = sol.U * (
            (1 - an[..., None])
            * np.cos(local_yaw * np.cos(sol.theta_mesh))
            * np.cos(local_yaw * np.sin(sol.theta_mesh))
        )
        sol._Vtan = (1 + aprime[..., None]) * sol.tsr * sol.mu_mesh - sol.U * (
            1 - an[..., None]
        ) * np.cos(local_yaw * np.sin(sol.theta_mesh)) * np.sin(
            local_yaw * np.cos(sol.theta_mesh)
        )
        sol._W = np.sqrt(sol._Vax**2 + sol._Vtan**2)

        # inflow angle
        sol._phi = np.arctan2(sol._Vax, sol._Vtan)
        sol._aoa = sol._phi - self.rotor.twist(sol.mu_mesh) - sol.pitch
        sol._aoa = np.clip(sol._aoa, -np.pi / 2, np.pi / 2)

        # Lift and drag coefficients
        sol._Cl, sol._Cd = self.rotor.clcd(sol.mu_mesh, sol._aoa)

        # axial and tangential force coefficients
        sol._Cax = sol._Cl * np.cos(sol._phi) + sol._Cd * np.sin(sol._phi)
        sol._Ctan = sol._Cl * np.sin(sol._phi) - sol._Cd * np.cos(sol._phi)

        # Tip-loss correction
        sol._tiploss = self.tiploss_func(sol.mu_mesh, sol._phi)

        sol._Ct = sol.solidity * annulus_average(
            self.sol.theta_mesh,
            self.sol._tiploss * sol._W**2 * sol._Cax,
        )

        sol._Ctprime = sol._Ct / ((1 - an) ** 2 * np.cos(sol.yaw) ** 2)
