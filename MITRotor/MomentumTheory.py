from dataclasses import dataclass
from abc import ABCMeta, abstractmethod
from typing import Union, Tuple

import numpy as np
import numpy.typing as npt
from .Utilities import fixedpointiteration


@dataclass
class MomentumSolution:
    """Stores the results of the Unified Momentum model solution."""

    Ctprime: float
    yaw: float
    # beta: float
    an: Union[float, npt.ArrayLike]
    u4: Union[float, npt.ArrayLike]
    v4: Union[float, npt.ArrayLike]
    dp: Union[float, npt.ArrayLike]
    niter: int
    converged: bool

    @property
    def solution(self):
        """Returns the solution variables an, u4, v4, dp."""
        return self.an, self.u4, self.v4, self.dp

    @property
    def Ct(self):
        """Returns the thrust coefficient Ct."""
        return self.Ctprime * (1 - self.an) ** 2 * np.cos(self.yaw) ** 2

    @property
    def Cp(self):
        """Returns the power coefficient Cp."""
        return self.Ctprime * ((1 - self.an) * np.cos(self.yaw)) ** 3

    @property
    def x0(self):
        """Returns the near wake length, x0."""
        return (
            np.cos(self.yaw)
            / 2
            * np.sqrt(((1 - self.an) * np.cos(self.yaw)) / (1 + self.u4))
            / (2 / (1 + self.u4) * (self.beta * (1 - self.u4) / 2))
        )


class MomentumBase(metaclass=ABCMeta):
    @abstractmethod
    def solve(self, Ctprime: float, yaw: float, **kwargs) -> MomentumSolution:
        ...


class LimitedHeck(MomentumBase):
    """
    Solves the limiting case when v_4 << u_4. (Eq. 2.19, 2.20). Also takes Numpy
    array arguments.
    """

    def solve(self, Ctprime: float, yaw: float) -> MomentumSolution:
        """
        Args:
            Ctprime (float): Rotor thrust coefficient.
            yaw (float): Rotor yaw angle (radians).

        Returns:
            Tuple[float, float, float]: induction and outlet velocities.
        """

        a = Ctprime * np.cos(yaw) ** 2 / (4 + Ctprime * np.cos(yaw) ** 2)
        u4 = (4 - Ctprime * np.cos(yaw) ** 2) / (4 + Ctprime * np.cos(yaw) ** 2)
        v4 = (
            -(4 * Ctprime * np.sin(yaw) * np.cos(yaw) ** 2)
            / (4 + Ctprime * np.cos(yaw) ** 2) ** 2
        )
        dp = 0.0 * a
        return MomentumSolution(Ctprime, yaw, a, u4, v4, dp, 1, True)


class Heck(MomentumBase):
    """
    Solves the iterative momentum equation for an actuator disk model.
    """

    def initial_condition(self, Ctprime, yaw):
        sol = LimitedHeck().solve(Ctprime, yaw)
        return sol.an, sol.u4, sol.v4

    def solve(
        self, Ctprime: float, yaw: float, x0=None, eps=0.00001
    ) -> MomentumSolution:
        """
        Args:
            Ctprime (float): Rotor thrust coefficient.
            yaw (float): Rotor yaw angle (radians).

        Returns:
            Tuple[float, float, float]: induction and outlet velocities.
        """
        if x0 is None:
            x0 = self.initial_condition(Ctprime, yaw)

        relax = 0.9 if np.max(Ctprime) > 15 else 0.1

        sol = fixedpointiteration(
            self.residual,
            x0,
            args=(Ctprime, yaw),
            eps=eps,
            relax=relax,
        )

        if sol.converged:
            a, u4, v4 = sol.x
        else:
            a, u4, v4 = np.nan * np.zeros_like(x0)
        dp = np.zeros_like(a)
        return MomentumSolution(Ctprime, yaw, a, u4, v4, dp, sol.niter, sol.converged)

    def residual(self, x: np.ndarray, Ctprime: float, yaw: float) -> np.ndarray:
        """
        Residual function of yawed-actuator disk model in Eq. 2.15.

        Args:
            x (np.ndarray): (a, u4, v4)
            Ctprime (float): Rotor thrust coefficient.
            yaw (float): Rotor yaw angle (radians).
            Uamb (float): Ambient wind velocity. Defaults to 1.0.

        Returns:
            np.ndarray: residuals of induction and outlet velocities.
        """

        a, u4, v4 = x
        e_a = 1 - np.sqrt(1 - u4**2 - v4**2) / (np.sqrt(Ctprime) * np.cos(yaw)) - a

        e_u4 = (1 - 0.5 * Ctprime * (1 - a) * np.cos(yaw) ** 2) - u4

        e_v4 = -0.25 * Ctprime * (1 - a) ** 2 * np.sin(yaw) * np.cos(yaw) ** 2 - v4
        return np.array([e_a, e_u4, e_v4])


class UnifiedMomentum(MomentumBase):
    def __init__(self, beta=0.1403):
        self.beta = beta

    def initial_condition(self, Ctprime, yaw):
        """Returns the initial guess for the solution variables."""
        an = (Ctprime * np.cos(yaw) ** 2) / (4 + Ctprime * np.cos(yaw) ** 2)
        u4 = (4 - Ctprime * np.cos(yaw) ** 2) / (4 + Ctprime * np.cos(yaw) ** 2)
        v4 = (4 * Ctprime * np.sin(yaw) * np.cos(yaw) ** 2) / (
            (4 + Ctprime * np.cos(yaw) ** 2) ** 2
        )
        dp = np.zeros_like(Ctprime)

        return np.vstack([an, u4, v4, dp])

    def residual(self, x: np.ndarray, Ctprime: float, yaw: float) -> Tuple[float, ...]:
        """Returns the residual equations for the fixed point iteration."""
        an, u4, v4, dp = x

        e_an = (
            1
            - np.sqrt(
                -dp / (0.5 * Ctprime * np.cos(yaw) ** 2)
                + (1 - u4**2 - v4**2) / (Ctprime * np.cos(yaw) ** 2)
            )
            - an
        )

        e_u4 = (
            -(1 / 4) * Ctprime * (1 - an) * np.cos(yaw) ** 2
            + (1 / 2)
            + (1 / 2)
            * np.sqrt(
                (1 / 2 * Ctprime * (1 - an) * np.cos(yaw) ** 2 - 1) ** 2 - (4 * dp)
            )
        ) - u4

        e_v4 = -(1 / 4) * Ctprime * (1 - an) ** 2 * np.sin(yaw) * np.cos(yaw) ** 2 - v4

        e_dp = (
            -(1 / (2 * np.pi))
            * Ctprime
            * (1 - an) ** 2
            * np.cos(yaw) ** 2
            * np.arctan(
                (2 / (1 + u4))
                * (self.beta * np.abs((1 - u4)) / 2)
                / ((np.cos(yaw) / 2) * np.sqrt((1 - an) * np.cos(yaw) / (1 + u4)))
            )
        ) - dp

        return e_an, e_u4, e_v4, e_dp

    def solve(self, Ctprime, yaw, x0=None, maxiter=400, relax=0.25):
        """Solves the unified momentum model for the given thrust and yaw inputs."""
        if x0 is None:
            x0 = self.initial_condition(Ctprime, yaw)

        sol = fixedpointiteration(
            self.residual,
            x0,
            args=(Ctprime, yaw),
            maxiter=maxiter,
            relax=relax,
        )
        if np.any(np.isnan(_x) for _x in sol.x):
            x0 = self.initial_condition(Ctprime, yaw)
            sol = fixedpointiteration(
                self.residual,
                x0,
                args=(Ctprime, yaw),
                maxiter=maxiter,
                relax=relax,
            )

        if sol.converged:
            a, u4, v4, dp = sol.x
        else:
            a, u4, v4, dp = np.nan * np.zeros_like(x0)

        return MomentumSolution(Ctprime, yaw, a, u4, v4, dp, sol.niter, sol.converged)


class ThrustBasedUnified(UnifiedMomentum):
    def __init__(self, beta=0.1403):
        self.beta = beta

    def initial_condition(self, Ct, yaw):
        an = 0.5 * Ct
        u4 = 1 - Ct
        v4 = np.zeros_like(Ct)
        dp = np.zeros_like(Ct)
        Ctprime = np.sign(Ct)

        return np.vstack([an, u4, v4, dp, Ctprime])

    def residual(self, x, Ct, yaw):
        an, u4, v4, dp, Ctprime = x

        e_an, e_u4, e_v4, e_dp = super().residual([an, u4, v4, dp], Ctprime, yaw)

        e_Ctprime = Ct / ((1 - an) ** 2 * np.cos(yaw) ** 2) - Ctprime
        return np.array([e_an, e_u4, e_v4, e_dp, e_Ctprime])

    def solve(self, Ct, yaw, x0=None, relax=0.25, maxiter=5000, **kwargs):
        if x0 is None:
            x0 = self.initial_condition(Ct, yaw)

        sol = fixedpointiteration(
            self.residual,
            x0,
            args=(Ct, yaw),
            kwargs=kwargs,
            maxiter=maxiter,
            relax=relax,
        )
        if np.any(np.isnan(_x) for _x in sol.x):
            x0 = self.initial_condition(Ct, yaw)
            sol = fixedpointiteration(
                self.residual,
                x0,
                args=(Ct, yaw),
                kwargs=kwargs,
                maxiter=maxiter,
                relax=relax,
            )

        if sol.converged:
            a, u4, v4, dp, Ctprime = sol.x
        else:
            a, u4, v4, dp, Ctprime = np.nan * np.zeros_like(x0)

        return MomentumSolution(Ctprime, yaw, a, u4, v4, dp, sol.niter, sol.converged)
