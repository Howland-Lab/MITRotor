from abc import ABC, abstractmethod
from typing import Union, TYPE_CHECKING
import numpy as np
import numpy.typing as npt

from .MomentumTheory import UnifiedMomentum, ThrustBasedUnified

if TYPE_CHECKING:
    from .BEM import BEMSolution


class CtaModel(ABC):
    @abstractmethod
    def __call__(self, bem_obj: "BEMSolution") -> "BEMSolution":
        pass


def build_cta_model(cta_model: Union[str, CtaModel]) -> CtaModel:
    if isinstance(cta_model, CtaModel):
        return cta_model

    elif isinstance(cta_model, str):
        if cta_model.lower() == "madsen":
            return Madsen()
        elif cta_model.lower() == "unified":
            return CTaUnifiedMomentum()
        elif cta_model.lower() == "heck":
            return CTaHeck()
        elif cta_model.lower() == "fixed":
            return FixedInduction()
        else:
            raise ValueError(f"Unknown model name {cta_model}")

    else:
        raise TypeError("cta_model must be a string or CtaModel instance")


class Madsen(CtaModel):
    def Ct_a(self, Ct, yaw):
        k3 = -0.6481 * yaw**3 + 2.1667 * yaw**2 - 2.0705 * yaw
        k2 = 0.8646 * yaw**3 - 2.6145 * yaw**2 + 2.1735 * yaw
        k1 = -0.1640 * yaw**3 + 0.4438 * yaw**2 - 0.5136 * yaw
        CT_lim = np.minimum(Ct, 0.9)

        Ka = k3 * np.abs(CT_lim) ** 3 + k2 * CT_lim**2 + k1 * np.abs(CT_lim) + 1.0

        a = Ct**3 * 0.0883 + Ct**2 * 0.0586 + Ct * 0.2460

        a_corrected = a * Ka
        return np.minimum(a_corrected, 2.0)

    def __call__(self, sol: "BEMSolution") -> "BEMSolution":
        sol._a = self.Ct_a(sol._Ct, sol.yaw)

        return sol


class CTaUnifiedMomentum(CtaModel):
    def __init__(self, beta=0.1403):
        self.beta = beta
        self.model_Ctprime = UnifiedMomentum(beta=beta)
        self.model_Ct = ThrustBasedUnified(beta=beta)

    def Ct_a(self, Ct: npt.ArrayLike, yaw: npt.ArrayLike) -> npt.ArrayLike:
        sol = self.model_Ct.solve(Ct, yaw)
        return sol.an

    def __call__(self, sol: "BEMSolution") -> "BEMSolution":
        x0 = np.vstack([sol._a, sol._u4, sol._v4, sol._dp])

        momentum_sol = self.model_Ctprime.solve(sol._Ctprime, sol.yaw, x0)
        sol._a, sol._u4, sol._v4, sol._dp = momentum_sol.solution

        sol.inner_niter = np.maximum(sol.inner_niter, momentum_sol.niter)

        return sol


class CTaHeck(CtaModel):
    def Ct_a(self, Ct: npt.ArrayLike, yaw: npt.ArrayLike) -> npt.ArrayLike:
        an = (2 * Ct - 4 + np.sqrt(-(Ct**2) * np.sin(yaw) ** 2 - 16 * Ct + 16)) / (
            -4 + np.sqrt(-(Ct**2) * np.sin(yaw) ** 2 - 16 * Ct + 16)
        )

        return an

    def __call__(self, sol: "BEMSolution") -> "BEMSolution":
        sol._a = self.Ct_a(sol._Ct, sol.yaw)

        return sol


class FixedInduction(CtaModel):
    def __init__(self, a=1 / 3):
        self.a = a

    def Ct_a(self, Ct: npt.ArrayLike, yaw: npt.ArrayLike) -> npt.ArrayLike:
        return self.a * np.ones_like(Ct)

    def __call__(self, sol: "BEMSolution") -> "BEMSolution":
        sol._a = 1 / 3 * np.ones_like(sol._a)
        return sol
