from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import ArrayLike

from UnifiedMomentumModel import Momentum as UMM

if TYPE_CHECKING:
    from .Geometry import BEMGeometry
    from .RotorDefinition import RotorDefinition
    from .Aerodynamics import AerodynamicProperties

__all__ = [
    "MomentumModel",
    "ConstantInduction",
    "ClassicalMomentum",
    "HeckMomentum",
    "UnifiedMomentum",
    "MadsenMomentum",
]


class MomentumModel(ABC):
    @abstractmethod
    def Ct_a(self, Ct: ArrayLike, yaw: float) -> ArrayLike:
        ...

    @abstractmethod
    def __call__(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
    ) -> ArrayLike:
        ...


class ConstantInduction(MomentumModel):
    def __init__(self, a=1 / 3):
        self.a = a

    def Ct_a(self, Ct: ArrayLike, yaw: float) -> ArrayLike:
        return self.a * np.ones_like(Ct)

    def __call__(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
    ) -> ArrayLike:
        return self.a * np.ones_like(aero_props.an)


class ClassicalMomentum(MomentumModel):
    def Ct_a(self, Ct, yaw, tiploss=1.0):
        return 0.5 * (1 + np.sqrt(1 - Ct / tiploss))

    def __call__(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
    ) -> ArrayLike:
        Ct = aero_props.solidity * geom.annulus_average(aero_props.W**2 * aero_props.Cax)
        a = self.Ct_a(Ct, yaw, tiploss=aero_props.F)

        return a


class HeckMomentum(MomentumModel):
    def __init__(self, v4_correction: float = 1.0):
        self.v4_correction = v4_correction

    def Ct_a(self, Ct: ArrayLike, yaw: float) -> ArrayLike:
        raise NotImplementedError

    def __call__(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
    ) -> ArrayLike:
        ac = 1 / 3

        Ct = aero_props.solidity * geom.annulus_average(aero_props.W**2 * aero_props.Cax)

        Ct_rotor = geom.rotor_average(Ct)

        Ctc = 4 * ac * (1 - ac) / (1 + 0.25 * (1 - ac) ** 2 * np.sin(yaw) ** 2)
        slope = (16 * (1 - ac) ** 2 * np.sin(yaw) ** 2 - 128 * ac + 64) / ((1 - ac) ** 2 * np.sin(yaw) ** 2 + 4) ** 2

        if Ct_rotor > Ctc:
            a_target = (Ct_rotor - Ctc) / slope + ac
        else:
            a_target = (2 * Ct_rotor - 4 + np.sqrt(-(Ct_rotor**2) * np.sin(yaw) ** 2 - 16 * Ct_rotor + 16)) / (
                -4 + np.sqrt(-(Ct_rotor**2) * np.sin(yaw) ** 2 - 16 * Ct_rotor + 16)
            )

        a_new = aero_props.F
        a_rotor = geom.rotor_average(a_new)
        a_new *= a_target / a_rotor

        return a_new


class UnifiedMomentum(MomentumModel):
    def __init__(self, beta=0.1403):
        self.beta = beta
        self.model_Ctprime = UMM.UnifiedMomentum(beta=beta)
        self.model_Ct = UMM.ThrustBasedUnified(beta=beta)

    def Ct_a(self, Ct: ArrayLike, yaw: float) -> ArrayLike:
        sol = self.model_Ct(Ct, yaw)
        return sol.an

    def __call__(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
    ) -> ArrayLike:
        Ct = aero_props.solidity * geom.annulus_average(aero_props.W**2 * aero_props.Cax)
        return self.Ct_a(Ct, yaw)
        # Ctprime = Ct / ((1 - aero_props.an) ** 2 * np.cos(yaw) ** 2)
        # momentum_sol = self.model_Ctprime(Ctprime / np.maximum(aero_props.F, 0.01), yaw)

        # return momentum_sol.an


class MadsenMomentum(MomentumModel):
    def __init__(self, cosine_exponent=None):
        self.cosine_exponent = cosine_exponent

    def Ct_a(self, Ct: ArrayLike, yaw: float, tiploss=1.0) -> ArrayLike:
        Ct_tiploss = np.clip(Ct / tiploss, 0.0, 2.0)
        if self.cosine_exponent:
            Ct_tiploss /= np.cos(yaw) ** self.cosine_exponent
        an = Ct_tiploss**3 * 0.0883 + Ct_tiploss**2 * 0.0586 + Ct_tiploss * 0.2460
        return an

    def __call__(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
    ) -> ArrayLike:
        Ct = aero_props.solidity * geom.annulus_average(aero_props.W**2 * aero_props.Cax)
        an = self.Ct_a(Ct, yaw, tiploss=aero_props.F)
        return an


# def calc_an(
#     aero_props: "AerodynamicProperties", pitch: float, tsr: float, yaw: float, rotor: "RotorDefinition", geom: "BEMGeometry"
# ) -> ArrayLike:
#     ac = 1 / 3

#     Ct = aero_props.solidity * geom.annulus_average(aero_props.W**2 * aero_props.Cax)

#     Ct_rotor = geom.rotor_average(Ct)

#     Ctc = 4 * ac * (1 - ac) / (1 + 0.25 * (1 - ac) ** 2 * np.sin(yaw) ** 2)
#     slope = (16 * (1 - ac) ** 2 * np.sin(yaw) ** 2 - 128 * ac + 64) / ((1 - ac) ** 2 * np.sin(yaw) ** 2 + 4) ** 2

#     if Ct_rotor > Ctc:
#         a_target = (Ct_rotor - Ctc) / slope + ac
#     else:
#         a_target = (2 * Ct_rotor - 4 + np.sqrt(-(Ct_rotor**2) * np.sin(yaw) ** 2 - 16 * Ct_rotor + 16)) / (
#             -4 + np.sqrt(-(Ct_rotor**2) * np.sin(yaw) ** 2 - 16 * Ct_rotor + 16)
#         )

#     a_new = aero_props.F
#     a_rotor = geom.rotor_average(a_new)
#     a_new *= a_target / a_rotor

#     return a_new


# class Madsen(MomentumModel):
#     def Ct_a(self, Ct, yaw, tiploss=1):
#         yaw = np.abs(yaw)
#         k3 = -0.6481 * yaw**3 + 2.1667 * yaw**2 - 2.0705 * yaw
#         k2 = 0.8646 * yaw**3 - 2.6145 * yaw**2 + 2.1735 * yaw
#         k1 = -0.1640 * yaw**3 + 0.4438 * yaw**2 - 0.5136 * yaw
#         CT_lim = np.clip(Ct, 0.0, 0.9)

#         Ka = k3 * np.abs(CT_lim) ** 3 + k2 * CT_lim**2 + k1 * np.abs(CT_lim) + 1.0

#         Ct_tiploss = np.clip(Ct / tiploss, 0.0, 2.0)
#         a = Ct_tiploss**3 * 0.0883 + Ct_tiploss**2 * 0.0586 + Ct_tiploss * 0.2460

#         a_corrected = a * Ka
#         return np.clip(a_corrected, 0.0, 2.0)

#     def __call__(
#         self,
#         aero_props: "AerodynamicProperties",
#         pitch: float,
#         tsr: float,
#         yaw: float,
#         rotor: "RotorDefinition",
#         geom: "BEMGeometry",
#     ) -> ArrayLike:
#         Ct = aero_props.solidity * geom.annulus_average(aero_props.W**2 * aero_props.Cax)
#         a = self.Ct_a(Ct, yaw, tiploss=aero_props.F)
#         a[geom.mu < 0.05] = 0.0
#         a[geom.mu > 0.99] = 0.0
#         # sol._Ctan[geom.mu > 0.99] = 0.0

#         return a


# class RotorAveragedHeck(MomentumModel):
#     def Ct_a(self, Ct: ArrayLike, yaw: float) -> ArrayLike:
#         raise NotImplementedError

#     def __call__(
#         self,
#         aero_props: "AerodynamicProperties",
#         pitch: float,
#         tsr: float,
#         yaw: float,
#         rotor: "RotorDefinition",
#         geom: "BEMGeometry",
#     ) -> ArrayLike:
#         ac = 1 / 3

#         Ct = aero_props.solidity * geom.annulus_average(aero_props.W**2 * aero_props.Cax)

#         Ct_rotor = geom.rotor_average(Ct)

#         Ctc = 4 * ac * (1 - ac) / (1 + 0.25 * (1 - ac) ** 2 * np.sin(yaw) ** 2)
#         slope = (16 * (1 - ac) ** 2 * np.sin(yaw) ** 2 - 128 * ac + 64) / ((1 - ac) ** 2 * np.sin(yaw) ** 2 + 4) ** 2

#         if Ct_rotor > Ctc:
#             a_target = (Ct_rotor - Ctc) / slope + ac
#         else:
#             a_target = (2 * Ct_rotor - 4 + np.sqrt(-(Ct_rotor**2) * np.sin(yaw) ** 2 - 16 * Ct_rotor + 16)) / (
#                 -4 + np.sqrt(-(Ct_rotor**2) * np.sin(yaw) ** 2 - 16 * Ct_rotor + 16)
#             )

#         a_new = aero_props.F
#         a_rotor = geom.rotor_average(a_new)
#         a_new *= a_target / a_rotor

#         return a_new


# class RotorAveragedUnified(MomentumModel):
#     def __init__(self, beta=0.1403):
#         self.beta = beta
#         self.model_Ctprime = UMM.UnifiedMomentum(beta=beta)
#         self.model_Ct = UMM.ThrustBasedUnified(beta=beta)

#     def Ct_a(self, Ct: ArrayLike, yaw: float) -> ArrayLike:
#         sol = self.model_Ct(Ct, yaw)
#         return sol.an

#     def __call__(
#         self,
#         aero_props: "AerodynamicProperties",
#         pitch: float,
#         tsr: float,
#         yaw: float,
#         rotor: "RotorDefinition",
#         geom: "BEMGeometry",
#     ) -> ArrayLike:
#         Ct = aero_props.solidity * geom.annulus_average(aero_props.W**2 * aero_props.Cax)

#         Ct_rotor = geom.rotor_average(Ct)
#         sol = self.model_Ct(Ct_rotor, yaw)
#         a_target = sol.an

#         a_new = aero_props.F
#         a_rotor = geom.rotor_average(a_new)
#         a_new *= a_target / a_rotor

#         return a_new


# class Lu(MomentumModel):
#     def __init__(self, v4_correction=1.5):
#         self.model_Ctprime = UMM.Heck(v4_correction=v4_correction)

#     def Ct_a(self, Ct: ArrayLike, yaw: float) -> ArrayLike:
#         ...

#     def __call__(
#         self,
#         aero_props: "AerodynamicProperties",
#         pitch: float,
#         tsr: float,
#         yaw: float,
#         rotor: "RotorDefinition",
#         geom: "BEMGeometry",
#     ) -> ArrayLike:
#         Ctprime = aero_props.solidity * geom.annulus_average(1 / np.sin(aero_props.phi) ** 2 * aero_props.Cax)
#         sol = self.model_Ctprime(np.minimum(Ctprime / aero_props.F, 4.0), yaw)
#         return sol.an


# class CTaRotorAveragedUnifiedMomentum:
#     def __init__(self, beta=0.1403):
#         self.beta = beta
#         self.model_Ctprime = UnifiedMomentum(beta=beta)
#         self.model_Ct = ThrustBasedUnified(beta=beta)

#     def Ct_a(self, Ct: ArrayLike, yaw: float) -> ArrayLike:
#         sol = self.model_Ct(Ct, yaw)
#         # print(f"if Ct={Ct}, a={sol.an} ({sol.converged})")
#         return sol.an

#     def __call__(self, sol: "BEMSolution") -> "BEMSolution":
#         x0 = np.vstack([sol._a, sol._u4, sol._v4, sol._dp])

#         Ct_rotor = sol.Ct()
#         # Ct_rotor = rotor_average(sol.mu, sol._Ct)
#         # print(f"{Ct_rotor=}")
#         a_target = self.Ct_a(Ct_rotor, sol.yaw)

#         # This is without tiploss (constant induction)
#         # a_new = np.ones_like(sol._tiploss)
#         # a_rotor = 1

#         # This is with tiploss
#         a_new = sol._tiploss
#         a_rotor = sol.tiploss()

#         a_new *= a_target / a_rotor

#         sol._a = a_new
#         # print(f"{sol.Ct()=}")
#         # what about u4, v4 and dp?
#         # print(sol.a())
#         # breakpoint()
#         return sol


# TODO
# class Mike:
#     def __call__(self, bem_obj):
#         Ct = bem_obj._W**2 * bem_obj.solidity * bem_obj._Cax

#         Ct_rotor = aggregate(bem_obj.mu, bem_obj.theta_mesh, Ct, agg="rotor")

#         a_target = (
#             2 * Ct_rotor
#             - 4
#             + np.sqrt(-(Ct_rotor**2) * np.sin(bem_obj.yaw) ** 2 - 16 * Ct_rotor + 16)
#         ) / (
#             -4
#             + np.sqrt(-(Ct_rotor**2) * np.sin(bem_obj.yaw) ** 2 - 16 * Ct_rotor + 16)
#         )

#         a_new = bem_obj._tiploss
#         a_rotor = aggregate(bem_obj.mu, bem_obj.theta_mesh, a_new, agg="rotor")
#         a_new *= a_target / a_rotor

#         return a_new


# class MikeCorrected:
#     def __call__(self, bem_obj, ac=1 / 3):
#         Ct = bem_obj._W**2 * bem_obj.solidity * bem_obj._Cax

#         Ct_rotor = aggregate(bem_obj.mu, bem_obj.theta_mesh, Ct, agg="rotor")

#         Ctc = 4 * ac * (1 - ac) / (1 + 0.25 * (1 - ac) ** 2 * np.sin(bem_obj.yaw) ** 2)
#         slope = (16 * (1 - ac) ** 2 * np.sin(bem_obj.yaw) ** 2 - 128 * ac + 64) / (
#             (1 - ac) ** 2 * np.sin(bem_obj.yaw) ** 2 + 4
#         ) ** 2

#         if Ct_rotor > Ctc:
#             a_target = (Ct_rotor - Ctc) / slope + ac
#         else:
#             a_target = (
#                 2 * Ct_rotor
#                 - 4
#                 + np.sqrt(
#                     -(Ct_rotor**2) * np.sin(bem_obj.yaw) ** 2 - 16 * Ct_rotor + 16
#                 )
#             ) / (
#                 -4
#                 + np.sqrt(
#                     -(Ct_rotor**2) * np.sin(bem_obj.yaw) ** 2 - 16 * Ct_rotor + 16
#                 )
#             )

#         a_new = bem_obj._tiploss
#         a_rotor = aggregate(bem_obj.mu, bem_obj.theta_mesh, a_new, agg="rotor")
#         a_new *= a_target / a_rotor

#         return a_new
