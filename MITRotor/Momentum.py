from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal
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
    def __init__(
            self, averaging: Literal["sector", "annulus", "rotor"] = "rotor"
    ):
        if averaging == "rotor":
            self._func = self._func_rotor
        elif averaging == "annulus":
            self._func = self._func_annulus
        elif averaging == "sector":
            self._func = self._func_sector
        else:
            raise ValueError(f"Averaging method {averaging} not found for ClassicalMomentum model.")
        self.averaging = averaging

    def Ct_a(self, Ct, yaw):
        return 0.5 * (1 - np.sqrt(1 - Ct))

    def __call__(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
    ) -> ArrayLike:
        an = self._func(aero_props, pitch, tsr, yaw, rotor, geom)
        return an
    
    def _func_rotor(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
    ) -> ArrayLike:

        # Compute the thrust coefficient
        Ct = aero_props.solidity * aero_props.W**2 * aero_props.Cax
        # Average the thrust coefficient over the rotor
        Ct_bar = geom.rotor_average(geom.annulus_average(Ct))
        # Compute the average axial induction factor
        a_bar = self.Ct_a(Ct_bar, yaw)
        # Compute the average tip loss factor
        f_bar = geom.rotor_average(geom.annulus_average(aero_props.F))
        # Compute the corrected average axial induction factor
        a_corr = np.ones_like(geom.mu_mesh) * (a_bar / f_bar)

        return a_corr
    
    def _func_annulus(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
    ) -> ArrayLike:
        
        # Compute the annulus averaged thrust coefficient
        Ct = geom.annulus_average(aero_props.solidity * aero_props.W**2 * aero_props.Cax)
        # Compute the annulus averaged axial induction factor and apply the tip loss correction
        a = self.Ct_a(Ct, yaw)[:, None] * np.ones(geom.shape) / aero_props.F

        return a
    
    def _func_sector(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
    ) -> ArrayLike:

        # Compute the thrust coefficient
        Ct = aero_props.solidity * aero_props.W**2 * aero_props.Cax
        # Compute the axial induction factor and apply the tip loss correction
        a = self.Ct_a(Ct.ravel(), yaw)
        return a.reshape(geom.shape) / aero_props.F


class HeckMomentum(MomentumModel):
    def __init__(
        self, averaging: Literal["sector", "annulus", "rotor"] = "rotor", ac: float = 1 / 3, v4_correction: float = 1.0
    ):
        self.v4_correction = v4_correction
        self.ac = ac
        if averaging == "rotor":
            self._func = self._func_rotor
        elif averaging == "annulus":
            self._func = self._func_annulus
        elif averaging == "sector":
            self._func = self._func_sector
        else:
            raise ValueError(f"Averaging method {averaging} not found for HeckMomentum model.")
        self.averaging = averaging

    def Ct_a(self, Ct: ArrayLike, yaw: float) -> ArrayLike:
        Ctc = 4 * self.ac * (1 - self.ac) / (1 + 0.25 * (1 - self.ac) ** 2 * np.sin(yaw) ** 2)
        slope = (16 * (1 - self.ac) ** 2 * np.sin(yaw) ** 2 - 128 * self.ac + 64) / (
            (1 - self.ac) ** 2 * np.sin(yaw) ** 2 + 4
        ) ** 2

        a_target = (2 * Ct - 4 + np.sqrt(-(Ct**2) * np.sin(yaw) ** 2 - 16 * Ct + 16)) / (
            -4 + np.sqrt(-(Ct**2) * np.sin(yaw) ** 2 - 16 * Ct + 16)
        )

        if np.iterable(Ct):
            mask = Ct > Ctc
            if np.any(mask):
                a_target[mask] = (Ct[mask] - Ctc) / slope + self.ac

        return a_target

    def __call__(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
    ) -> ArrayLike:
        an = self._func(aero_props, pitch, tsr, yaw, rotor, geom)
        return an

    def _func_rotor(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
    ) -> ArrayLike:

        # Compute the thrust coefficient
        Ct = aero_props.solidity * aero_props.W**2 * aero_props.Cax
        # Average the thrust coefficient over the rotor
        Ct_bar = geom.rotor_average(geom.annulus_average(Ct))
        # Compute the average axial induction factor
        a_bar = self.Ct_a(Ct_bar, yaw)
        # Compute the average tip loss factor
        f_bar = geom.rotor_average(geom.annulus_average(aero_props.F))
        # Compute the corrected average axial induction factor
        a_corr = np.ones_like(geom.mu_mesh) * (a_bar / f_bar)

        return a_corr

    def _func_annulus(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
    ) -> ArrayLike:
        
        # Compute the annulus averaged thrust coefficient
        Ct = geom.annulus_average(aero_props.solidity * aero_props.W**2 * aero_props.Cax)
        # Compute the annulus averaged axial induction factor and apply the tip loss correction
        a = self.Ct_a(Ct, yaw)[:, None] * np.ones(geom.shape) / aero_props.F

        return a

    def _func_sector(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
    ) -> ArrayLike:

        # Compute the thrust coefficient
        Ct = aero_props.solidity * aero_props.W**2 * aero_props.Cax
        # Compute the axial induction factor and apply the tip loss correction
        a = self.Ct_a(Ct.ravel(), yaw)
        return a.reshape(geom.shape) / aero_props.F


class UnifiedMomentum(MomentumModel):
    def __init__(self, averaging: Literal["sector", "annulus", "rotor"] = "rotor", beta=0.1403):
        self.beta = beta

        if averaging == "rotor":
            self._func = self._func_rotor
        elif averaging == "annulus":
            self._func = self._func_annulus
        elif averaging == "sector":
            self._func = self._func_sector
        else:
            raise ValueError(f"Averaging method {averaging} not found for UnifiedMomentum model.")
        self.averaging = averaging

        self.model_Ctprime = UMM.UnifiedMomentum(beta=beta)
        self.model_Ct = UMM.ThrustBasedUnified(beta=beta)

    def Ct_a(self, Ct: ArrayLike, yaw: float) -> ArrayLike:
        sol = self.model_Ct(Ct, yaw)
        return sol.an

    def _func_rotor(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
    ) -> ArrayLike:

        # Compute the thrust coefficient
        Ct = aero_props.solidity * aero_props.W**2 * aero_props.Cax
        # Average the thrust coefficient over the rotor
        Ct_bar = geom.rotor_average(geom.annulus_average(Ct))
        # Compute the average axial induction factor
        a_bar = self.Ct_a(Ct_bar, yaw)
        # Compute the average tip loss factor
        f_bar = geom.rotor_average(geom.annulus_average(aero_props.F))
        # Compute the corrected average axial induction factor
        a_corr = np.ones_like(geom.mu_mesh) * (a_bar / f_bar)

        return a_corr

    def _func_annulus(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
    ) -> ArrayLike:
        
        # Compute the annulus averaged thrust coefficient
        Ct = geom.annulus_average(aero_props.solidity * aero_props.W**2 * aero_props.Cax)
        # Compute the annulus averaged axial induction factor and apply the tip loss correction
        a = self.Ct_a(Ct, yaw)[:, None] * np.ones(geom.shape) / aero_props.F

        return a

    def _func_sector(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
    ) -> ArrayLike:
        
        # Compute the thrust coefficient
        Ct = aero_props.solidity * aero_props.W**2 * aero_props.Cax
        # Compute the axial induction factor
        a = self.Ct_a(Ct.ravel(), yaw)
        # Apply the tip loss correction and return the axial induction factor
        return a.reshape(geom.shape) / aero_props.F

    def __call__(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
    ) -> ArrayLike:
        an = self._func(aero_props, pitch, tsr, yaw, rotor, geom)
        return an


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
        Ct = aero_props.solidity * aero_props.W**2 * aero_props.Cax
        an = self.Ct_a(Ct, yaw, tiploss=aero_props.F)
        return an
