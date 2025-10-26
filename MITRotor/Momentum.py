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
    def __call__(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
        tilt: float = 0.0,
    ) -> ArrayLike:
        ...

    @abstractmethod
    def compute_induction(self, Cx: ArrayLike, yaw: float = 0, tilt:float = 0) -> ArrayLike:
        ...

    @abstractmethod
    def compute_initial_wake_velocities(self, Ct: float, yaw: float = 0, tilt: float = 0.0) -> ArrayLike:
        ...


    
    def _func_rotor(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
        tilt: float = 0.0,
    ) -> ArrayLike:
        
        rotor_avg_axial_force = (
            geom.rotor_average(
                geom.annulus_average(
                    np.clip(aero_props.C_x_corr, 0, 1.69)
                    )
                    )
        )

        return self.compute_induction(rotor_avg_axial_force, yaw = yaw, tilt = tilt)



    def _func_annulus(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
        tilt: float = 0.0,
    ) -> ArrayLike:
        
        annulus_avg_axial_force = (
            
                geom.annulus_average(
                    np.clip(aero_props.C_x_corr, 0, 1.69)
                    )
                    )[:, None] * np.ones(geom.shape)
        

        return self.compute_induction(annulus_avg_axial_force, yaw = yaw, tilt = tilt)

    def _func_sector(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
        tilt: float = 0.0,
    ) -> ArrayLike:
        axial_force = np.clip(aero_props.C_x_corr, 0, 1.69)

        return self.compute_induction(axial_force, yaw = yaw, tilt = tilt)

    def __call__(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
        tilt: float = 0.0,
    ) -> ArrayLike:
        an = self._func(aero_props, pitch, tsr, yaw, rotor, geom, tilt = tilt)
        return np.clip(an, 0, 1)


class ConstantInduction(MomentumModel):
    def __init__(self, a = 1/3):
        self.a = a

    def compute_induction(self, Cx, yaw, tilt = 0) -> ArrayLike:
        if tilt != 0:
            raise ValueError("Tilt not supported by the ConstantInduction momentum model. Use UMM.")
        return self.a * np.ones_like(yaw)
    
    def compute_initial_wake_velocities(self, Ct: float, yaw: float, tilt: float = 0.0) -> ArrayLike:
        if tilt != 0:
            raise ValueError("Tilt not supported by the ConstantInduction momentum model. Use UMM.")
        u4 = 1 - 2 * self.a
        v4 = - (1/4) * Ct * np.sin(yaw)
        w4 = 0.0
        return u4, v4, w4


class ClassicalMomentum(MomentumModel):
    def __init__(self, averaging: Literal["sector", "annulus", "rotor"] = "rotor"):
        if averaging == "rotor":
            self._func = self._func_rotor
        elif averaging == "annulus":
            self._func = self._func_annulus
        elif averaging == "sector":
            self._func = self._func_sector
        else:
            raise ValueError(f"Averaging method {averaging} not found for ClassicalMomentum model.")
        self.averaging = averaging

    def compute_induction(self, Cx, yaw, tilt = 0):
        if tilt != 0:
            raise ValueError("Tilt not supported by the ClassicalMomentum momentum model. Use UMM.")
        return 0.5 * (1 - np.sqrt(1 - Cx))
    
    def compute_initial_wake_velocities(self, Ct: float, yaw: float, tilt: float = 0.0) -> ArrayLike:
        if tilt != 0:
            raise ValueError("Tilt not supported by the ClassicalMomentum momentum model. Use UMM.")
        u4 = np.sqrt(1 - Ct)
        v4 = - (1/4) * Ct * np.sin(yaw)
        w4 = 0.0
        return u4, v4, w4



class HeckMomentum(MomentumModel):
    """
    Heck Momentum model based on 2023 paper:
    https://doi.org/10.1017/jfm.2023.129

    Note that this version takes in CT and has a high thrust correction when calculating induction.
    """
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

    def compute_induction(self, Cx: ArrayLike, yaw: float, tilt: float = 0.0) -> ArrayLike:
        if tilt != 0:
            raise ValueError("Tilt not supported by the HeckMomentum model for BEM. Use UMM.")
        Ctc = 4 * self.ac * (1 - self.ac) / (1 + 0.25 * (1 - self.ac) ** 2 * np.sin(yaw) ** 2)
        slope = (16 * (1 - self.ac) ** 2 * np.sin(yaw) ** 2 - 128 * self.ac + 64) / (
            (1 - self.ac) ** 2 * np.sin(yaw) ** 2 + 4
        ) ** 2

        a = (2 * Cx - 4 + np.sqrt(-(Cx**2) * np.sin(yaw) ** 2 - 16 * Cx + 16)) / (
            -4 + np.sqrt(-(Cx**2) * np.sin(yaw) ** 2 - 16 * Cx + 16)
        )

        mask = Cx > Ctc
        if np.iterable(Cx):
            if np.any(mask):
                a[mask] = (Cx[mask] - Ctc) / slope + self.ac
        elif isinstance(Cx, (int, float)):
            if mask:
                a = (Cx - Ctc) / slope + self.ac
        else:
            raise ValueError(f"Unsupported type of Cx ({Cx}) - not iterable and not a float - so high thrust correction in Heck can't be applied.")

        return a
    
    def compute_initial_wake_velocities(self, Ct: float, yaw: float, tilt: float = 0.0) -> ArrayLike:
        if tilt != 0:
            raise ValueError("Tilt not supported by the HeckMomentum model for BEM. Use UMM.")
        a = self.compute_induction(Ct, yaw)
        u4 = 1 - Ct /(2  * (1 - a))
        v4 = - (1/4) * Ct * np.sin(yaw)
        w4 = 0.0
        return u4, v4, w4


class UnifiedMomentum(MomentumModel):
    """
    Unified Momentum Model based on 2024 paper:
    https://www.nature.com/articles/s41467-024-50756-5 

    Note that this version takes in CT and thus uses the thrust based unified momentum model.
    """
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

        self.model_Ct = UMM.ThrustBasedUnified(beta=beta)

    def compute_induction(self, *args, **kwargs) -> ArrayLike:
        sol = self.model_Ct(*args, **kwargs)
        return sol.an
    
    def compute_initial_wake_velocities(self, *args, **kwargs) -> ArrayLike:
        sol = self.model_Ct(*args, **kwargs)
        return sol.u4, sol.v4, sol.w4


class MadsenMomentum(MomentumModel):
    """
    Madsen Momentum model based on 2020 paper:
    https://wes.copernicus.org/articles/5/1/2020/
    """
    def __init__(self, 
                 averaging: Literal["sector", "annulus", "rotor"] = "rotor",
                 cosine_exponent: bool = False):
        if averaging == "rotor":
            self._func = self._func_rotor
        elif averaging == "annulus":
            self._func = self._func_annulus
        elif averaging == "sector":
            self._func = self._func_sector
        else:
            raise ValueError(f"Averaging method {averaging} not found for MadsenMomentum model.")
        self.averaging = averaging
        self.cosine_exponent = cosine_exponent


    def compute_induction(self, Cx: ArrayLike, yaw: float, tilt: float = 0.0) -> ArrayLike:
        if tilt != 0:
            raise ValueError("Tilt not supported by the Madsen momentum model. Use UMM.")
        if self.cosine_exponent:
            Ct = Cx / (np.cos(yaw)**2)
        else:
            Ct = Cx

        an = Ct**3 * 0.0883 + Ct**2 * 0.0586 + Ct * 0.2460
        return an

    def compute_initial_wake_velocities(self, Ct: float, yaw: float, tilt: float = 0.0) -> ArrayLike:
        if tilt != 0:
            raise ValueError("Tilt not supported by the Madsen momentum model. Use UMM.")
        u4 = np.sqrt(1 - Ct)
        v4 = - (1/4) * Ct * np.sin(yaw)
        w4 = 0.0
        return u4, v4, w4
