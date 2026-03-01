from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal
import numpy as np
import polars as pl
import itertools
from numpy.typing import ArrayLike
from MITRotor.CachedLUT import CachedLUT
from pathlib import Path
from UnifiedMomentumModel import Momentum as UMM
from UnifiedMomentumModel.Utilities.Geometry import calc_eff_yaw, eff_yaw_inv_rotation

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
                    np.clip(aero_props.C_x_corr, -10, 10)
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
        axial_force = np.clip(aero_props.C_x_corr, -10, 10)

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
        self._func = self._func_rotor

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
        Cx = np.asarray(Cx)
        sqrt_term = np.sqrt(np.where(Cx < 1, 1 - Cx, np.nan))
        return 0.5 * (1 - sqrt_term)
    
    def compute_initial_wake_velocities(self, Ct: float, yaw: float, tilt: float = 0.0) -> ArrayLike:
        if tilt != 0:
            raise ValueError("Tilt not supported by the ClassicalMomentum momentum model. Use UMM.")
        u4 = np.sqrt(1 - Ct)
        v4 = - (1/4) * Ct * np.sin(yaw)
        w4 = 0.0
        return u4, v4, w4



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
        u4 = np.sqrt(np.maximum(1 - Ct, 0))
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

        sqrt_term = np.sqrt(np.maximum(-(Cx**2) * np.sin(yaw) ** 2 - 16 * Cx + 16, 0))
        a = (2 * Cx - 4 + sqrt_term) / (-4 + sqrt_term + 1e-16)

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
    def __init__(self, averaging: Literal["sector", "annulus", "rotor"] = "rotor", beta=0.1403, model_Ct=None):
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

        self.model_Ct = (
            model_Ct if model_Ct is not None
            else UMM.ThrustBasedUnified(beta=beta)
        )

    def compute_induction(self, Cx: ArrayLike, yaw: float = 0.0, tilt: float = 0.0) -> ArrayLike:
        sol = self.model_Ct(Cx, yaw = yaw, tilt = tilt)
        return sol.an
    
    def compute_initial_wake_velocities(self, Cx: ArrayLike, yaw: float = 0.0, tilt: float = 0.0) -> ArrayLike:
        sol = self.model_Ct(Cx, yaw = yaw, tilt = tilt)
        return sol.u4, sol.v4, sol.w4


# Look-up table for unified model
def func_Ct(x, beta, cached) -> dict:
    Ct, eff_yaw = x
    model_Ct = UMM.ThrustBasedUnified(beta = beta, cached=cached)
    sol = model_Ct(Ct, np.deg2rad(np.round(eff_yaw, 2)))
    return dict(
        Ct=Ct,
        eff_yaw=np.round(eff_yaw, 2),
        Ctprime=sol.Ctprime,
        Cp=sol.Cp,
        an=sol.an,
        u4=sol.u4,
        v4=sol.v4,
        w4=sol.w4, # will be 0 as no tilt in yaw-only frame
        x0=sol.x0,
        dp=sol.dp,
        dp_NL=sol.dp_NL,
    )

CACHE_FN_CT = Path(__file__).parent / "unified_momentum_model_Ct_table.csv"
class ThrustBasedUnifiedLUT(CachedLUT, UMM.MomentumBase):
    """
    Note: This is NOT a BEM model!! This is the look-up-table version of 
        ThrustBasedUnified as implemented in the UnifiedMomentumModel library!!

        This is used as the momentum theory (actuator disk theory) part of BEM in the 
        UnifiedMomentumLUT MomentumModel below!!!

    - Args:
        - cache_fn (Path): path to cache file
        - regenerate (Bool): True if cache should regenerate each call, False otherwise
        - s (float): smoothing parameter used in interpolation in CachedLUT.py
        - LUT_Cts (ArrayLike): list of CT values to cache results for
        - LUT_yaws (ArrayLike): list of effective yaw values (in degrees) to cache results for -->
            as described in UMM documentation, yaw and tilt can be combined into an "effective yaw"
            the cache uses effective yaw, so these values should cover the range of effective yaws you
            want. You can use UnifiedMomentumModel.Utilities.Geometry.calc_eff_yaw to find
        - beta (float, optional) - used in UMM
        - cached (boolean, optional) - cache nonlinear pressure in UMM
    """
    def __init__(self, cache_fn: Path = CACHE_FN_CT, regenerate=False, s=0.025,
                LUT_Cts=None, LUT_yaws=None, beta=0.1403, cached=True,
    ):
        self._beta = beta
        self._nlp_cached = cached
        # set reasonable CT and eff_yaw range
        self._Cts = LUT_Cts if LUT_Cts is not None else np.linspace(-1, 1.5, 100)
        self._eff_yaws = LUT_yaws if LUT_yaws is not None else np.arange(-50.0, 50.1, 2.0)
        # create look-up-table
        super().__init__(
            "eff_yaw",
            "Ct",
            ["Ctprime", "Cp", "an", "u4", "v4", "w4", "x0", "dp", "dp_NL"],
            cache_fn,
            regenerate=regenerate,
            s=s,
        )

    def generate_table(self) -> pl.DataFrame:
        params = list(itertools.product(self._Cts, self._eff_yaws))
        # Run unified model and variations
        results = []
        for param in params:  # this can use the vectorization once branches are merged!
            results.append(func_Ct(param, beta = self._beta, cached = self._nlp_cached))
        return pl.from_dicts(results).unique(["Ct", "eff_yaw"])

    def __call__(self, Ct: ArrayLike, yaw: ArrayLike, tilt: ArrayLike = 0) -> UMM.MomentumSolution:
        """
        - Args:
            - Ct (float): thrust coefficent
            - yaw (float): degrees yaw (in radians)
            - tilt (float): degrees tilt (in radians)
        """
        # calculate effective yaw
        eff_yaw = calc_eff_yaw(yaw, tilt)
        eff_yaw_deg = np.rad2deg(eff_yaw)
        # rotate back into yaw-tilt frame -> has to be done here to keep LUT 2D
        # see UnifiedMomentum library's Geometry file for more information
        u4 = self.interpolators["u4"](Ct, eff_yaw_deg, grid=False)
        v4 = self.interpolators["v4"](Ct, eff_yaw_deg, grid=False)
        w4 = self.interpolators["w4"](Ct, eff_yaw_deg, grid=False)
        u4, v4, w4 = eff_yaw_inv_rotation(u4, v4, w4, eff_yaw, yaw, tilt)
        # create Momentum solution
        return UMM.MomentumSolution(
            self.interpolators["Ctprime"](Ct, eff_yaw_deg, grid=False),
            yaw,
            self.interpolators["an"](Ct, eff_yaw_deg, grid=False),
            u4,
            v4,
            self.interpolators["x0"](Ct, eff_yaw_deg, grid=False),
            self.interpolators["dp"](Ct, eff_yaw_deg, grid=False),
            tilt = tilt,
            w4 = w4,
        )


class UnifiedMomentumLUT(UnifiedMomentum):
    """
    Note: This is a BEM model!! This is the look-up-table version of 
        UnifiedMomentum. It uses ThrustBasedUnifiedLUT (see above!) rather
        than ThrustBasedUnified from the UnifiedMomentumModel library

    Note: **kwargs can be used to pass arguments to ThrustBasedUnifiedLUT constructor.
    """
    def __init__(self, averaging = "rotor", **kwargs):
        super().__init__(
            averaging = averaging,
            model_Ct = ThrustBasedUnifiedLUT(**kwargs),
        )