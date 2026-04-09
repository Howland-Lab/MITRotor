from pathlib import Path
import numpy as np
import polars as pl
import itertools
from MITRotor.CachedLUT import CachedLUT
from MITRotor.FlorisInterface.FlorisInterface import default_bem_factory

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.optimize import brentq
from scipy.optimize import newton
from dataclasses import dataclass

@dataclass
class RatedInfo:
    windspeed: float
    power: float
    omega: float
    R: float
    A: float
    rho: float

def get_rated_info(bem_model):
    D = 242.23775645
    R = D / 2
    A = np.pi * R**2
    rho = 1.225
    P_rated = 15e6
    omega = 0.7916813487046278

    def rated_residual(V):
        tsr = omega * R / V
        Cp = bem_model(pitch=0.0, tsr=tsr, yaw=0.0, tilt=0.0).Cp()
        P = 0.5 * rho * A * V**3 * Cp
        return P - P_rated

    V_rated = brentq(rated_residual, 5.0, 20.0)

    return RatedInfo(
        V_rated,
        P_rated,
        omega,
        R,
        A,
        rho,
    )


def get_region2_k(bem_model, tsr_bounds=(4.0, 14.0)):
    """
    Compute:
        lambda_opt
        k  (torque constant for Q = k Omega^2)

    Parameters
    ----------
    bem_model : callable
        Function like:
            bem_model(pitch=?, tsr=?, yaw=?, tilt=?)
        returning object with .Cp
    rho : float
        Air density
    tsr_bounds : tuple
        Bounds for TSR search
    """
    def neg_cp(tsr): # objective function
        sol = bem_model(pitch=0.0, tsr=tsr, yaw=0.0, tilt=0.0)
        return -sol.Cp()

    result = minimize_scalar(
        neg_cp,
        bounds=tsr_bounds,
        method="bounded",
        options={"xatol": 1e-4},
    )

    if not result.success:
        raise RuntimeError("Cp optimization failed")

    lambda_opt = result.x
    Cp_opt = -result.fun

    return Cp_opt / lambda_opt**3, lambda_opt

def get_region2_setpoints(
    bem_model,
    wind_speeds,
    yaws,
    K,
    tsr_opt,
    tsr_bounds=(4.0, 14.0),
):
    print("Solving Region 2")
    def torque_residual(tsr, yaw):
        rad_yaw = np.deg2rad(yaw)
        sol = bem_model(pitch=0.0, tsr=tsr, yaw=rad_yaw, tilt=0.0)
        Cp = sol.Cp()
        return (Cp / tsr**3) - K

    Ny = yaws.size
    Nw = wind_speeds.size

    tsr_yaw = np.zeros(Ny)

    for j, yaw in enumerate(yaws):
        print(f"Yaw: {yaw}")
        try:
            tsr_eq = brentq(
                lambda x: torque_residual(x, yaw),
                tsr_bounds[0],
                tsr_bounds[1],
                xtol=1e-4,
            )
        except ValueError:
            raise RuntimeError(f"K equilibrium not found for {yaw}")

        tsr_yaw[j] = tsr_eq

    # Broadcast across wind speeds
    tsrs = np.tile(tsr_yaw, (Nw, 1))       # (Nw, Ny)
    pitches = np.zeros((Nw, Ny))           # Region 2 pitch = 0

    return tsrs, pitches

def get_region3_setpoints(
        bem_model,
        windspeeds,
        yaws,
        power,
        omega,
        R,
        A,
        rho,
        pitch_bounds = (0.0, 20.0),
    ):
    print("Solving Region 3")
    def Cp_residual(pitch, tsr, yaw, Cp_rated):
        rad_yaw = np.deg2rad(yaw)
        rad_pitch = np.deg2rad(pitch)
        sol = bem_model(pitch=rad_pitch, tsr=tsr, yaw=rad_yaw, tilt=0.0)
        Cp = sol.Cp()
        return Cp - Cp_rated
    
    tsrs = (omega * R) / windspeeds
    Cp_rated = power / (0.5 * rho * A * windspeeds**3)

    pitches = np.zeros((windspeeds.size, yaws.size), dtype=float)
    for i, speed in enumerate(windspeeds):
        print(f"Wind Speed: {speed}")
        for j, yaw in enumerate(yaws):
            print(f"Yaw: {yaw}")
            try:
                pitch_eq = brentq(
                lambda x: Cp_residual(x, tsrs[i], yaw, Cp_rated[i]),
                pitch_bounds[0],
                pitch_bounds[1],
                xtol=1e-5,
            )
            except ValueError:
                raise RuntimeError(f"Cp equilibrium not found for {speed} and {yaw}")
            pitches[i, j] = pitch_eq

    tsrs_2d = np.tile(tsrs[:, None], (1, yaws.size))
    return tsrs_2d, pitches

CACHE_FN_CT = Path(__file__).parent / "kOmegaSqaredControl.csv"
class KOmegaSquaredLUT(CachedLUT):
    def __init__(self, cache_fn: Path = CACHE_FN_CT, regenerate=False, s=0.025,
                LUT_windspeeds=None, LUT_yaws=None, bem_model = None,
    ):
        self._bem_model = bem_model if bem_model is not None else default_bem_factory()
        # set reasonable CT and eff_yaw range
        self._windspeeds = LUT_windspeeds if LUT_windspeeds is not None else np.linspace(6, 20, 10)
        self._eff_yaws = LUT_yaws if LUT_yaws is not None else np.linspace(0.0, 30.0, 10)
        # get k value
        self._K, self._tsr_opt = get_region2_k(bem_model = self._bem_model, tsr_bounds=(4.0, 14.0))
        print(f"TSR OPT: {self._tsr_opt}")
        # create look-up-table
        super().__init__(
            "eff_yaw",
            "windspeed",
            ["pitch", "tsr"],
            cache_fn,
            regenerate=regenerate,
            s=s,
        )

    def generate_table(self) -> pl.DataFrame:
        tsrs, pitches = self.compute_setpoints(
            self._bem_model,
            self._windspeeds,
            self._eff_yaws,
        )

        winds, yaws = self._windspeeds, self._eff_yaws
        Nw, Ny = winds.size, yaws.size

        # Build full grid
        wind_grid = np.repeat(winds, Ny)
        yaw_grid = np.tile(yaws, Nw)
        tsr_flat = tsrs.reshape(-1)
        pitch_flat = pitches.reshape(-1)

        df = pl.DataFrame({
            "windspeed": wind_grid,
            "eff_yaw": yaw_grid,
            "tsr": tsr_flat,
            "pitch": pitch_flat,
        })
        return df.unique(["windspeed", "eff_yaw"])

    def compute_setpoints(self, bem_model, windspeeds, yaws):

        rated_info = get_rated_info(bem_model)
        region2_idx = windspeeds <= rated_info.windspeed
        region3_idx = ~region2_idx


        V_rated = rated_info.windspeed
        tsr_rated = self._tsr_opt

        Cp_model = bem_model(
            pitch=0.0,
            tsr=tsr_rated,
            yaw=0.0,
            tilt=0.0,
        ).Cp()

        Cp_required = rated_info.power / (0.5 * rated_info.rho * rated_info.A * V_rated**3)

        print("Cp_model:", Cp_model)
        print("Cp_required:", Cp_required)

        Nw, Ny = windspeeds.size, yaws.size
        tsrs = np.zeros((Nw, Ny))
        pitches = np.zeros((Nw, Ny))

        # ---- Region 2 ----
        tsr_r2, pitch_r2 = get_region2_setpoints(
            bem_model,
            windspeeds[region2_idx],
            yaws,
            self._K,
            self._tsr_opt,
        )
        tsrs[region2_idx, :] = tsr_r2
        pitches[region2_idx, :] = pitch_r2

        # ---- Region 3 ----
        tsr_r3, pitch_r3 = get_region3_setpoints(
            bem_model,
            windspeeds[region3_idx],
            yaws,
            rated_info.power,
            rated_info.omega,
            rated_info.R,
            rated_info.A,
            rated_info.rho,
        )
        tsrs[region3_idx, :] = tsr_r3
        pitches[region3_idx, :] = pitch_r3

        return tsrs, pitches
    
KOmegaSquaredLUT()
