import os
import numpy as np
import polars as pl
from attrs import define, field
from typing import Optional, Callable, Union
from scipy.interpolate import interp1d, RegularGridInterpolator
# FLORIS Imports
from floris.type_dec import floris_float_type, NDArrayFloat
from floris.core.turbine.operation_models import BaseOperationModel
from floris.core.rotor_velocity import average_velocity
from floris.utilities import cosd
# MITRotor / UMM Imports
from MITRotor.ReferenceTurbines import IEA15MW
from MITRotor.Momentum import UnifiedMomentum
from MITRotor.Geometry import BEMGeometry
from MITRotor.TipLoss import NoTipLoss
from MITRotor.BEMSolver import BEM
from MITRotor.FlorisInterface.ROSCOInterface import query_controls
from UnifiedMomentumModel.Utilities.Geometry import calc_eff_yaw

InterpLike = Union[interp1d, RegularGridInterpolator, Callable]

# default rotor if none provided by user (IEA 15MW)
def default_bem_factory():
    return BEM(
        rotor=IEA15MW(),
        momentum_model=UnifiedMomentum(averaging="rotor"),
        geometry=BEMGeometry(Nr=10, Ntheta=20),
        tiploss_model=NoTipLoss()
    )
# default 1D pitch vs windspeed interpolater if none provided by user
# for IEA 15MW from figure 2 (https://docs.nrel.gov/docs/fy22osti/82134.pdf)
def default_pitch_interp():
    module_dir = os.path.dirname(__file__)
    pitch_file = os.path.join(module_dir, "IEA_15mw_rotor.csv")
    df = pl.read_csv(pitch_file)
    wind_table = df["Wind [m/s]"].to_numpy()
    pitch_table = np.deg2rad(df["Pitch [deg]"])
    # TODO: should fill_value be extrapolate?
    return interp1d(wind_table, pitch_table, kind="linear", fill_value="extrapolate", bounds_error=False)

# default 1D tsr vs windspeed interpolater if none provided by user
# for IEA 15MW from figure 2 (https://docs.nrel.gov/docs/fy22osti/82134.pdf)
def default_tsr_interp():
    module_dir = os.path.dirname(__file__)
    tsr_file = os.path.join(module_dir, "IEA_15mw_rotor.csv")
    df = pl.read_csv(tsr_file)
    wind_table = df["Wind [m/s]"].to_numpy()
    tip_speed_table = df["Tip Speed [m/s]"].to_numpy()
    tsr_table = tip_speed_table / wind_table
    # TODO: should fill_value be extrapolate?
    return interp1d(wind_table, tsr_table, kind="linear", fill_value="extrapolate", bounds_error=False)

@define
class MITRotorTurbine(BaseOperationModel):
    """
    Turbine operation model as described by Liew et al. (2024).

    Args:
        bem_model (BEM): optional BEM model as defined in MITRotor, defaults to IEA15MW with UMM momentum model
        pitch_interp (interp1d): optional 1D or 2D interpolator for pitch trajectory based on wind speed,
            if 1D by wind speed then yaw not accounted for, if 2D then wind speed and yaw accounted for. 
            Defaults to 1D IEA15MW Figure 2 (https://docs.nrel.gov/docs/fy22osti/82134.pdf).
            See ROSCOInterface.py for information on how to generate 2D control scheme interpolator.
        tsr_interp (interp1d): optional  1D interpolator for tsr trajectory based on wind speed,
            if 1D by wind speed then yaw not accounted for, if 2D then wind speed and yaw accounted for. 
            Defaults to IEA15MW Figure 2 (https://docs.nrel.gov/docs/fy22osti/82134.pdf).
            See ROSCOInterface.py for information on how to generate 2D control scheme interpolator.

    Methods:
        power
        thrust_coefficient
        axial_induction
    """
    # user can define a BEM model if they want a different rotor, momentum model, or geometry
    bem_model = field(init = True, factory = default_bem_factory, type = BEM)
    gen_eff = field(init = True, default = 95.756, type=Optional[float]) # [%]
    eff_ratio = field(init=True, default=None, type=Optional[float])  # allow override

    # create interp objects based on pitch and tsr csvs
    pitch_interp = field(init = True, factory = default_pitch_interp, type = Optional[InterpLike], repr = False)
    pitch_rad = field(init = True, default = True, type = bool)
    tsr_interp   = field(init = True, factory = default_tsr_interp, type = Optional[InterpLike], repr=False)
    rated_rotor_speed = field(init = True, default = None, type = Optional[float])  # [rad/s]

    # save most recent solution by unique floris arguments
    _last_key = field(init=False, default=None, type = bytes)
    _a = field(init=False, default=None, type = NDArrayFloat)
    _Ct = field(init=False, default=None, type = NDArrayFloat)
    _u4 = field(init=False, default=None, type = NDArrayFloat)
    _v4 = field(init=False, default=None, type = NDArrayFloat)
    _w4 = field(init=False, default=None, type = NDArrayFloat)
    _power = field(init=False, default=None, type = NDArrayFloat)

    # calculate a few needed fields post-initialization
    def __attrs_post_init__(self):
        if self.eff_ratio is None:
            gearbox_eff = self.bem_model.rotor.rosco_values[1]
            self.eff_ratio = (self.gen_eff / 100.0) * (gearbox_eff / 100.0)
        if self.pitch_interp is None:
            self.pitch_interp = default_pitch_interp()   # 1D legacy default
        if self.tsr_interp is None:
            self.tsr_interp = default_tsr_interp()       # 1D legacy default

    def _get_state_key(self, velocities: np.ndarray, yaw_angles: np.ndarray, tilt_angles: np.ndarray) -> tuple:
        # saves key to uniquely identify farm state -> avoids re-solving for calls to power, thrust, and induction for same state
        return velocities.tobytes(), yaw_angles.tobytes(), tilt_angles.tobytes()

    def _update_solution(self,
        velocities: NDArrayFloat,
        air_density: float,
        yaw_angles: NDArrayFloat,
        tilt_angles: NDArrayFloat,
        average_method: str = "cubic-mean",
        cubature_weights: Optional[NDArrayFloat] = None,
        power_thrust_table: Optional[dict] = None,
        **_,
    ):
        # create cache key for current inputs
        key = self._get_state_key(velocities, yaw_angles, tilt_angles)
        # update solution if conditions are different
        if key != self._last_key:
            n_findex, n_turbines = yaw_angles.shape

            # save new key and clear fields
            self._last_key = key
            self._a = np.empty((n_findex, n_turbines), dtype=floris_float_type)
            self._Ct = np.empty((n_findex, n_turbines), dtype=floris_float_type)
            self._u4 = np.empty((n_findex, n_turbines), dtype=floris_float_type)
            self._v4 = np.empty((n_findex, n_turbines), dtype=floris_float_type)
            self._w4 = np.empty((n_findex, n_turbines), dtype=floris_float_type)
            self._power = np.empty((n_findex, n_turbines), dtype=floris_float_type)

            # compute the power-effective wind speed across the rotor
            rotor_average_velocities = average_velocity( # NOT adjusted for yaw
                velocities=velocities,
                method=average_method,
                cubature_weights=cubature_weights,
            )
            # calculate rotor area
            rotor_area = np.pi * self.bem_model.rotor.R**2 

            # get setpoints
            yaw, tilt = np.deg2rad(yaw_angles), np.deg2rad(tilt_angles)
            eff_yaw = calc_eff_yaw(yaw, tilt)

            pitch = query_controls(
                self.pitch_interp, rotor_average_velocities, eff_yaw, kind = "pitch"
            )
            tsr = query_controls(
                self.tsr_interp, rotor_average_velocities, eff_yaw, kind = "tsr",
                rated_rotor_speed = self.rated_rotor_speed, rotor_radius = self.bem_model.rotor.R,
            )
            if not self.pitch_rad:
                pitch = np.deg2rad(pitch)

            # solve BEM for setpoints from control curves
            for tindex in range(n_turbines):
                # solve BEM
                bem_sol = self.bem_model(pitch[:, tindex], tsr[:, tindex], yaw = yaw[:, tindex], tilt = tilt[:, tindex])
                # get induction and thrust coeff
                self._a[:, tindex] = bem_sol.a()
                self._Ct[:, tindex] = bem_sol.Ct()
                # get near wake velocities
                self._u4 = bem_sol.u4,
                self._v4 = bem_sol.v4
                self._w4 = bem_sol.w4 
                # compute power 
                self._power[:, tindex] = (
                    0.5 * bem_sol.Cp() * air_density * rotor_area
                    * (rotor_average_velocities[:, tindex])**3
                    * self.eff_ratio
                )
        return
    
    def power(self, **kwargs) -> NDArrayFloat:
        self._update_solution(**kwargs)
        return self._power

    def thrust_coefficient(self, **kwargs) -> NDArrayFloat:
        self._update_solution(**kwargs)
        return self._Ct

    def axial_induction(self, **kwargs) -> NDArrayFloat:
        self._update_solution(**kwargs)
        return self._a
    
    def near_wake_velocities(self, **kwargs) -> NDArrayFloat:
        self._update_solution(**kwargs)
        return (self._u4, self._v4, self._w4)