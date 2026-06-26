import os
import numpy as np
import polars as pl
from attrs import define, field
from typing import Optional
from scipy.interpolate import interp1d
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

# default rotor if none provided by user (IEA 15MW)
def default_bem_factory():
    return BEM(
        rotor=IEA15MW(),
        momentum_model=UnifiedMomentum(averaging="rotor"),
        geometry=BEMGeometry(Nr=10, Ntheta=20),
        tiploss_model=NoTipLoss()
    )
# pitch vs windspeed interpolater if none provided by user
# for IEA 15MW from figure 2 (https://docs.nrel.gov/docs/fy22osti/82134.pdf)
def default_pitch_interp():
    module_dir = os.path.dirname(__file__)
    pitch_file = os.path.join(module_dir, "IEA_15mw_rotor.csv")
    df = pl.read_csv(pitch_file)
    wind_table = df["Wind [m/s]"].to_numpy()
    pitch_table = np.deg2rad(df["Pitch [deg]"])
    # TODO: should fill_value be extrapolate?
    return interp1d(wind_table, pitch_table, kind="linear", fill_value="extrapolate", bounds_error=False)

# tsr vs windspeed interpolater if none provided by user
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
        pitch_interp (interp1d): optional 1D interpolator for pitch trajectory based on wind speed,
            defaults to IEA15MW Figure 2 (https://docs.nrel.gov/docs/fy22osti/82134.pdf)
        tsr_interp (interp1d): optional  1D interpolator for tsr trajectory based on wind speed,
            defaults to IEA15MW Figure 2 (https://docs.nrel.gov/docs/fy22osti/82134.pdf)

    Methods:
        power
        thrust_coefficient
        axial_induction
    """
    # user can define a BEM model if they want a different rotor, momentum model, or geometry
    bem_model = field(init = True, factory = default_bem_factory, type = BEM)

    # create interp objects based on pitch and tsr csvs
    pitch_interp = field(init = True, factory = default_pitch_interp, type = interp1d, repr = False)
    pitch_rad = field(init = True, default = True, type = bool)
    tsr_interp = field(init = True, factory = default_tsr_interp, type = interp1d, repr = False)
    rated_rotor_speed = field(init=True, default=None, type=Optional[float])  # rad/s

    # save most recent solution by unique floris arguments
    _last_key = field(init=False, default=None, type = bytes)
    _a = field(init=False, default=None, type = NDArrayFloat)
    _Ct = field(init=False, default=None, type = NDArrayFloat)
    _power = field(init=False, default=None, type = NDArrayFloat)

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
            self._power = np.empty((n_findex, n_turbines), dtype=floris_float_type)

            # compute the power-effective wind speed across the rotor
            rotor_average_velocities = average_velocity( # NOT adjusted for yaw
                velocities=velocities,
                method=average_method,
                cubature_weights=cubature_weights,
            )
            # self.farm.cosine_loss_exponent_yaw
            pW = power_thrust_table["cosine_loss_exponent_yaw"] / 3.0
            rotor_normal_average_velocities = rotor_average_velocities  * (cosd(yaw_angles) ** pW)
            # rotor_normal_average_velocities = rotor_average_velocities * cosd(yaw_angles) * cosd(tilt_angles)
            # calculate rotor area
            rotor_area = np.pi * self.bem_model.rotor.R**2 

            # get setpoints
            yaw, tilt = np.deg2rad(yaw_angles), np.deg2rad(tilt_angles)
            pitch = self.pitch_interp(rotor_normal_average_velocities)
            if not self.pitch_rad:
                pitch = np.deg2rad(pitch)

            if self.rated_rotor_speed is None:
                tsr = self.tsr_interp(rotor_normal_average_velocities)
            else:
                R = self.bem_model.rotor.R
                tsr_lookup = self.tsr_interp(rotor_normal_average_velocities)
                omega_lookup = tsr_lookup * rotor_normal_average_velocities / R  # rad/s implied by lookup
                tsr_from_rated_speed = self.rated_rotor_speed * R / rotor_average_velocities  # above-rated branch
                tsr = np.where(omega_lookup <= self.rated_rotor_speed, tsr_lookup, tsr_from_rated_speed)
            tsr = np.maximum(tsr, 0.0)

            for tindex in range(n_turbines):
                # solve BEM
                bem_sol = self.bem_model(pitch[:, tindex], tsr[:, tindex], yaw = yaw[:, tindex], tilt = tilt[:, tindex])
                # get induction and thrust coeff
                self._a[:, tindex] = bem_sol.a()
                self._Ct[:, tindex] = bem_sol.Ct()
                # compute power
                self._power[:, tindex] = 0.5 * bem_sol.Cp() * air_density * rotor_area * (rotor_average_velocities[:, tindex])**3
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