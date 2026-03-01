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
    pitch_table = df["Pitch [deg]"].to_numpy()
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
        pitch_csv (str): optional path to pitch trajectory based on wind speed, defaults to IEA15MW Figure 2 (https://docs.nrel.gov/docs/fy22osti/82134.pdf)
        tsr_csv (str)): optional path to tsr trajectory based on wind speed, defaults to IEA15MW Figure 2 (https://docs.nrel.gov/docs/fy22osti/82134.pdf)

    Methods:
        power
        thrust_coefficient
        axial_induction
    """
    # user can define a BEM model if they want a different rotor, momentum model, or geometry
    bem_model = field(init = True, factory = default_bem_factory, type = BEM)

    # create interp objects based on pitch and tsr csvs
    pitch_interp = field(init=True, factory=default_pitch_interp, type = interp1d, repr = False)
    tsr_interp = field(init=True, factory=default_tsr_interp, type = interp1d, repr = False)

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
            rotor_average_velocities = average_velocity(
                velocities=velocities,
                method=average_method,
                cubature_weights=cubature_weights,
            )

            # calculate rotor area
            rotor_area = np.pi * self.bem_model.rotor.R**2 

            # loop over flow conditions -> TODO: should this be vectorized?
            for findex in range(n_findex):
                for tindex in range(n_turbines):
                    # get setpoints
                    vel = rotor_average_velocities[findex, tindex]
                    yaw, tilt = np.deg2rad(yaw_angles[findex, tindex]), np.deg2rad(tilt_angles[findex, tindex])
                    pitch = np.deg2rad(self.pitch_interp(vel))
                    tsr = self.tsr_interp(vel)
                    # solve BEM
                    bem_sol = self.bem_model(pitch, tsr, yaw = yaw, tilt = tilt)
                    # get induction and thrust coeff
                    self._a[findex, tindex] = bem_sol.a()
                    self._Ct[findex, tindex] = bem_sol.Ct()
                    # compute power
                    self._power[findex, tindex] = 0.5 * bem_sol.Cp() * air_density * rotor_area * (vel)**3
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