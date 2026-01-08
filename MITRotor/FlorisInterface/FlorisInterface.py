import numpy as np
import os
from attrs import define, field
from typing import Optional
from scipy.interpolate import interp1d
# FLORIS Imports
from floris.type_dec import floris_float_type, NDArrayFloat
from floris.core.turbine.operation_models import BaseOperationModel
from floris.core.rotor_velocity import average_velocity, rotor_velocity_air_density_correction
# MITRotor / UMM Imports
from MITRotor.ReferenceTurbines import IEA15MW
from MITRotor.Momentum import UnifiedMomentum
from MITRotor.Geometry import BEMGeometry
from MITRotor.BEMSolver import BEM
from UnifiedMomentumModel.Utilities.Geometry import calc_eff_yaw

# default rotor if none provided by user (IEA 15MW)
def default_bem_factory():
    return BEM(
        rotor=IEA15MW(),
        momentum_model=UnifiedMomentum(averaging="rotor"),
        geometry=BEMGeometry(Nr=10, Ntheta=20),
    )
# pitch vs windspeed curve if none provided by user
# for IEA 15MW from figure 2 (https://docs.nrel.gov/docs/fy22osti/82134.pdf)
def default_pitch_csv():
    module_dir = os.path.dirname(__file__)
    return os.path.join(module_dir, "pitch_15mw.csv")

# tsr vs windspeed curve if none provided by user
# for IEA 15MW from figure 2 (https://docs.nrel.gov/docs/fy22osti/82134.pdf)
def default_tsr_csv():
    module_dir = os.path.dirname(__file__)
    return os.path.join(module_dir, "tsr_15mw.csv")

def csv_to_interp(csv_file):
    # read in csv
    data = np.loadtxt(csv_file, delimiter=",", skiprows=1)
    # split data into x (wind speed) and y (either pitch or tsr)
    x = data[:, 0]
    y = data[:, 1]
    # sort by x (wind speed)
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    # return interpolator for y
    return interp1d(x, y, kind="linear", fill_value="extrapolate", bounds_error=False) # TODO: should fill_value be extrapolate?

@define
class MITRotorTurbine(BaseOperationModel):
    """
    A turbine operation model that calls MITRotor.
    """
    # user can define a BEM model if they want a different rotor, momentum model, or geometry
    bem_model = field(init = True, factory = default_bem_factory, type = BEM)

    # user can define csv paths for pitch and tsr values
    pitch_csv = field(init = True, factory = default_pitch_csv, type = str)
    tsr_csv = field(init = True, factory = default_tsr_csv, type = str)

    # create interp objects based on pitch and tsr csvs
    _pitch_interp = field(init=False, default=None, type = interp1d, repr = False)
    _tsr_interp = field(init=False, default=None, type = interp1d, repr = False)

    # save most recent solution by unique floris arguments
    _last_key = field(init=False, default=None, type = bytes)
    _a = field(init=False, default=None, type = NDArrayFloat)
    _Ct = field(init=False, default=None, type = NDArrayFloat)
    _power = field(init=False, default=None, type = NDArrayFloat)

    def __attrs_post_init__(self):
        # creates interpolation objects
        self._pitch_interp = csv_to_interp(self.pitch_csv)
        self._tsr_interp = csv_to_interp(self.tsr_csv)

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
                    pitch = np.deg2rad(self._pitch_interp(vel))
                    tsr = self._tsr_interp(vel)
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