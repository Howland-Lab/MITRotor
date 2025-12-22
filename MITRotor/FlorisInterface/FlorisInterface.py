import numpy as np
from attrs import define, field
from typing import Literal, Optional
from scipy.interpolate import interp1d
# FLORIS Imports
from floris.type_dec import floris_float_type, NDArrayFloat
from floris.core.turbine.operation_models import BaseOperationModel
from floris.core.rotor_velocity import average_velocity
# MITRotor Imports
from MITRotor.ReferenceTurbines import IEA15MW
from MITRotor.Momentum import UnifiedMomentum
from MITRotor.Geometry import BEMGeometry
from MITRotor.BEMSolver import BEM, BEMSolution

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
    return interp1d(x, y, kind="linear", bounds_error=False, fill_value="extrapolate")

@define
class MITRotorTurbine(BaseOperationModel):
    """
    A turbine operation model that calls MITRotor.
    """
    # user can define a BEM model if they want a different rotor, momentum model, or geometry
    default_bem = BEM(
        rotor=IEA15MW(),
        momentum_model = UnifiedMomentum(averaging = "rotor"),
        geometry = BEMGeometry(Nr = 10, Ntheta = 20),
    )
    bem_model = field(init = False, default = default_bem, type = BEM)
    # save most recent solution by unique floris arguments
    _bem_sol = field(init=False, default=None, type = Optional[list[BEMSolution]])
    _avg_vels = field(init=False, default=None, type = Optional[NDArrayFloat])
    _last_key = field(init=False, default=None, type = bytes)
    # save blade pitch and tsr interpolation objects
    # TODO -> figure out how to make csv change with rotor type
    _pitch_interp = field(init = False, default = csv_to_interp("pitch_15mw.csv"))
    _tsr_interp = field(init = False, default= csv_to_interp("tsr_15mw.csv"))

    def _get_solution_key(self, velocities: np.ndarray) -> bytes: # TODO: add more inputs
        # Fast, deterministic, and explicit
        return velocities.tobytes()

    def _get_solutions(self,
        velocities: NDArrayFloat,
        yaw_angles: NDArrayFloat,
        tilt_angles: NDArrayFloat,
    ):
        n_findex, n_turbines = yaw_angles.shape
        # create cache key for current inputs
        key = self._get_solution_key(velocities) # TODO: add more inputs
        # update solution if conditions are different
        if key != self._last_key:
            self._bem_sol = [None] * n_findex
            self._avg_vels = np.empty((n_findex, n_turbines), dtype=velocities.dtype)
            self._last_key = key
            # loop over flow conditions
            for findex in range(n_findex):
                cond_vels, cond_yaws, cond_tilts = velocities[findex], yaw_angles[findex], tilt_angles[findex]
                rotor_avg_vels = average_velocity(cond_vels, method="cubic-mean") # TODO: does method need to be user input?
                pitch_vals = self._pitch_interp(rotor_avg_vels)
                tsr_vals = self._tsr_interp(rotor_avg_vels)
                self._bem_sol[findex] = self.bem_model(pitch_vals, tsr_vals, yaw = cond_yaws, tilt = cond_tilts)
                self._avg_vels[findex] = rotor_avg_vels

        return self._bem_sol
    
    def power(self,
        velocities: NDArrayFloat,
        yaw_angles: NDArrayFloat,
        tilt_angles: NDArrayFloat,
        **_
    ) -> NDArrayFloat:
        return self._get_solutions(velocities, yaw_angles, tilt_angles).Cp() # TODO: what type of averaging do we want? AND make into POWER


    def thrust_coefficient(self,
        velocities: NDArrayFloat,
        yaw_angles: NDArrayFloat,
        tilt_angles: NDArrayFloat,
        **_
    ) -> NDArrayFloat:
        return self._get_solutions(velocities, yaw_angles, tilt_angles).Ct() # TODO: what type of averaging do we want?


    def axial_induction(self,
        velocities: NDArrayFloat,
        yaw_angles: NDArrayFloat,
        tilt_angles: NDArrayFloat,
        **_
    ) -> NDArrayFloat:
        return self._get_solutions(velocities, yaw_angles, tilt_angles).a() # TODO: what type of averaging do we want?