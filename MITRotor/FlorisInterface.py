import numpy as np
from attrs import define, field
from typing import Literal
# FLORIS Imports
from floris.type_dec import floris_float_type, NDArrayFloat
from floris.core.turbine.operation_models import BaseOperationModel
# MITRotor Imports
from MITRotor.ReferenceTurbines import IEA15MW
from MITRotor.RotorDefinition import RotorDefinition
from MITRotor.Geometry import BEMGeometry
from MITRotor.Momentum import UnifiedMomentum
from MITRotor.BEMSolver import BEM, BEMSolution

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
    _bem_solution: BEMSolution = field(init=False, default=None)
    _last_key: bytes = field(init=False, default=None)

    def _get_solution_key(self, velocities: np.ndarray) -> bytes: # TODO: add more inputs
        # Fast, deterministic, and explicit
        return velocities.tobytes()

    def _get_solution(self,
        power_thrust_table,
        velocities,
        air_density=None,
    ):
        # create cache key for current inputs
        key = self._velocity_key(velocities) # TODO: add more inputs
        # update solution if conditions are different
        if key != self._last_key:
            self._bem_solution = self.bem_model( # TODO: make sure these inputs are right
                power_thrust_table,
                velocities,
                air_density,
            )
            self._last_key = key
        return self._bem_solution
    
    def power(
        self,
        velocities: NDArrayFloat,
        turbulence_intensities: NDArrayFloat,
        yaw_angles: NDArrayFloat,
        tilt_angles: NDArrayFloat,
        **_
    ) -> NDArrayFloat:
        return self._get_solution(
            power_thrust_table, velocities, air_density
        ).power


    def thrust_coefficient(
        self,
        velocities: NDArrayFloat,
        **_
    ) -> NDArrayFloat:
        return self._get_solution(power_thrust_table, velocities, air_density).power


    def axial_induction(
        self,
        velocities: NDArrayFloat,
        **_
    ) -> NDArrayFloat:
        return self._get_solution(power_thrust_table, velocities, air_density).power