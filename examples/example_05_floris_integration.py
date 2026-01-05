from MITRotor.FlorisInterface.FlorisInterface import csv_to_interp, MITRotorTurbine
from floris import FlorisModel, TimeSeries
from floris.core.turbine.unified_momentum_model import UnifiedMomentumModelTurbine
import numpy as np
import matplotlib.pyplot as plt

from floris import FlorisModel, TimeSeries

fmodel = FlorisModel("defaults")
time_series = TimeSeries(
    wind_directions=np.array([270.0, 270.0, 280.0]),
    wind_speeds=np.array([8.0, 10.0, 12.0]),
    turbulence_intensities=np.array([0.06, 0.06, 0.06]),
)
yaw_angles = np.array([
    [0.0, 0.0],   # condition 1
    [0.0, 0.0],   # condition 2
    [0.0, 0.0],   # condition 3
])

fmodel.set(
    layout_x = [0.0, 500.0],
    layout_y = [0.0, 0.0],
    wind_data = time_series,
    yaw_angles = yaw_angles
)
fmodel.set_operation_model(UnifiedMomentumModelTurbine)

# fmodel.set_operation_model(MITRotorTurbine())

fmodel.run()

print("Powers [W]:\n", fmodel.get_turbine_powers(), "\n")
print("Thrust coefficients [-]:\n", fmodel.get_turbine_thrust_coefficients(), "\n")
print("Axial induction factors [-]:\n", fmodel.get_turbine_axial_induction_factors(), "\n")