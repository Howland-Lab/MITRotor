import numpy as np
import matplotlib.pyplot as plt
from MITRotor import BEM, IEA15MW, UnifiedMomentum, IEA15MW
from UnifiedMomentumModel.Utilities.Geometry import calc_eff_yaw

# Initialize rotor using the IEA10MW reference wind turbine model.
rotor = IEA15MW()
bem = BEM(rotor=rotor, momentum_model=UnifiedMomentum(averaging="rotor"))

# solve BEM for a control set point.
yaw = np.deg2rad(20)
tilt = yaw
misalignment = calc_eff_yaw(yaw, tilt)
pitch, tsr = np.deg2rad(0), 7.0

yaw_sol = bem(pitch, tsr, yaw = misalignment, tilt = 0)
tilt_sol = bem(pitch, tsr, yaw = 0, tilt = misalignment)
yaw_tilt_sol = bem(pitch, tsr, yaw = yaw, tilt = tilt)

idx = 5
r_mesh = yaw_sol.geom.mu_mesh[idx, :]
theta_mesh = yaw_sol.geom.theta_mesh[idx, :]
yaw_val = yaw_sol.Cn(grid = "sector")[idx, :]
tilt_val = tilt_sol.Cn(grid = "sector")[idx, :]
yaw_tilt_val = yaw_tilt_sol.Cn(grid = "sector")[idx, :]

rounded_deg_misalignment = np.round(np.rad2deg(misalignment), decimals=2)
plt.plot(theta_mesh, yaw_val, label = f"Yaw ${rounded_deg_misalignment}^\circ$", c = "r")
plt.plot(theta_mesh, tilt_val, label = f"Tilt ${rounded_deg_misalignment}^\circ$",  c = "blue")
plt.plot(theta_mesh, yaw_tilt_val, label = f"Yaw ${np.rad2deg(yaw)}^\circ$ and Tilt ${np.rad2deg(yaw)}^\circ$",  c = "purple")
plt.xlabel("$\\theta$ (radians)")
plt.ylabel("$C_T$")
plt.title(f"$C_T$ of BEM Rotor at r/R ={np.round(r_mesh[0], decimals=2)}")
plt.legend()

print("woah")