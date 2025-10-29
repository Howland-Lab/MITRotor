import numpy as np
import matplotlib.pyplot as plt
from MITRotor import BEM, IEA15MW, UnifiedMomentum, IEA15MW, BEMGeometry
from UnifiedMomentumModel.Utilities.Geometry import calc_eff_yaw
from pathlib import Path

figdir = Path("fig")
# Initialize rotor using the IEA10MW reference wind turbine model.
rotor = IEA15MW()
Nr, Ntheta = 10, 20
bem = BEM(rotor=rotor, momentum_model=UnifiedMomentum(averaging="rotor"), geometry = BEMGeometry(Nr=Nr, Ntheta=Ntheta))

# solve BEM for a control set point.
yaw = np.deg2rad(20)
tilt = yaw
misalignment = calc_eff_yaw(yaw, tilt)
pitch, tsr = np.deg2rad(0), 7.0

yaw_sol = bem(pitch, tsr, yaw = misalignment, tilt = 0)
tilt_sol = bem(pitch, tsr, yaw = 0, tilt = misalignment)
yaw_tilt_sol = bem(pitch, tsr, yaw = yaw, tilt = tilt)

fig, axes = plt.subplots(nrows = 2, ncols = 5, figsize = (12, 6), sharex=True, sharey=False)
for idx in range(0, Nr):
    i = 0 if idx < 5 else 1
    j = np.mod(idx, 5)
    r_mesh = yaw_sol.geom.mu_mesh[idx, :]
    theta_mesh = yaw_sol.geom.theta_mesh[idx, :]
    yaw_val = yaw_sol.Cn(grid = "sector")[idx, :]
    tilt_val = tilt_sol.Cn(grid = "sector")[idx, :]
    yaw_tilt_val = yaw_tilt_sol.Cn(grid = "sector")[idx, :]

    rounded_deg_misalignment = np.round(np.rad2deg(misalignment), decimals=2)
    axes[i, j].plot(theta_mesh, yaw_val, c = "r")
    axes[i, j].plot(theta_mesh, tilt_val, c = "blue")
    axes[i, j].plot(theta_mesh, yaw_tilt_val, c = "purple")
    if i == 1:
        axes[i, j].set_xlabel("$\\theta$ (radians)")
    if j == 0:
        axes[i, j].set_ylabel("$C_T$")
    axes[i, j].set_title(f"r/R ={np.round(r_mesh[0], decimals=2)}")

fig.legend(
    [f"Yaw ${rounded_deg_misalignment}^\circ$", f"Tilt ${rounded_deg_misalignment}^\circ$", f"Yaw ${np.rad2deg(yaw)}^\circ$ and Tilt ${np.rad2deg(yaw)}^\circ$"],  # labels
    loc='lower center',
    ncol=3,
    bbox_to_anchor=(0.5, 0.05)
)
fig.subplots_adjust(bottom=0.2, hspace= 0.5, wspace = 0.75)
fig.suptitle("Azimuthal $C_T$ Values", size = 16)
plt.savefig(figdir / "example_4_ct_azimuthal_over_radius.png", dpi=300)
