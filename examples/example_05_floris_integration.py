import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from floris import FlorisModel, TimeSeries
from MITRotor.FlorisInterface.FlorisInterface import csv_to_interp, default_bem_factory, MITRotorTurbine

# ------------------ run basic case --------------------------------------------------------
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
fmodel.set_operation_model(MITRotorTurbine())
fmodel.run()
print("Powers [W]:\n", fmodel.get_turbine_powers(), "\n")
print("Thrust coefficients [-]:\n", fmodel.get_turbine_thrust_coefficients(), "\n")
print("Axial induction factors [-]:\n", fmodel.get_turbine_axial_induction_factors(), "\n")

# -------------------- plot pitch and tsr control curves, as well as CT for IEA15MW ------------------
module_dir = os.path.dirname(__file__)  # examples/
csv_dir = os.path.join(module_dir, "..", "MITRotor", "FlorisInterface")

wind_speeds = np.linspace(5, 25.0, 20)
wind_dirs = np.full_like(wind_speeds, 270.0)
turbulence_intensity = np.zeros_like(wind_speeds)

# plot pitch and tsr control curves for IEA15MW from figure 2 (https://docs.nrel.gov/docs/fy22osti/82134.pdf)
pitch_csv = os.path.join(csv_dir, "pitch_15mw.csv")
tsr_csv   = os.path.join(csv_dir, "tsr_15mw.csv")
pitch_interp = csv_to_interp(pitch_csv)
tsr_interp   = csv_to_interp(tsr_csv)

# plot interpolated pitch and tsr data
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(
    wind_speeds,
    pitch_interp(wind_speeds),
    s=40,
    edgecolors="k",
    label = "Interpolated Pitch [deg]"
)
ax.scatter(
    wind_speeds,
    tsr_interp(wind_speeds),
    s=40,
    edgecolors="k",
    label = "Interpolated Tip-Speed Ratio [-]"
)

# load and plot raw CSV data
tsr_data = np.loadtxt(tsr_csv, delimiter=",", skiprows=1)
pitch_data = np.loadtxt(pitch_csv, delimiter=",", skiprows=1)
tsr_ws, tsr_vals = tsr_data[:, 0], tsr_data[:, 1]
pitch_ws, pitch_vals = pitch_data[:, 0], pitch_data[:, 1]
ax.plot(
    pitch_ws,
    pitch_vals,
    label = "Pitch [deg]"
)
ax.plot(
    tsr_ws,
    tsr_vals,
    label = "Tip-Speed Ratio [-]"
)
ax.set_xlabel("Wind Speed [m/s]")
ax.set_title("IEA 15MW: Fixed Bottom Trajectories")
plt.legend()

# compute CT from floris
fmodel = FlorisModel("defaults")

time_series = TimeSeries(
    wind_speeds=wind_speeds,
    wind_directions=wind_dirs,
    turbulence_intensities=turbulence_intensity,
)

fmodel.set(layout_x = [0.0], layout_y = [0.0], wind_data = time_series)
fmodel.set_operation_model(MITRotorTurbine())
fmodel.run()
floris_Ct = fmodel.get_turbine_thrust_coefficients()

fig, ax = plt.subplots(figsize=(8, 6))
cmap = cm.viridis

# plot setpoint curves for IEA15MW from figure 3.2 (https://docs.nrel.gov/docs/fy20osti/75698.pdf)
setpoint_curves = os.path.join(csv_dir, "pitch_tsr_ct_15mw.csv")
data = np.loadtxt(setpoint_curves, delimiter=",", skiprows=1)
all_cts = np.concatenate([data[:, 2], floris_Ct.flatten()])
norm = Normalize(vmin=all_cts.min(), vmax=all_cts.max())

for ct in np.unique(data[:, 2]):
    mask = data[:, 2] == ct
    ax.plot(
        data[mask, 0],
        data[mask, 1],
        color=cmap(norm(ct)),
        linewidth=2,
        label=f"Ct ≈ {ct:.3f}",
    )

# plot FLORIS points
sc = ax.scatter(
    pitch_interp(wind_speeds),
    tsr_interp(wind_speeds),
    c=floris_Ct,
    cmap=cmap,
    norm=norm,
    s=40,
    edgecolors="k",
    label = "FLORIS with MITRotor"
)

# colorbar that covers all Ct values
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
plt.colorbar(sm, ax=ax, label="$C_T$")

# label plot
ax.set_xlabel("Pitch [deg]")
ax.set_ylabel("Tip-Speed Ratio [-]")
ax.set_title("IEA 15MW: $C_T$ Control Curves vs FLORIS/MITRotor")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()
