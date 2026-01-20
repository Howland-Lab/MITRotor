import os
import numpy as np
import polars as pl
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from floris import FlorisModel, TimeSeries
from MITRotor.FlorisInterface.FlorisInterface import MITRotorTurbine, default_bem_factory, default_pitch_interp, default_tsr_interp

figdir = Path("fig")
floris_air_density = 1.225
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
# Credit to Ilan Upfal for initial validation of MITRotor vs IEA15MW and much of the script below.
# Floris Interface written and tested by Skylar Gering.
module_dir = os.path.dirname(__file__)  # examples/
csv_file = os.path.join(module_dir, "..", "MITRotor", "FlorisInterface", "IEA_15mw_rotor.csv")
df = pl.read_csv(csv_file)

wind_table = df["Wind [m/s]"].to_numpy()
wind_speeds = np.linspace(3, 25, 50)
wind_dirs = np.full_like(wind_speeds, 270.0)
turbulence_intensity = np.zeros_like(wind_speeds)

# ------- plot pitch and tsr control curves for IEA15MW from figure 2 (https://docs.nrel.gov/docs/fy22osti/82134.pdf) --------
pitch_interp = default_pitch_interp()
tsr_interp   = default_tsr_interp()
tsrs = [tsr_interp(u) for u in wind_speeds]
pitches = [pitch_interp(u) for u in wind_speeds]

# plot interpolated pitch and tsr data
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(
    wind_speeds,
    pitches,
    s=40,
    edgecolors="k",
    label = "Interpolated Pitch [deg]"
)
ax.scatter(
    wind_speeds,
    tsrs,
    s=40,
    edgecolors="k",
    label = "Interpolated Tip-Speed Ratio [-]"
)
# load and plot raw CSV data
ax.plot(
    wind_table,
    df["Pitch [deg]"].to_numpy(),
    label = "Pitch [deg]"
)
ax.plot(
    wind_table,
    df["Tip Speed [m/s]"].to_numpy() / wind_table,
    label = "Tip-Speed Ratio [-]"
)
ax.set_title("IEA 15MW: Fixed Bottom Trajectories", size = 18)
ax.set_xlabel("Wind Speed [m/s]", size = 16)
ax.tick_params(labelsize=14)
ax.legend(fontsize = 14)
plt.savefig(figdir / "example_5_pitch_tsr_interpolation.png", dpi=300)

# -------- plot CT and CP values against one another and against IEA15MW from figure 3.1-C (https://docs.nrel.gov/docs/fy20osti/75698.pdf) -------
# solve Madsen Momentum Model though MITRotor
bem = default_bem_factory()
mit_sols = [bem(pitch=np.deg2rad(pitch_interp(u)), tsr=tsr_interp(u), yaw=0, tilt=0) for u in wind_speeds]
mit_Ct = [sol.Ct() for sol in mit_sols]
mit_Cp = [sol.Cp() for sol in mit_sols]

# solve FLORIS with MITRotor
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
rotor_area = np.pi * bem.rotor.R**2 
floris_power = np.squeeze(fmodel.get_turbine_powers())
floris_Cp =  floris_power / (0.5 * 1.225 * rotor_area * (wind_speeds)**3)

# plot CT values against one another and against IEA15MW from figure 3.1-C (https://docs.nrel.gov/docs/fy20osti/75698.pdf)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(
    wind_table,
    df["Thrust Coefficient [-]"].to_list(),
    linewidth=2,
    label="IEA15MW $C_T$",
    color='tab:orange',
    linestyle = "solid"
)

ax.plot(
    wind_speeds,
    mit_Ct,
    label="MITRotor $C_T$",
    linewidth=2,
    color='tab:orange',
    linestyle = "dashed"
)
ax.plot(
    wind_speeds,
    floris_Ct,
    label="FLORIS $C_T$",
    linewidth=2,
    color='darkorange',
    linestyle = "dotted"
)

ax.set_xlabel("Wind Speed [m/s]")


# plot Cp values against one another and against IEA15MW from figure 3.1-C (https://docs.nrel.gov/docs/fy20osti/75698.pdf)
ax.plot(
    wind_table,
    df["Aero Power Coefficient [-]"].to_list(),
    label="IEA15MW $C_P$",
    linewidth=2,
    color='tab:blue',
    linestyle = "solid"
)
ax.plot(
    wind_speeds,
    mit_Cp,
    label="MITRotor $C_P$",
    linewidth=2,
    color='tab:blue',
    linestyle = "dashed"
)
ax.plot(
    wind_speeds,
    floris_Cp,
    label="FLORIS $C_P$",
    linewidth=2,
    color='tab:blue',
    linestyle = "dotted"
)
ax.set_title("IEA 15 MW FLORIS Interface Validation", size = 18)
ax.set_xlabel("Wind Speed [m/s]", size = 16)
ax.tick_params(labelsize=14)
ax.legend(fontsize = 14)
plt.savefig(figdir / "example_5_IEA15mw_CT_CP.png", dpi=300)

