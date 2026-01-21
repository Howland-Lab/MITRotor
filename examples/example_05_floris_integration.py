import os
import numpy as np
import polars as pl
from pathlib import Path
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning) # JUST FOR NOW


import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Floris imports
from floris import FlorisModel, TimeSeries

# MITRotor / UMM Imports
from MITRotor.FlorisInterface.FlorisInterface import MITRotorTurbine, default_bem_factory, default_pitch_interp, default_tsr_interp
from MITRotor.ReferenceTurbines import IEA15MW
from MITRotor.Momentum import UnifiedMomentum, MadsenMomentum
from MITRotor.Geometry import BEMGeometry
from MITRotor.TipLoss import NoTipLoss
from MITRotor.BEMSolver import BEM

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
# solve UMM-BEM though MITRotor - rotor averaged
bem_rotor_umm = default_bem_factory()
mit_rotor_umm_start = time.time()
mit_sols_rotor_umm = [bem_rotor_umm(pitch=np.deg2rad(pitch_interp(u)), tsr=tsr_interp(u), yaw=0, tilt=0) for u in wind_speeds]
mit_rotor_umm_end = time.time()
mit_Ct_rotor_umm = [sol.Ct() for sol in mit_sols_rotor_umm]
mit_Cp_rotor_umm = [sol.Cp() for sol in mit_sols_rotor_umm]
print("MITRotor UMM-BEM Rotor-Averaged: " + str(mit_rotor_umm_end - mit_rotor_umm_start) + " seconds")

# solve UMM-BEM though MITRotor - annulus averaged
bem_annulus_umm = BEM(
        rotor=IEA15MW(),
        momentum_model=UnifiedMomentum(averaging="annulus"),
        geometry=BEMGeometry(Nr=10, Ntheta=20),
        tiploss_model=NoTipLoss(),
    )
mit_annulus_umm_start = time.time()
mit_sols_annulus_umm = [bem_annulus_umm(pitch=np.deg2rad(pitch_interp(u)), tsr=tsr_interp(u), yaw=0, tilt=0) for u in wind_speeds]
mit_annulus_umm_end = time.time()
mit_Ct_annulus_umm = [sol.Ct() for sol in mit_sols_annulus_umm]
mit_Cp_annulus_umm = [sol.Cp() for sol in mit_sols_annulus_umm]
print("MITRotor UMM-BEM Annulus-Averaged: " + str(mit_annulus_umm_end - mit_annulus_umm_start) + " seconds")

# solve Madsen-BEM though MITRotor - rotor averaged
bem_rotor_madesen = BEM(
        rotor=IEA15MW(),
        momentum_model=MadsenMomentum(averaging="rotor"),
        geometry=BEMGeometry(Nr=10, Ntheta=20),
        tiploss_model=NoTipLoss()
    )
mit_rotor_madsen_start = time.time()
mit_sols_rotor_madsen = [bem_rotor_madesen(pitch=np.deg2rad(pitch_interp(u)), tsr=tsr_interp(u), yaw=0, tilt=0) for u in wind_speeds]
mit_rotor_madsen_end = time.time()
mit_Ct_rotor_madsen = [sol.Ct() for sol in mit_sols_rotor_madsen]
mit_Cp_rotor_madsen = [sol.Cp() for sol in mit_sols_rotor_madsen]
print("MITRotor Madsen-BEM Rotor-Averaged: " + str(mit_rotor_madsen_end - mit_rotor_madsen_start) + " seconds")

# solve Madsen-BEM though MITRotor - annulus averaged
bem_annulus_madesen = BEM(
        rotor=IEA15MW(),
        momentum_model=MadsenMomentum(averaging="annulus"),
        geometry=BEMGeometry(Nr=10, Ntheta=20),
        tiploss_model=NoTipLoss()
    )
mit_annulus_madsen_start = time.time()
mit_sols_annulus_madsen = [bem_annulus_madesen(pitch=np.deg2rad(pitch_interp(u)), tsr=tsr_interp(u), yaw=0, tilt=0) for u in wind_speeds]
mit_annulus_madsen_end = time.time()
mit_Ct_annulus_madsen = [sol.Ct() for sol in mit_sols_annulus_madsen]
mit_Cp_annulus_madsen = [sol.Cp() for sol in mit_sols_annulus_madsen]
print("MITRotor Madsen-BEM Annulus-Averaged: " + str(mit_annulus_madsen_end - mit_annulus_madsen_start) + " seconds")

# solve FLORIS  with UMM-BEM though MITRotor - rotor averaged
time_series = TimeSeries(
    wind_speeds=wind_speeds,
    wind_directions=wind_dirs,
    turbulence_intensities=turbulence_intensity,
)
fmodel_rotor_umm = FlorisModel("defaults")
fmodel_rotor_umm.set(layout_x = [0.0], layout_y = [0.0], wind_data = time_series)
fmodel_rotor_umm.set_operation_model(MITRotorTurbine()) # default bem_model uses rotor-averaging
floris_rotor_umm_start = time.time()
fmodel_rotor_umm.run()
floris_rotor_umm_end = time.time()
floris_Ct_rotor_umm = fmodel_rotor_umm.get_turbine_thrust_coefficients()
rotor_area = np.pi * bem_rotor_umm.rotor.R**2 
floris_power_rotor_umm = np.squeeze(fmodel_rotor_umm.get_turbine_powers())
floris_Cp_rotor_umm =  floris_power_rotor_umm / (0.5 * 1.225 * rotor_area * (wind_speeds)**3)
print("FLORIS UMM-BEM Rotor-Averaged: " + str(floris_rotor_umm_end - floris_rotor_umm_start) + " seconds")

# solve FLORIS  with UMM-BEM though MITRotor - annulus averaged
fmodel_annulus_umm = FlorisModel("defaults")
fmodel_annulus_umm.set(layout_x = [0.0], layout_y = [0.0], wind_data = time_series)
fmodel_annulus_umm.set_operation_model(MITRotorTurbine(bem_model = bem_annulus_umm)) # default bem_model uses rotor-averaging
floris_annulus_umm_start = time.time()
fmodel_annulus_umm.run()
floris_annulus_umm_end = time.time()
floris_Ct_annulus_umm = fmodel_annulus_umm.get_turbine_thrust_coefficients()
floris_power_annulus_umm = np.squeeze(fmodel_annulus_umm.get_turbine_powers())
floris_Cp_annulus_umm =  floris_power_annulus_umm / (0.5 * 1.225 * rotor_area * (wind_speeds)**3)
print("FLORIS UMM-BEM Annulus-Averaged: " + str(floris_annulus_umm_end - floris_annulus_umm_start) + " seconds")

# solve FLORIS  with Madsen-BEM though MITRotor - rotor averaged
fmodel_rotor_madsen = FlorisModel("defaults")
fmodel_rotor_madsen.set(layout_x = [0.0], layout_y = [0.0], wind_data = time_series)
fmodel_rotor_madsen.set_operation_model(MITRotorTurbine(bem_model = bem_rotor_madesen)) # default bem_model uses rotor-averaging
floris_rotor_madsen_start = time.time()
fmodel_rotor_madsen.run()
floris_rotor_madsen_end = time.time()
floris_Ct_rotor_madsen = fmodel_rotor_madsen.get_turbine_thrust_coefficients()
floris_power_rotor_madsen = np.squeeze(fmodel_rotor_madsen.get_turbine_powers())
floris_Cp_rotor_madsen =  floris_power_rotor_madsen / (0.5 * 1.225 * rotor_area * (wind_speeds)**3)
print("FLORIS Madsen-BEM Rotor-Averaged: " + str(floris_rotor_madsen_end - floris_rotor_madsen_start) + " seconds")

# solve FLORIS  with Madsen-BEM though MITRotor - annulus averaged
fmodel_annulus_madsen = FlorisModel("defaults")
fmodel_annulus_madsen.set(layout_x = [0.0], layout_y = [0.0], wind_data = time_series)
fmodel_annulus_madsen.set_operation_model(MITRotorTurbine(bem_model = bem_annulus_madesen)) # default bem_model uses rotor-averaging
floris_annulus_madsen_start = time.time()
fmodel_annulus_madsen.run()
floris_annulus_madsen_end = time.time()
floris_Ct_annulus_madsen = fmodel_annulus_madsen.get_turbine_thrust_coefficients()
floris_power_annulus_madsen = np.squeeze(fmodel_annulus_madsen.get_turbine_powers())
floris_Cp_annulus_madsen =  floris_power_annulus_madsen / (0.5 * 1.225 * rotor_area * (wind_speeds)**3)
print("FLORIS Madsen-BEM Annulus-Averaged: " + str(floris_annulus_madsen_end - floris_annulus_madsen_start) + " seconds")

# plot CT values against one another and against IEA15MW from figure 3.1-C (https://docs.nrel.gov/docs/fy20osti/75698.pdf)
fig, (ax0, ax1) = plt.subplots(figsize=(16, 8), ncols = 2, sharex = True, sharey = True)
fig.suptitle("IEA 15 MW FLORIS Interface Validation", size = 18)
alpha = 0.8
ax0.plot(
    wind_table,
    df["Thrust Coefficient [-]"].to_list(),
    linewidth=4,
    label="IEA15MW",
    color='tab:orange',
    linestyle = "solid",
    alpha = alpha,
)

ax0.plot(
    wind_speeds,
    mit_Ct_rotor_umm,
    label="MITRotor UMM Rotor-Averaged",
    linewidth=4,
    color='tab:blue',
    linestyle = "solid",
    alpha = alpha,
)
ax0.plot(
    wind_speeds,
    mit_Ct_annulus_umm,
    label="MITRotor UMM Annulus-Averaged",
    linewidth=4,
    color='tab:green',
    linestyle = "solid",
    alpha = alpha,
)
# ax0.plot(
#     wind_speeds,
#     mit_Ct_rotor_madsen,
#     label="MITRotor Madsen Rotor-Averaged",
#     linewidth=4,
#     color='tab:purple',
#     linestyle = "solid",
#     alpha = alpha,
# )
# ax0.plot(
#     wind_speeds,
#     mit_Ct_annulus_madsen,
#     label="MITRotor Madsen Annulus-Averaged",
#     linewidth=4,
#     color='tab:red',
#     linestyle = "solid",
#     alpha = alpha,
# )
ax0.plot(
    wind_speeds,
    floris_Ct_rotor_umm,
    label="FLORIS-MITRotor UMM Rotor-Averaged",
    linewidth=4,
    color='tab:blue',
    linestyle = "dashed",
    alpha = alpha,
)
ax0.plot(
    wind_speeds,
    floris_Ct_annulus_umm,
    label="FLORIS-MITRotor UMM Annulus-Averaged",
    linewidth=4,
    color='tab:green',
    linestyle = "dashed",
    alpha = alpha,
)

ax0.plot(
    wind_speeds,
    floris_Ct_rotor_madsen,
    label="FLORIS-MITRotor Madsen Rotor-Averaged",
    linewidth=4,
    color='tab:purple',
    linestyle = "dashed",
    alpha = alpha,
)
ax0.plot(
    wind_speeds,
    floris_Ct_annulus_madsen,
    label="FLORIS-MITRotor Madsen Annulus-Averaged",
    linewidth=4,
    color='tab:red',
    linestyle = "dashed",
    alpha = alpha,
)

ax0.set_xlabel("Wind Speed [m/s]", size = 16)
ax0.set_ylabel("$C_T$", size = 16)
ax0.tick_params(labelsize = 14)
ax0.set_title("$C_T$", size = 18)

# plot Cp values against one another and against IEA15MW from figure 3.1-C (https://docs.nrel.gov/docs/fy20osti/75698.pdf)
ax1.plot(
    wind_table,
    df["Aero Power Coefficient [-]"].to_list(),
    label="IEA15MW",
    linewidth=4,
    color='tab:orange',
    linestyle = "solid",
    alpha = alpha,
)
ax1.plot(
    wind_speeds,
    mit_Cp_rotor_umm,
    label="MITRotor UMM Rotor-Averaged",
    linewidth=4,
    color='tab:blue',
    linestyle = "solid",
    alpha = alpha,
)
ax1.plot(
    wind_speeds,
    mit_Cp_annulus_umm,
    label="MITRotor UMM Annulus-Averaged",
    linewidth=4,
    color='tab:green',
    linestyle = "solid",
    alpha = alpha,
)
# ax1.plot(
#     wind_speeds,
#     mit_Cp_rotor_madsen,
#     label="MITRotor Madsen Rotor-Averaged",
#     linewidth=4,
#     color='tab:purple',
#     linestyle = "solid",
#     alpha = alpha,
# )
# ax1.plot(
#     wind_speeds,
#     mit_Cp_annulus_madsen,
#     label="MITRotor Madsen Annulus-Averaged",
#     linewidth=4,
#     color='tab:red',
#     linestyle = "solid",
#     alpha = alpha,
# )
ax1.plot(
    wind_speeds,
    floris_Cp_rotor_umm,
    label="FLORIS-MITRotor UMM Rotor-Averaged",
    linewidth=4,
    color='tab:blue',
    linestyle = "dashed",
    alpha = alpha,
)
ax1.plot(
    wind_speeds,
    floris_Cp_annulus_umm,
    label="FLORIS-MITRotor UMM Annulus-Averaged",
    linewidth=4,
    color='tab:green',
    linestyle = "dashed",
    alpha = alpha,
)
ax1.plot(
    wind_speeds,
    floris_Cp_rotor_madsen,
    label="FLORIS-MITRotor Madsen Rotor-Averaged",
    linewidth=4,
    color='tab:purple',
    linestyle = "dashed",
    alpha = alpha,
)
ax1.plot(
    wind_speeds,
    floris_Cp_annulus_madsen,
    label="FLORIS-MITRotor Madsen Annulus-Averaged",
    linewidth=4,
    color='tab:red',
    linestyle = "dashed",
    alpha = alpha,
)

ax1.set_xlabel("Wind Speed [m/s]", size = 16)
ax1.set_ylabel("$C_P$", size = 16)
ax1.tick_params(labelsize = 14)
ax1.set_title("$C_P$", size = 18)
ax1.legend(
    fontsize=14,
    loc="upper right",
)

plt.savefig(figdir / "example_5_IEA15mw_CT_CP.png", dpi=300)
