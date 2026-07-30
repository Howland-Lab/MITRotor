import numpy as np
from pathlib import Path
from MITRotor.Momentum import UnifiedMomentumLUT, UnifiedMomentum
from MITRotor.TipLoss import NoTipLoss
from MITRotor import BEM, IEA15MW, BEMGeometry
import pandas as pd
import time
import matplotlib.pyplot as plt

# Floris imports
from floris import FlorisModel, TimeSeries
from MITRotor.FlorisInterface.FlorisInterface import MITRotorTurbine, default_bem_factory

figdir = Path("fig")
floris_air_density = 1.225

bem_rotor_umm = default_bem_factory()
bem_rotor_umm_LUT = BEM(
    rotor=IEA15MW(),
    momentum_model=UnifiedMomentumLUT(averaging="rotor", cache_fn = Path("cache")/ "rotor_lut.csv"),
    geometry=BEMGeometry(Nr=10, Ntheta=20),
    tiploss_model=NoTipLoss(),
)
bem_annulus_umm_LUT = BEM(
    rotor=IEA15MW(),
    momentum_model=UnifiedMomentumLUT(averaging="annulus", cache_fn = Path("cache")/ "annulus_lut.csv"),
    geometry=BEMGeometry(Nr=10, Ntheta=20),
    tiploss_model=NoTipLoss(),
)
rotor_area = np.pi * bem_rotor_umm.rotor.R**2 

bem_rotor_times = []
bem_rotor_LUT_times = []
bem_annulus_LUT_times = []
wind_speeds_all = []
bem_rotor_values = []
bem_rotor_LUT_values = []
bem_annulus_LUT_values = []
ns = [5 * i for i in range(1, 21)]
for n in ns:
    print(f"{n} wind speeds")
    wind_speeds = np.linspace(5, 20, n)
    wind_dirs = np.full_like(wind_speeds, 270.0)
    turbulence_intensity = np.zeros_like(wind_speeds)
    wind_speeds_all.extend(
        np.squeeze(wind_speeds)
    )

    time_series = TimeSeries(
        wind_speeds=wind_speeds,
        wind_directions=wind_dirs,
        turbulence_intensities=turbulence_intensity,
    )
    # solve FLORIS  with UMM-BEM though MITRotor - rotor averaged
    fmodel_rotor_umm = FlorisModel("defaults")
    fmodel_rotor_umm.set(layout_x = [0.0], layout_y = [0.0], wind_data = time_series)
    floris_rotor_model = MITRotorTurbine(bem_model = bem_rotor_umm)
    fmodel_rotor_umm.set_operation_model(floris_rotor_model) # default bem_model uses rotor-averaging
    floris_rotor_umm_start = time.time()
    fmodel_rotor_umm.run()
    floris_rotor_umm_end = time.time()
    dt_rotor = floris_rotor_umm_end - floris_rotor_umm_start
    print("FLORIS UMM-BEM Rotor-Averaged: " + str(dt_rotor) + " seconds")
    bem_rotor_times.append(dt_rotor)
    cp_calc_denominator = (0.5 * floris_air_density * rotor_area * (wind_speeds)**3 * floris_rotor_model.eff_ratio)
    floris_Cp_rotor_umm =  np.squeeze(fmodel_rotor_umm.get_turbine_powers()) / cp_calc_denominator
    bem_rotor_values.extend(np.squeeze(floris_Cp_rotor_umm))

    # solve FLORIS  with UMM-BEM with LUT though MITRotor - rotor averaged
    fmodel_rotor_umm_LUT = FlorisModel("defaults")
    fmodel_rotor_umm_LUT.set(layout_x = [0.0], layout_y = [0.0], wind_data = time_series)
    fmodel_rotor_umm_LUT.set_operation_model(MITRotorTurbine(bem_model = bem_rotor_umm_LUT))
    floris_rotor_umm_LUT_start = time.time()
    fmodel_rotor_umm_LUT.run()
    floris_rotor_umm_LUT_end = time.time()
    dt_rotor_LUT = floris_rotor_umm_LUT_end - floris_rotor_umm_LUT_start
    print("FLORIS UMM-BEM LUT Rotor-Averaged: " + str(dt_rotor_LUT) + " seconds")
    bem_rotor_LUT_times.append(dt_rotor_LUT)
    floris_Cp_rotor_umm_LUT =  np.squeeze(fmodel_rotor_umm_LUT.get_turbine_powers()) / cp_calc_denominator
    bem_rotor_LUT_values.extend(
        np.squeeze(floris_Cp_rotor_umm_LUT)
    )

    # solve FLORIS  with UMM-BEM with LUT though MITRotor - annulus averaged
    fmodel_annulus_umm_LUT = FlorisModel("defaults")
    fmodel_annulus_umm_LUT.set(layout_x = [0.0], layout_y = [0.0], wind_data = time_series)
    fmodel_annulus_umm_LUT.set_operation_model(MITRotorTurbine(bem_model = bem_annulus_umm_LUT))
    floris_annulus_umm_LUT_start = time.time()
    fmodel_annulus_umm_LUT.run()
    floris_annulus_umm_LUT_end = time.time()
    dt_annulus_LUT = floris_annulus_umm_LUT_end - floris_annulus_umm_LUT_start
    print("FLORIS UMM-BEM LUT Annulus-Averaged: " + str(dt_annulus_LUT) + " seconds")
    bem_annulus_LUT_times.append(dt_annulus_LUT)
    floris_Cp_annulus_umm_LUT =  np.squeeze(fmodel_annulus_umm_LUT.get_turbine_powers()) / cp_calc_denominator
    bem_annulus_LUT_values.extend(
        np.squeeze(floris_Cp_annulus_umm_LUT)
    )

# make timing CSV
vectorized = True
rows = []

for n, dt in zip(ns, bem_rotor_times):
    rows.append({
        "n_wind_speeds": n,
        "runtime_seconds": dt,
        "model": "rotor_umm",
        "vectorized": vectorized
    })

for n, dt in zip(ns, bem_rotor_LUT_times):
    rows.append({
        "n_wind_speeds": n,
        "runtime_seconds": dt,
        "model": "rotor_lut",
        "vectorized": vectorized
    })

for n, dt in zip(ns, bem_annulus_LUT_times):
    rows.append({
        "n_wind_speeds": n,
        "runtime_seconds": dt,
        "model": "annulus_lut",
        "vectorized": vectorized
    })

df = pd.DataFrame(rows)

csv_path = Path("cache")/ "timing_results.csv"

if csv_path.exists():
    df.to_csv(csv_path, mode="a", header=False, index=False)
else:
    df.to_csv(csv_path, index=False)

# ------------------------------------------------------------------
# Plot timings
# ------------------------------------------------------------------
plt.figure()
plt.plot(ns, bem_rotor_times, "o-", label="Rotor-averaged UMM")
plt.plot(ns,bem_rotor_LUT_times, "v-",label="Rotor-averaged UMM with LUT")
plt.plot(ns,bem_annulus_LUT_times, "s-",label="Annulus-averaged UMM with LUT")
plt.xlabel("Number of wind speeds")
plt.ylabel("Runtime (s)")
plt.yscale("log")
plt.title("FLORIS Runtime Comparison")
plt.legend()
plt.savefig(figdir / "example_7_floris_runtimes.png", dpi=300)

# make values CSV
rows = []
# Rotor
for wind, val in zip(wind_speeds_all, bem_rotor_values):
    rows.append({
        "wind_speed": wind,
        "power": val,
        "model": "rotor_umm",
        "vectorized": vectorized
    })

# Rotor LUT
for wind, val in zip(wind_speeds_all, bem_rotor_LUT_values):
    rows.append({
        "wind_speed": wind,
        "power": val,
        "model": "rotor_lut",
        "vectorized": vectorized
    })

# Annulus LUT
for wind, val in zip(wind_speeds_all, bem_annulus_LUT_values):
    rows.append({
        "wind_speed": wind,
        "power": val,
        "model": "annulus_lut",
        "vectorized": vectorized
    })

df = pd.DataFrame(rows)

csv_path = Path("cache") / "value_results.csv"

if csv_path.exists():
    df.to_csv(csv_path, mode="a", header=False, index=False)
else:
    df.to_csv(csv_path, index=False)

import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Plot results
# ------------------------------------------------------------------
# Sort by wind speed for plotting
sort_idx = np.argsort(wind_speeds_all)

wind_speeds_plot = np.array(wind_speeds_all)[sort_idx]
bem_rotor_plot = np.array(bem_rotor_values)[sort_idx]
bem_rotor_LUT_plot = np.array(bem_rotor_LUT_values)[sort_idx]
bem_annulus_LUT_plot = np.array(bem_annulus_LUT_values)[sort_idx]

plt.figure()

plt.plot(wind_speeds_plot, bem_rotor_plot, label="Rotor-averaged UMM", linestyle="-", linewidth=3)
plt.plot(wind_speeds_plot, bem_rotor_LUT_plot, label="Rotor-averaged UMM with LUT", linestyle="--", linewidth=3)
plt.plot(wind_speeds_plot, bem_annulus_LUT_plot, label="Annulus-averaged UMM with LUT", linestyle=":", linewidth=3)

plt.xlabel("Wind speed (m/s)")
plt.ylabel(r"$C_P$")
plt.title("Power Coefficient Comparison")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(figdir / "example_7_floris_cp_vals.png", dpi=300)
