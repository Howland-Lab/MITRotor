import numpy as np
from pathlib import Path
from MITRotor.Momentum import UnifiedMomentumLUT, UnifiedMomentum
from MITRotor.TipLoss import NoTipLoss
from MITRotor import BEM, IEA15MW, BEMGeometry
import pandas as pd
import time

# Floris imports
from floris import FlorisModel, TimeSeries
from MITRotor.FlorisInterface.FlorisInterface import MITRotorTurbine, default_bem_factory

bem_rotor_umm = MITRotorTurbine()
bem_annulus_umm_LUT = BEM(
    rotor=IEA15MW(),
    momentum_model=UnifiedMomentumLUT(averaging="annulus", cache_fn = Path("cache")/ "lut.csv"),
    geometry=BEMGeometry(Nr=10, Ntheta=20),
    tiploss_model=NoTipLoss(),
)

bem_rotor_times = []
bem_annulus_LUT_times = []
ns = [5 * i for i in range(1, 20)]
for n in ns:
    print(f"{n} wind speeds")
    wind_speeds = np.linspace(5, 20, n)
    wind_dirs = np.full_like(wind_speeds, 270.0)
    turbulence_intensity = np.zeros_like(wind_speeds)

    time_series = TimeSeries(
        wind_speeds=wind_speeds,
        wind_directions=wind_dirs,
        turbulence_intensities=turbulence_intensity,
    )

    fmodel_rotor_umm = FlorisModel("defaults")
    fmodel_rotor_umm.set(layout_x = [0.0], layout_y = [0.0], wind_data = time_series)
    fmodel_rotor_umm.set_operation_model(bem_rotor_umm) # default bem_model uses rotor-averaging
    floris_rotor_umm_start = time.time()
    fmodel_rotor_umm.run()
    floris_rotor_umm_end = time.time()
    dt_rotor = floris_rotor_umm_end - floris_rotor_umm_start
    print("FLORIS UMM-BEM Rotor-Averaged: " + str(dt_rotor) + " seconds")
    bem_rotor_times.append(dt_rotor)

    # solve FLORIS  with UMM-BEM with LUT though MITRotor - annulus averaged
    fmodel_annulus_umm_LUT = FlorisModel("defaults")
    fmodel_annulus_umm_LUT.set(layout_x = [0.0], layout_y = [0.0], wind_data = time_series)
    fmodel_annulus_umm_LUT.set_operation_model(MITRotorTurbine(bem_model = bem_annulus_umm_LUT)) # default bem_model uses rotor-averaging
    floris_annulus_umm_LUT_start = time.time()
    fmodel_annulus_umm_LUT.run()
    floris_annulus_umm_LUT_end = time.time()
    dt_annulus_LUT = floris_annulus_umm_LUT_end - floris_annulus_umm_LUT_start
    print("FLORIS UMM-BEM LUT Annulus-Averaged: " + str(dt_annulus_LUT) + " seconds")
    bem_annulus_LUT_times.append(dt_annulus_LUT)


rows = []

for n, dt in zip(ns, bem_rotor_times):
    rows.append({
        "n_wind_speeds": n,
        "runtime_seconds": dt,
        "model": "rotor_umm",
        "vectorized": False
    })

for n, dt in zip(ns, bem_annulus_LUT_times):
    rows.append({
        "n_wind_speeds": n,
        "runtime_seconds": dt,
        "model": "annulus_lut",
        "vectorized": False
    })

df = pd.DataFrame(rows)

csv_path = Path("cache")/ "timing_results.csv"

if csv_path.exists():
    df.to_csv(csv_path, mode="a", header=False, index=False)
else:
    df.to_csv(csv_path, index=False)