import os
import numpy as np
import polars as pl
from pathlib import Path
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning) # JUST FOR NOW

# MITRotor / UMM Imports
from MITRotor.FlorisInterface.FlorisInterface import default_bem_factory, default_pitch_interp, default_tsr_interp
from MITRotor.ReferenceTurbines import IEA15MW
from MITRotor.Momentum import UnifiedMomentum
from MITRotor.Geometry import BEMGeometry
from MITRotor.TipLoss import NoTipLoss
from MITRotor.BEMSolver import BEM

import cProfile
import pstats

module_dir = os.path.dirname(__file__)  # examples/
csv_file = os.path.join(module_dir, "..", "MITRotor", "FlorisInterface", "IEA_15mw_rotor.csv")
df = pl.read_csv(csv_file)

wind_table = df["Wind [m/s]"].to_numpy()
wind_speeds = np.linspace(5, 20, 20)
# wind_speeds = np.array([5.0])
wind_dirs = np.full_like(wind_speeds, 270.0)
turbulence_intensity = np.zeros_like(wind_speeds)

# ------- plot pitch and tsr control curves for IEA15MW from figure 2 (https://docs.nrel.gov/docs/fy22osti/82134.pdf) --------
pitch_interp = default_pitch_interp()
tsr_interp   = default_tsr_interp()

# -------- plot CT and CP values against one another and against IEA15MW from figure 3.1-C (https://docs.nrel.gov/docs/fy20osti/75698.pdf) -------
# solve UMM-BEM though MITRotor - rotor averaged
pitches = np.deg2rad(pitch_interp(wind_speeds))
tsrs = tsr_interp(wind_speeds)
yaws, tilts = np.zeros_like(pitches), np.zeros_like(pitches)

bem_rotor_umm = default_bem_factory()
mit_rotor_umm_start = time.time()
mit_sols_rotor_umm = bem_rotor_umm(pitch=pitches, tsr=tsrs, yaw=yaws, tilt=tilts)
mit_rotor_umm_end = time.time()
mit_Ct_rotor_umm = mit_sols_rotor_umm.Ct()
mit_Cp_rotor_umm = mit_sols_rotor_umm.Cp()
print("MITRotor UMM-BEM Rotor-Averaged: " + str(mit_rotor_umm_end - mit_rotor_umm_start) + " seconds")

# solve UMM-BEM though MITRotor - annulus averaged
bem_annulus_umm = BEM(
        rotor=IEA15MW(),
        momentum_model=UnifiedMomentum(averaging="annulus"),
        geometry=BEMGeometry(Nr=10, Ntheta=20),
        tiploss_model=NoTipLoss(),
    )
mit_annulus_umm_start = time.time()

# profiler = cProfile.Profile()
# profiler.enable()
mit_sols_annulus_umm = bem_annulus_umm(pitch=pitches, tsr=tsrs, yaw=yaws, tilt=tilts)
# profiler.disable()

mit_annulus_umm_end = time.time()
# mit_Ct_annulus_umm = mit_sols_annulus_umm.Ct()
# mit_Cp_annulus_umm = mit_sols_annulus_umm.Cp()
print("MITRotor UMM-BEM Annulus-Averaged: " + str(mit_annulus_umm_end - mit_annulus_umm_start) + " seconds")
# stats = pstats.Stats(profiler)
# stats.sort_stats("cumtime").print_stats(20)