import numpy as np
import polars as pl
import pytest
from pathlib import Path
import matplotlib.pyplot as plt
from MITRotor.Momentum import UnifiedMomentumLUT, UnifiedMomentum
from MITRotor import BEM, IEA15MW, BEMGeometry
import time

cache_dir = Path("cache")
cache_dir.mkdir(exist_ok=True, parents=True)
cache_file = cache_dir / "lut.csv"

figdir = Path("fig")
figdir.mkdir(exist_ok=True, parents=True)

#------------------------------------------------------------
# Example for UMM vs LUT UMM Momentum Model
#------------------------------------------------------------
print("Making Cache!")
start_cache = time.perf_counter()
ref_model = UnifiedMomentum()
lut_model = UnifiedMomentumLUT(
    cache_fn=cache_file,
    regenerate=True,
    LUT_Cts=np.linspace(-0.5,1.5,40),
    LUT_yaws=np.linspace(0.0,40.1,40),
)
end_cache = time.perf_counter()
print(f"Cache created in {end_cache - start_cache} seconds!")

Cts = np.linspace(-0.5, 1.5, 30)
yaws = np.linspace(0.0, 40.0, 30)
rad_yaws = np.deg2rad(yaws)
an_ref = np.zeros((Cts.size, yaws.size))
an_lut = np.zeros_like(an_ref)
tilt = np.deg2rad(5)
start_ref = time.perf_counter()
for (i, Ct) in enumerate(Cts):
    for (j, rad_yaw) in enumerate(rad_yaws):
        an_ref[i, j] = ref_model.compute_induction(Ct, rad_yaw, tilt=tilt)
end_ref = time.perf_counter()
print(f"Ran reference UMM in {end_ref - start_ref} seconds!")

start_lut = time.perf_counter()
for (i, Ct) in enumerate(Cts):
    for (j, rad_yaw) in enumerate(rad_yaws):
        an_lut[i, j] = lut_model.compute_induction(Ct, rad_yaw, tilt=tilt)
end_lut = time.perf_counter()
print(f"Ran LUT UMM in {end_lut - start_lut} seconds!")

an_lut_masked = np.ma.masked_invalid(an_lut)
an_ref_masked = np.ma.masked_invalid(an_ref)
diff = an_lut - an_ref
diff_masked = np.ma.masked_invalid(diff)

# Presentation-friendly typography
plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
})

fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
for ax in axes:
    ax.set_box_aspect(1)

YAW, CT = np.meshgrid(yaws, Cts)

# --- Shared levels for reference and LUT ---
nlevels = 20
vmin = min(np.nanmin(an_ref), np.nanmin(an_lut))
vmax = max(np.nanmax(an_ref), np.nanmax(an_lut))
levels = np.linspace(vmin, vmax, nlevels)

cmap = plt.cm.viridis.copy()
cmap.set_bad(color="lightgray")

# Panel 1 — Reference
cf0 = axes[0].contourf(YAW, CT, an_ref_masked, levels=levels, cmap=cmap)
axes[0].set_title("Reference Model\nInduction Factor ($a_n$)")
axes[0].set_xlabel("Yaw Angle (deg)")
axes[0].set_ylabel("Thrust Coefficient ($C_t$)")

# Panel 2 — LUT
cf1 = axes[1].contourf(YAW, CT, an_lut_masked, levels=levels, cmap=cmap)
axes[1].set_title("LUT Model\nInduction Factor ($a_n$)")
axes[1].set_xlabel("Yaw Angle (deg)")
axes[1].set_ylabel("Thrust Coefficient ($C_t$)")

# Shared colorbar for first two
cbar = fig.colorbar(cf0, ax=axes[:2], shrink=0.9)
cbar.set_label("Induction Factor ($a_n$)")

# --- Difference plot ---
absmax = np.nanmax(np.abs(diff))
diff_levels = np.linspace(-absmax, absmax, nlevels)

cf2 = axes[2].contourf(YAW, CT, diff_masked, levels=diff_levels, cmap=cmap)
axes[2].set_title("Difference\n(LUT − Reference)")
axes[2].set_xlabel("Yaw Angle (deg)")
axes[2].set_ylabel("Thrust Coefficient ($C_t$)")

cbar_diff = fig.colorbar(cf2, ax=axes[2], shrink=0.9)
cbar_diff.set_label("Δ $a_n$")

plt.savefig(figdir / "example_5_an_LUT_compare.png", dpi=300)

#------------------------------------------------------------
# Example for UMM vs LUT UMM BEM Model
#------------------------------------------------------------

rotor = IEA15MW()
Nr, Ntheta = 10, 20
ref_bem = BEM(rotor=rotor, momentum_model=ref_model, geometry = BEMGeometry(Nr=Nr, Ntheta=Ntheta))
lut_bem = BEM(rotor=rotor, momentum_model=lut_model, geometry = BEMGeometry(Nr=Nr, Ntheta=Ntheta))

tsrs = np.linspace(2.5, 12.5, 12)
pitches = np.linspace(0.0, 15.0, 12)
rad_pitches = np.deg2rad(pitches)
cp_ref = np.zeros((tsrs.size, pitches.size))
cp_lut = np.zeros_like(cp_ref)

start_ref = time.perf_counter()
for (i, tsr) in enumerate(tsrs):
    for (j, rad_pitch) in enumerate(rad_pitches):
        cp_ref[i, j] = ref_bem(rad_pitch, tsr, yaw = 0.0, tilt = 0.0).Cp()
end_ref = time.perf_counter()
print(f"Ran reference BEM in {end_ref - start_ref} seconds!")

start_lut = time.perf_counter()
for (i, tsr) in enumerate(tsrs):
    for (j, rad_pitch) in enumerate(rad_pitches):
        cp_lut[i, j] = lut_bem(rad_pitch, tsr, yaw = 0.0, tilt = 0.0).Cp()
end_lut = time.perf_counter()
print(f"Ran LUT BEM in {end_lut - start_lut} seconds!")

cp_lut_masked = np.ma.masked_invalid(cp_lut)
cp_ref_masked = np.ma.masked_invalid(cp_ref)
diff = cp_lut - cp_ref
diff_masked = np.ma.masked_invalid(diff)

fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
for ax in axes:
    ax.set_box_aspect(1)

PITCH, TSR = np.meshgrid(pitches, tsrs)

# --- Shared levels for reference and LUT ---
nlevels = 20
vmin = min(np.nanmin(cp_ref), np.nanmin(cp_lut))
vmax = max(np.nanmax(cp_ref), np.nanmax(cp_lut))
levels = np.linspace(vmin, vmax, nlevels)

cmap = plt.cm.viridis.copy()
cmap.set_bad(color="lightgray")

# Panel 1 — Reference
cf0 = axes[0].contourf(PITCH, TSR, cp_ref_masked, levels=levels, cmap=cmap)
axes[0].set_title("Reference Model\nCoefficent of Power ($C_P$)")
axes[0].set_xlabel("Pitch Angle (deg)")
axes[0].set_ylabel("TSR")

# Panel 2 — LUT
cf1 = axes[1].contourf(PITCH, TSR, cp_lut_masked, levels=levels, cmap=cmap)
axes[1].set_title("LUT Model\nCoefficent of Power ($C_P$)")
axes[1].set_xlabel("Pitch Angle (deg)")
axes[1].set_ylabel("TSR")

# Shared colorbar for first two
cbar = fig.colorbar(cf0, ax=axes[:2], shrink=0.9)
cbar.set_label("Power Coefficent ($C_P$)")

# --- Difference plot ---
absmax = np.nanmax(np.abs(diff))
diff_levels = np.linspace(-absmax, absmax, nlevels)

cf2 = axes[2].contourf(PITCH, TSR, diff_masked, levels=diff_levels, cmap=cmap)
axes[2].set_title("Difference\n(LUT − Reference)")
axes[2].set_xlabel("Pitch Angle (deg)")
axes[2].set_ylabel("TSR")

cbar_diff = fig.colorbar(cf2, ax=axes[2], shrink=0.9)
cbar_diff.set_label("Δ $C_P$")

plt.savefig(figdir / "example_5_cp_LUT_compare.png", dpi=300)

