import re
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.lines import Line2D

def get_vals(logfile):

    # Parse lines that contain exactly 5 numbers:
    # wind_speed, yaw, blade_pitch, tsr, avg_runtime
    rows = []
    num_re = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

    with open(logfile, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            nums = re.findall(num_re, line)
            if len(nums) == 5:
                rows.append(tuple(map(float, nums)))

    if not rows:
        raise RuntimeError("No 5-number data lines found in crash.log")

    wind, yaw, pitch, tsr, runtime = zip(*rows)

    wind = np.array(wind, dtype=float)
    yaw = np.array(yaw, dtype=float)
    pitch = np.array(pitch, dtype=float)
    tsr = np.array(tsr, dtype=float)
    runtime = np.array(runtime, dtype=float)

    # ---- filter unrealistic pitch values ----
    # Set these for your units:
    # degrees example: -5 to 45
    # radians example: -0.1 to 0.8
    PITCH_MIN, PITCH_MAX = -5.0, 45.0

    m = (
        np.isfinite(pitch) & np.isfinite(tsr) & np.isfinite(runtime) &
        (pitch >= PITCH_MIN) & (pitch <= PITCH_MAX) & (wind >= 3)
    )
    wind = wind[m]
    pitch = pitch[m]
    yaw = yaw[m]
    tsr = tsr[m]
    runtime = runtime[m]
    return wind, pitch, yaw, tsr, runtime

logfile1 = "crash_ps_3_yaw.log"
wind1, pitch1, yaw1, tsr1, runtime1 = get_vals(logfile1)

logfile2 = "crash_ps_0_yaw.log"
wind2, pitch2, yaw2, tsr2, runtime2 = get_vals(logfile2)

# 1) time vs wind speed, colored by yaw
plt.figure(figsize=(6, 4))
sc = plt.scatter(wind1, runtime1, c=yaw1, cmap="viridis", alpha = 0.7, s=35)
plt.xlabel("Wind speed")
plt.ylabel("Average runtime")
plt.colorbar(sc, label="Yaw")
plt.tight_layout()
plt.savefig("time_vs_windspeed_colored_by_yaw.png", dpi=200)
plt.close()

# 2) time vs tsr, colored by wind speed
plt.figure(figsize=(6, 4))
sc = plt.scatter(tsr1, runtime1, c=yaw1, cmap="plasma", alpha = 0.7, s=35)
plt.xlabel("TSR")
plt.ylabel("Average runtime")
plt.colorbar(sc, label="Yaw")
plt.tight_layout()
plt.savefig("time_vs_tsr_colored_by_yaw.png", dpi=200)
plt.close()

# 3) time vs wind speed, colored by tsr
plt.figure(figsize=(6, 4))
sc = plt.scatter(wind1, runtime1, c=tsr1, cmap="plasma", alpha = 0.7, s=35)
plt.xlabel("Windspeed")
plt.ylabel("Average runtime")
plt.colorbar(sc, label="TSR")
plt.tight_layout()
plt.savefig("time_vs_wind_colored_by_tsr.png", dpi=200)
plt.close()

# 4) time vs tsr colored by pitch

# shared color normalization across both datasets
pitch1_deg = np.rad2deg(pitch1)
pitch2_deg = np.rad2deg(pitch2)
vmin = min(np.min(pitch1_deg), np.min(pitch2_deg))
vmax = max(np.max(pitch1_deg), np.max(pitch2_deg))
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
cmap = "viridis_r"

fig, ax = plt.subplots(figsize=(6, 4))

sc1 = ax.scatter(tsr1, runtime1,c = pitch1_deg,  cmap=cmap, norm=norm,
                 alpha=0.7, s=35, marker="o")
sc2 = ax.scatter(tsr2, runtime2, c = pitch2_deg, cmap=cmap, norm=norm,
                 alpha=0.7, s=35, marker="v")

ax.set_xlabel("TSR")
ax.set_ylabel("Average BEM Runtime")

# one shared colorbar
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])  # compatibility
cbar = fig.colorbar(sm, ax=ax, label="Pitch (deg)")

# shape legend (independent of colormap)
shape_handles = [
    Line2D([0], [0], marker='o', linestyle='None', color='k',
           markerfacecolor='gray', label='PS_Mode = 3'),
    Line2D([0], [0], marker='v', linestyle='None', color='k',
           markerfacecolor='gray', label='PS_Mode = 0'),
]
ax.legend(handles=shape_handles, title="Marker shape", loc="best")

fig.tight_layout()
fig.savefig("time_vs_tsr_colored_by_pitch.png", dpi=200)
plt.close(fig)