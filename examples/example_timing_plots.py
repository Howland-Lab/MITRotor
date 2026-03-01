import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---- Load data ----
figdir = Path("fig")
time_df = pd.read_csv(Path("cache")/ "timing_results.csv")
value_df = pd.read_csv(Path("cache")/ "value_results.csv")

#------------------------------------
# Timing plot
#------------------------------------
df_avg = (
    time_df.groupby(["n_wind_speeds", "model", "vectorized"], as_index=False)
      .agg(
          runtime_mean=("runtime_seconds", "mean"),
          runtime_std=("runtime_seconds", "std"),
      )
)

# ---- Create label column for plotting ----
def make_label(row):
    base = {
        "rotor_umm": "UMM Rotor-Averaged",
        "annulus_lut": "UMM LUT Annulus-Averaged",
    }[row["model"]]

    if row["vectorized"]:
        return base + " (Vectorized)"
    else:
        return base

df_avg["label"] = df_avg.apply(make_label, axis=1)

label_order = [
    "UMM Rotor-Averaged",
    "UMM LUT Annulus-Averaged",
    "UMM Rotor-Averaged (Vectorized)",
    "UMM LUT Annulus-Averaged (Vectorized)",
]

# ---- Plot ----
plt.figure(figsize=(8, 6))

for label in label_order:
    group = df_avg[df_avg["label"] == label]
    group = group.sort_values("n_wind_speeds")

    plt.errorbar(
        group["n_wind_speeds"],
        group["runtime_mean"],
        yerr=group["runtime_std"],
        marker="o",
        capsize=3,
        label=label,
    )

plt.xlabel("Number of Wind Speeds")
plt.ylabel("Runtime (seconds)")
plt.title("Runtime vs Number of Wind Speeds")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(figdir / "example_timings.png", dpi=300)

#------------------------------------
# Value plot
#------------------------------------
plt.figure(figsize=(8, 6))

df_avg = (
    value_df.groupby(["wind_speed", "model", "vectorized"], as_index=False)
      .agg(
          power_mean=("power", "mean"),
          power_std=("power", "std"),
      )
)
df_avg["label"] = df_avg.apply(make_label, axis=1)

for label in label_order:
    group = df_avg[df_avg["label"] == label]
    group = group.sort_values("wind_speed")

    plt.errorbar(
        group["wind_speed"],
        group["power_mean"],
        yerr=group["power_std"],
        marker="o",
        capsize=3,
        label=label,
    )

plt.xlabel("Wind Speed")
plt.ylabel("Coefficent of Power $C_P$")
plt.title("$C_P$ vs Wind Speed")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(figdir / "example_values.png", dpi=300)