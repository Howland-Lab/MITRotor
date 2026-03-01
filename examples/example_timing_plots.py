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
plt.figure(figsize=(6, 4))

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
        linewidth = 2,
    )
    # plt.yscale("log")

plt.xlabel("Number of Wind Speeds Run\n(evenly-spaced between 5-20 m/s)")
plt.ylabel("Runtime (seconds)")
plt.title("Runtime vs Number of Wind Speeds")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(figdir / "example_timings.png", dpi=300)

#------------------------------------
# Value plot
#------------------------------------
plt.figure(figsize=(6, 4))

value_df["label"] = value_df.apply(make_label, axis=1)

for label in label_order:
    group = value_df[value_df["label"] == label]
    group = group.sort_values("wind_speed")
    is_vectorized = group["vectorized"].iloc[0]
    linestyle = "dashed" if is_vectorized else "solid"
    zorder = 2 if is_vectorized else 1
    plt.plot(
        group["wind_speed"],
        group["power"],
        linewidth = 3,
        label=label,
        linestyle = linestyle,
        zorder = zorder,
    )

plt.xlabel("Wind Speed [m/s]")
plt.ylabel("Coefficent of Power ($C_P$)")
plt.title("$C_P$ vs Wind Speed")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(figdir / "example_values.png", dpi=300)