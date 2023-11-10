from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import polars as pl
import matplotlib.pyplot as plt
from tqdm import tqdm
from MITRotor.PressureSolver.ADPressureField import NonlinearADPressureField


# Use Latex Fonts
plt.rcParams.update({"text.usetex": True, "font.family": "serif"})


FIGDIR = Path("fig")
FIGDIR.mkdir(exist_ok=True, parents=True)

fig_fn = FIGDIR / "example_010_combined_pressure.png"

dps = np.linspace(0.3, 1, 20)
xs = np.linspace(0, 15, 100)

NITER = 6


def main():
    dp_mesh, x_mesh = np.meshgrid(dps, xs, indexing="ij")

    fields = []
    for relax in tqdm(np.arange(0.0, 0.3, 0.05)):
        solver = NonlinearADPressureField(dx=0.5, iterations=NITER, relax=relax)
        fields.append(solver.get_pressure(dp_mesh, x_mesh))

    fields = np.array(fields)
    fields[np.isinf(fields)] = 0
    combined_field = np.nanmin(fields, axis=0)

    levels = np.arange(-0.3, 0.001, 0.025)

    plt.figure(figsize=(6, 3))
    CF = plt.contourf(xs, dps, combined_field, levels=levels, cmap="viridis_r")
    CS = plt.contour(xs, dps, combined_field, levels=levels, colors="k")
    plt.clabel(CS, inline=True, fontsize=10, fmt="%1.3f")

    cbar = plt.colorbar(CF)
    cbar.set_label(label="$p^g$ [-]")
    plt.xlabel("$x_0$ [R]")
    plt.ylabel("$\Delta p$ [-]")

    plt.savefig(fig_fn, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
