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

fig_fn = FIGDIR / "example_009_nonlinear_pressure.png"

dps = np.linspace(0.4, 1, 20)
xs = np.linspace(0, 15, 100)


def main():
    solver = NonlinearADPressureField(dx=0.1, iterations=4)

    dp_mesh, x_mesh = np.meshgrid(dps, xs, indexing="ij")
    ps_interp = solver.get_pressure(dp_mesh, x_mesh)
    levels = np.arange(-0.3, 0.001, 0.025)

    plt.figure(figsize=(6, 3))
    CF = plt.contourf(xs, dps, ps_interp, levels=levels, cmap="viridis_r")
    CS = plt.contour(xs, dps, ps_interp, levels=levels, colors="k")
    plt.clabel(CS, inline=True, fontsize=10, fmt="%1.3f")

    cbar = plt.colorbar(CF)
    cbar.set_label(label="$p^g$ [-]")
    plt.xlabel("$x_0$ [R]")
    plt.ylabel("$\Delta p$ [-]")

    plt.savefig(fig_fn, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
