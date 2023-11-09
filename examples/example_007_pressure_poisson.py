from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from MITRotor.PressureSolver.ADPressureField import NonlinearADPressureField

# Use Latex Fonts
plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

FIGDIR = Path("fig")
FIGDIR.mkdir(exist_ok=True, parents=True)

MAX_ITER = 4

fig_fn = FIGDIR / f"example_007_pressure_poisson_{MAX_ITER}.png"

if __name__ == "__main__":
    xs = np.linspace(-3, 10, 200)
    dps = np.arange(0, 1, 0.1)
    solver = NonlinearADPressureField(iterations=MAX_ITER)

    plt.figure()
    for dp in dps:
        p = solver.get_pressure(dp, xs)
        plt.plot(xs, p, c=plt.cm.viridis(dp))

    plt.grid()

    plt.ylim(-0.6, 0.6)

    plt.xlabel("x")
    plt.ylabel(r"$p^{NL}$")
    plt.savefig(fig_fn, dpi=300, bbox_inches="tight")
