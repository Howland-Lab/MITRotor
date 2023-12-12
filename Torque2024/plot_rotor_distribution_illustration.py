from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from MITRotor.BEM import BEM
from MITRotor.Geometry import BEMGeometry
from MITRotor.ReferenceTurbines import IEA15MW
from Torque2024.cache import cache_pickle

from Torque2024.shared_generation import find_optimal_setpoint
FIGDIR = Path("fig")
FIGDIR.mkdir(exist_ok=True, parents=True)




@cache_pickle("data/rotor_distribution_illustration.pkl")
def generate(regenerate=False):
    rotor = IEA15MW()
    bem = BEM(rotor, BEMGeometry(10, 200))

    sol_opt = find_optimal_setpoint(bem)
    print("optimal θ_p [deg]: ", np.rad2deg(sol_opt.pitch))
    print("optimal λ: ", sol_opt.tsr)

    sol = bem(sol_opt.pitch, sol_opt.tsr, np.deg2rad(40))

    mu_mesh = bem.geometry.mu_mesh
    theta_mesh = bem.geometry.theta_mesh

    X, Y = mu_mesh * np.cos(theta_mesh), mu_mesh * np.sin(theta_mesh)

    Z = sol.solidity(grid="radial")[..., None] * sol.W(grid="full") ** 2 * sol.Cax(grid="full")

    return X, Y, Z


def plot(X, Y, Z):
    plt.figure(figsize=(7, 7))

    thres = 0.6

    plt.pcolormesh(
        X,
        Y,
        Z,
        edgecolors="face",
        antialiased=True,
        cmap="copper",
        vmin=thres,
        vmax=1.2,
    )
    # plt.colorbar()
    plt.axis("equal")
    plt.axis("off")
    plt.savefig(FIGDIR / "rotor_distribution_illustration.png", dpi=400, bbox_inches="tight")


if __name__ == "__main__":
    X, Y, Z = generate(regenerate=False)
    plot(X, Y, Z)
