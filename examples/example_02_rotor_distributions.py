from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from MITRotor import BEM, BEMGeometry, IEA10MW, BEMSolution

figdir = Path("fig")
figdir.mkdir(exist_ok=True, parents=True)


def plot_radial_distributions(sol: BEMSolution, save_to: Path):
    fig, axes = plt.subplots(3, 1, sharex=True)
    axes[0].plot(sol.geom.mu, sol.Cp(grid="annulus"), label="$C_P$")
    axes[1].plot(sol.geom.mu, sol.Ct(grid="annulus"), label="$C_T$")
    axes[2].plot(sol.geom.mu, sol.a(grid="annulus"), label="$a_n$")

    [ax.legend(loc="lower center") for ax in axes]

    axes[-1].set_xlabel("Radial position, $\mu$ [-]")
    plt.xlim(0, 1)
    plt.savefig(save_to, dpi=300, bbox_inches="tight")


def plot_azimuthal_variations(sol: BEMSolution, save_to: Path):
    fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=np.array([6, 6]))

    azim = np.rad2deg(sol.geom.theta)
    for mu, ax in zip([0.9, 0.6, 0.3], axes):
        # find closest grid point to target radial position.
        idx = np.searchsorted(sol.geom.mu, mu)
        ax.plot(azim, sol.W(grid="sector")[idx, :])

    axes[-1].set_xlabel("Azimuth angle (deg)")
    axes[0].set_ylabel(r"$W_N/U_0$ @ 90% span")
    axes[1].set_ylabel(r"$W_N/U_0$ @ 60% span")
    axes[2].set_ylabel(r"$W_N/U_0$ @ 30% span")


    plt.xlim(0, 360)

    plt.savefig(save_to, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    # Initialize rotor with increased radial resolution.
    rotor = IEA10MW()
    bem = BEM(rotor=rotor, geometry=BEMGeometry(Nr=100, Ntheta=10))

    # solve BEM for a control set point.
    pitch, tsr, yaw = np.deg2rad(0), 7.0, np.deg2rad(0.0)
    sol = bem(pitch, tsr, yaw)

    # Plot
    plot_radial_distributions(sol, figdir / "example_02_rotor_distributions.png")
    plot_azimuthal_variations(sol, figdir / "example_02_rotor_distributions_azimuth.png")
