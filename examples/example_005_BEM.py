from pathlib import Path
import itertools


import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from MITRotor.BEM import BEM
from MITRotor.ReferenceTurbines import IEA15MW

# Use Latex Fonts
plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

rotor = IEA15MW()


FIGDIR = Path("fig")
FIGDIR.mkdir(exist_ok=True, parents=True)

fig_fn = FIGDIR / "example_005_BEM.png"


to_plot = {
    "a": dict(decomp=True),
    # "aprime": dict(decomp=True),
    "Ct": dict(decomp=True),
    # "Ctprime": dict(decomp=True),
    "Cp": dict(decomp=True),
    # "u4": dict(decomp=True),
    # "v4": dict(decomp=True),
    # "dp": dict(decomp=True),
    # "Vax": dict(decomp=True),
    # "Vtan": dict(decomp=True),
    # "torque": dict(U_inf=10, decomp=True),
    "Ftan": dict(U_inf=10, decomp=True),
    "Fax": dict(U_inf=10, decomp=True),
}

to_plot_label = {
    "a": "$a$",
    "Ct": "$C_T$",
    "Ctprime": "$C_T'$",
    "Cp": "$C_P$",
    "Ftan": "$F_{tan}$\n[N/m]",
    "Fax": "$F_{ax}$\n[N/m]",
}
methods = [
    "madsen",
    "unified",
    "rotoraveragedunified",
    # "heck",
    # "fixed",
]

method_labels = {
    "madsen": "Madsen (2020)",
    "unified": "Unified Momentum",
    "rotoraveragedunified": "Rotor-averaged Unified Momentum",
}
relaxation = {
    "madsen": 0.25,
    "unified": 0.25,
    "rotoraveragedunified": 0.25,
}
PITCH, TSR = np.deg2rad(-10), 10
PITCH, TSR = np.deg2rad(0), 7


def main():
    out = {}
    for method in methods:
        print()
        print(f"Simulating rotor using {method=}")
        print(f"at {np.rad2deg(PITCH)=:.2f} deg, {TSR=:.2f}...")

        sol = BEM(
            rotor, Cta_method=method, Nr=100, outer_relax=relaxation[method]
        ).solve(PITCH, TSR)
        out[method] = sol

        print("converged!" if sol.converged else "not converged!")
        print(f"{sol.outer_niter=}")

    fig, axes = plt.subplots(
        len(to_plot), 1, sharex=True, figsize=1.2 * np.array((4, 4))
    )

    for method, sol in out.items():
        for ax, (key, kwargs) in zip(axes, to_plot.items()):
            ax.plot(sol.mu, getattr(sol, key)(**kwargs), label=method_labels[method])
            ax.set_ylabel(to_plot_label[key])

    [ax.grid() for ax in axes]
    plt.xlim(0, 1)
    axes[-1].set_xlabel("Radial position [-]")
    axes[0].legend(ncol=len(methods), bbox_to_anchor=(0.5, 1.05), loc="lower center")

    plt.savefig(fig_fn, dpi=300, bbox_inches="tight")

    return all([x.converged for x in out.values()])


if __name__ == "__main__":
    main()
