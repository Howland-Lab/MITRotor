from pathlib import Path
import itertools


import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from MITRotor.BEM import BEM
from MITRotor.ReferenceTurbines import IEA15MW

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
    # "heck",
    # "fixed",
]

method_labels = {
    "madsen": "Madsen (2020)",
    "unified": "Unified Momentum",
}
PITCH, TSR = np.deg2rad(-10), 10


def main():
    pass


if __name__ == "__main__":
    main()

    out = {}
    for method in methods:
        sol = BEM(rotor, Cta_method=method, Nr=100).solve(PITCH, TSR)
        out[method] = sol
        print(sol.converged, sol.inner_niter, sol.outer_niter)

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
