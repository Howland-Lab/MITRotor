from pathlib import Path
from typing import List

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tqdm import tqdm

from MITRotor import MomentumTheory


# Use Latex Fonts
plt.rcParams.update({"text.usetex": True, "font.family": "serif"})


ITERS = 5
DX = 0.1

FIGDIR = Path("fig")
FIGDIR.mkdir(exist_ok=True, parents=True)

# LES_data_fn = Path(__file__).parent.parent / f"LES_streamtube_data_DX{DX}_iters{ITERS}.mat"
LES_data_fn = Path(__file__).parent.parent / f"LES_streamtube_data.mat"


models_to_compare = {
    "linear": MomentumTheory.UnifiedMomentum(
        nonlinear_pressure_kwargs=dict(iterations=0)
    ),
    "nonlinear": MomentumTheory.UnifiedMomentum(
        nonlinear_pressure_kwargs=dict(iterations=ITERS, dx=DX)
    ),
    # "nonlinear_variation": MomentumTheory.UnifiedMomentum(
    #     nonlinear_pressure_kwargs=dict(iterations=4)
    # ),
}
Ctprimes = np.concatenate(
    [np.linspace(0, 12, 100), np.linspace(12, 4000, 200), [100000]]
)

plot_kwargs = {
    "LES_ctp": dict(ls="", marker=".", c="k", label="LES ($C_T'$ input)"),
    "LES_ct": dict(ls="", marker="x", c="k", label="LES ($C_T$ input)"),
    "linear": dict(label="Unified model (linear)"),
    "nonlinear": dict(label="Unified model (nonlinear)"),
    "nonlinear_variation": dict(label="nonlinear (variation)"),
}


def load_LES_data_ctp(filename: Path) -> List[MomentumTheory.MomentumSolution]:
    data = loadmat(filename)
    print(list(data))

    out = []
    for Cp, Ct, Ctp, p4, a, u4 in zip(
        data["Cp_ctp_input"][0],
        data["Ct_ctp_input"][0],
        data["Ctp_vector"][0],
        data["P4_ctp_input"][0],
        data["a_ctp_input"][0],
        data["u4_ctp_input"][0],
    ):
        out.append(MomentumTheory.MomentumSolution(Ctp, 0, a, u4, 0, p4, 0, 0, True, 0))

    return out


def load_LES_data_ct(filename: Path) -> List[MomentumTheory.MomentumSolution]:
    data = loadmat(filename)
    out = []
    for Cp, Ct, p4, a, u4 in zip(
        data["Cp_ct_input"][0],
        data["Ct_ct_input"][0],
        data["P4_ct_input"][0],
        data["a_ct_input"][0],
        data["u4_ct_input"][0],
    ):
        Ctp = Ct / (1 - a) ** 2
        out.append(MomentumTheory.MomentumSolution(Ctp, 0, a, u4, 0, p4, 0, 0, True, 0))
    return out


if __name__ == "__main__":
    # Load LES
    results = {
        "LES_ctp": load_LES_data_ctp(LES_data_fn),
        "LES_ct": load_LES_data_ct(LES_data_fn),
    }

    # Run unified model and variations
    for name, model in models_to_compare.items():
        results[name] = []
        for Ctprime in tqdm(Ctprimes):
            results[name].append(model.solve(Ctprime, 0))

    # Plots like Mike's
    fig, axes = plt.subplots(1, 2, sharey=False, figsize=(8, 2))
    for name, sol_list in results.items():
        p = [-sol.dp for sol in sol_list]
        a = [sol.an for sol in sol_list]
        Ctprime = [sol.Ctprime for sol in sol_list]

        axes[0].plot(Ctprime, p, **plot_kwargs[name])
        axes[1].plot(a, p, **plot_kwargs[name])

    axes[0].set_xlabel("$C_T'$")
    axes[1].set_xlabel("$a_n$")
    axes[0].set_ylabel("$p_1-p_4$")

    axes[0].set_ylim(0, 0.15)
    axes[1].set_ylim(0, 0.5)
    axes[0].set_xlim(0, 12)
    axes[1].set_xlim(0, 1)

    axes[0].legend(ncol=len(results), loc="lower center", bbox_to_anchor=(1, 1.01))

    plt.savefig(
        FIGDIR / "example_008_modified_unified.png", dpi=300, bbox_inches="tight"
    )

    # Plot x0, p_g
    fig, axes = plt.subplots(2, 2, sharex="col", sharey="row", figsize=(5, 5))
    for name, sol_list in results.items():
        if name in ["LES_ctp", "LES_ct"]:
            continue
        p_g = [-sol.dp_NL for sol in sol_list]
        x0 = [sol.x0 for sol in sol_list]

        Ctprime = [sol.Ctprime for sol in sol_list]
        an = [sol.an for sol in sol_list]

        axes[0, 0].plot(an, p_g, **plot_kwargs[name])
        axes[1, 0].plot(an, x0, **plot_kwargs[name])

        axes[0, 1].plot(Ctprime, p_g, **plot_kwargs[name])
        axes[1, 1].plot(Ctprime, x0, **plot_kwargs[name])

        # axes[1].plot(a, p, **plot_kwargs[name])

    axes[0, 0].set_xlim(0, 1)
    axes[1, 1].set_xlim(0, 12)
    axes[1, 1].set_ylim(0, 15)

    axes[0, 0].set_ylabel("$p^g$")
    axes[1, 0].set_ylabel("$x_0$")
    axes[1, 0].set_xlabel("$a_n$")
    axes[1, 1].set_xlabel("$C_T'$")

    axes[0, 0].legend(ncol=len(results), loc="lower center", bbox_to_anchor=(1, 1.01))

    plt.savefig(
        FIGDIR / "example_008_modified_unified_x0.png", dpi=300, bbox_inches="tight"
    )

    plt.figure(figsize=(6, 3))
    sol_list = results["nonlinear"]
    dp_grid, x0_grid = models_to_compare[
        "nonlinear"
    ].nonlinear_pressure.interpolator.grid
    p_g_vals = models_to_compare["nonlinear"].nonlinear_pressure.interpolator.values

    levels = np.arange(-0.3, 0.001, 0.025)

    CF = plt.contourf(x0_grid, dp_grid, p_g_vals, levels=levels, cmap="viridis_r")
    CS = plt.contour(x0_grid, dp_grid, p_g_vals, levels=levels, colors="k")
    plt.clabel(CS, inline=True, fontsize=10, fmt="%1.3f")

    cbar = plt.colorbar(CF)
    cbar.set_label(label="$p^g$ [-]")

    dp = [sol.Ct / 2 for sol in sol_list]
    x0 = [sol.x0 for sol in sol_list]

    plt.plot(x0, dp, "r--")

    plt.xlabel("$x_0$")
    plt.ylabel("$\Delta p$")

    plt.xlim(0, 15)

    plt.savefig(
        FIGDIR / "example_008_modified_unified_x0_vs_pg.png",
        dpi=300,
        bbox_inches="tight",
    )
