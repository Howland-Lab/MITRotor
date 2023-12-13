from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from foreach import foreach
from tqdm import tqdm
from UnifiedMomentumModel.Momentum import Heck

from MITRotor.BEM import BEM
from MITRotor.ReferenceTurbines import IEA10MW
from Torque2024.cache import cache_polars
from Torque2024.shared_generation import find_power_maximising_Ctprime_setpoints

# Use Latex Fonts
plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

FIGDIR = Path("fig")
FIGDIR.mkdir(exist_ok=True, parents=True)

CTPRIMES = np.arange(0.4, 2.001, 0.4)
YAWS = np.arange(-30.0, 30.1, 5)


AD = Heck()
bem = BEM(IEA10MW())


def _generate_single(x):
    ctprime, yaw = x
    sol_AD = AD(ctprime, np.deg2rad(yaw))
    sol_BEM = find_power_maximising_Ctprime_setpoints(bem, ctprime, np.deg2rad(yaw))

    return sol_AD, sol_BEM


@cache_polars("data/BEM_AD_equivalency.csv")
def generate(regenerate=False):
    params = list(product(CTPRIMES, YAWS))
    out = []

    solutions = foreach(_generate_single, params)

    for sol_AD, sol_BEM in tqdm(solutions):
        out.append(
            dict(
                model="AD",
                ctprime=np.round(sol_AD.Ctprime, 1),
                yaw=np.round(np.rad2deg(sol_AD.yaw), 1),
                Ct=sol_AD.Ct,
                Cp=sol_AD.Cp,
                a=sol_AD.an,
                u4=sol_AD.u4,
                v4=sol_AD.v4,
            )
        )
        out.append(
            dict(
                model="BEM",
                ctprime=np.round(sol_BEM.Ctprime(), 1),
                yaw=np.round(np.rad2deg(sol_BEM.yaw), 1),
                Ct=sol_BEM.Ct(),
                Cp=sol_BEM.Cp(),
                a=sol_BEM.a(),
                u4=sol_BEM.u4(),
                v4=sol_BEM.v4(),
            )
        )
    df = (
        pl.from_dicts(out)
        .groupby(["model", "ctprime"])
        .agg([pl.all(), pl.col("Cp").filter(pl.col("yaw") == 0).first().alias("Cp_ref")])
        .explode(pl.col(pl.List(pl.Float64)))
        .with_columns((pl.col("Cp") / pl.col("Cp_ref")).alias("power_loss"))
    )
    return df


def plot(df: pl.DataFrame):
    keys_to_plot = {
        "u4": "$u_4/u_\infty$",
        "power_loss": "$C_P/C_P(\gamma=0)$",
        "v4": "$v_4/u_\infty$",
        "a": "$a_n$",
    }
    cmap = plt.cm.viridis

    fig, axes = plt.subplots(2, 2, sharex=True, figsize=1.5 * np.array((5, 2)))
    plt.subplots_adjust(wspace=0.3, hspace=0.15)

    print(df["ctprime"].unique())
    for ax, key in zip(axes.ravel(), keys_to_plot):
        for ctprime, _df in df.sort("ctprime").group_by("ctprime", maintain_order=True):
            # draw AD
            ax.plot(
                _df.filter(pl.col("model") == "AD")["yaw"],
                _df.filter(pl.col("model") == "AD")[key],
                c=cmap(ctprime / 2),
            )
            # draw BEM
            ax.plot(
                _df.filter(pl.col("model") == "BEM")["yaw"],
                _df.filter(pl.col("model") == "BEM")[key],
                ".",
                ms=15,
                c=cmap(ctprime / 2),
            )
    # Set ticks
    axes[0, 0].set_xticks(np.arange(-30, 30.1, 15))
    axes[0, 1].set_yticks(np.arange(0.7, 1.001, 0.1))

    # set xlim
    axes[0,0].set_xlim(-30, 30)

    # ylabels
    for ax, label in zip(axes.ravel(), keys_to_plot.values()):
        ax.set_ylabel(label)

    # xlabels
    axes[1, 0].set_xlabel("$\gamma~(^o)$")
    axes[1, 1].set_xlabel("$\gamma~(^o)$")

    # Add a custom colorbar
    norm = plt.Normalize(df["ctprime"].min(), df["ctprime"].max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes, pad=0.05, aspect=30)
    cbar.set_label("$C_T'$")

    plt.savefig(FIGDIR / "BEM_AD_equivalency.png", dpi=500, bbox_inches="tight")


if __name__ == "__main__":
    df = generate(regenerate=False)
    plot(df)
