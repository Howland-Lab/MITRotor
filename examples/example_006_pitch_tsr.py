from pathlib import Path
import itertools


import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from MITRotor.BEM import BEM
from MITRotor.ReferenceTurbines import IEA15MW
from MITRotor.Utilities import for_each

rotor = IEA15MW()

# Use Latex Fonts
plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

FIGDIR = Path("fig")
FIGDIR.mkdir(exist_ok=True, parents=True)

fig_fn = FIGDIR / "example_006_pitch_tsr.png"

FN_PITCH_TSR = Path("pitch_tsr_blag_new_tiploss.csv")


def func(x):
    method, pitch, tsr, yaw = x
    relax = 0.5 if method == "madsen" else 0.25
    sol = BEM(rotor, Cta_method=method, outer_relax=relax).solve(np.deg2rad(pitch), tsr)

    if True:  # sol.converged:
        return dict(
            method=method,
            pitch=pitch,
            tsr=tsr,
            yaw=yaw,
            converged=sol.converged,
            Cp=sol.Cp(),
            Ct=sol.Ct(),
            a=sol.a(),
            aprime=sol.aprime(),
            dp=sol.dp(),
            u4=sol.u4(),
            v4=sol.v4(),
            Ctprime=sol.Ctprime(),
            inner_niter=sol.inner_niter,
            # inner_relax=sol.inner_relax,
            outer_niter=sol.outer_niter,
            # outer_relax=sol.outer_relax,
        )
    else:
        return dict(
            method=method,
            pitch=pitch,
            tsr=tsr,
            yaw=yaw,
            converged=sol.converged,
            Cp=np.nan,
            Ct=np.nan,
            a=sol.a(),
            aprime=sol.aprime(),
            dp=sol.dp(),
            u4=sol.u4(),
            v4=sol.v4(),
            Ctprime=np.nan,
            inner_niter=sol.inner_niter,
            # inner_relax=sol.inner_relax,
            outer_niter=sol.outer_niter,
            # outer_relax=sol.outer_relax,
        )


def main():
    pass


if __name__ == "__main__":
    main()
    methods = ["madsen", "unified"]
    pitches = np.linspace(-15, 20, 50)
    tsrs = np.linspace(1, 20, 50)
    yaws = [0.0]

    params = list(itertools.product(methods, pitches, tsrs, yaws))

    # Load data if already generate. Otherwise, generate.
    if FN_PITCH_TSR.exists():
        df = pl.read_csv(FN_PITCH_TSR)
    else:
        out = for_each(func, params, parallel=True)
        df = pl.from_dicts(out)
        df.write_csv(FN_PITCH_TSR)

    print(df)

    # points of interest
    pitch_best_unified, tsr_best_unified, Cp_best_unified = (
        df.filter(pl.col("method") == "unified")
        .filter(pl.col("Cp") == pl.col("Cp").max())
        .select("pitch", "tsr", "Cp")
        .to_numpy()[0]
    )
    pitch_best_madsen, tsr_best_madsen, Cp_best_madsen = (
        df.filter(pl.col("method") == "madsen")
        .filter(pl.col("Cp") == pl.col("Cp").max())
        .select("pitch", "tsr", "Cp")
        .to_numpy()[0]
    )

    points_of_interest = {
        "optimal_unified": (pitch_best_unified, tsr_best_unified),
        "optimal_madsen": (pitch_best_madsen, tsr_best_madsen),
        "high_thrust": (-10, 10),
    }

    # point_marker_style = {
    #     "optimal": "*",
    #     "high_thrust": "+",
    #     "negative_stall": "P",
    # }

    # point_color = {
    #     "optimal": "k",
    #     "high_thrust": "r",
    #     "negative_stall": "b",
    # }
    marker_properties = {
        "optimal_unified": dict(
            label="Optimal\n(Unified momentum)",
            marker="*",
            ms=5,
            c="k",
            ls="",
        ),
        "optimal_madsen": dict(
            label="Optimal\n(Madsen (2020))",
            marker="*",
            ms=5,
            c="tab:orange",
            ls="",
        ),
        "high_thrust": dict(
            label="High-thrust sample",
            marker="X",
            ms=7,
            c="magenta",
            ls="",
        ),
    }

    # # plot

    df = df.with_columns(
        Cp_filt=pl.when(pl.col("Cp") < 0).then(0.0).otherwise(pl.col("Cp"))
    )

    df_piv_unified_Cp = (
        df.filter(pl.col("method") == "unified")
        .pivot(index="tsr", columns="pitch", values="Cp_filt", aggregate_function=None)
        .fill_nan(None)
        .interpolate()
    )
    df_piv_madsen_Cp = (
        df.filter(pl.col("method") == "madsen")
        .pivot(index="tsr", columns="pitch", values="Cp_filt", aggregate_function=None)
        .fill_nan(None)
        .interpolate()
    )
    df_piv_unified_Ct = (
        df.filter(pl.col("method") == "unified")
        .pivot(index="tsr", columns="pitch", values="Ct", aggregate_function=None)
        .fill_nan(None)
        .interpolate()
    )
    df_piv_madsen_Ct = (
        df.filter(pl.col("method") == "madsen")
        .pivot(index="tsr", columns="pitch", values="Ct", aggregate_function=None)
        .fill_nan(None)
        .interpolate()
    )
    tsr = df_piv_unified_Cp["tsr"].to_numpy()
    pitch = np.array(df_piv_unified_Cp.columns[1:], dtype=float)
    Cp_unified = df_piv_unified_Cp.to_numpy()[:, 1:]
    Ct_unified = df_piv_unified_Ct.to_numpy()[:, 1:]
    Cp_madsen = df_piv_madsen_Cp.to_numpy()[:, 1:]
    Ct_madsen = df_piv_madsen_Ct.to_numpy()[:, 1:]

    mask_unified = np.isnan(Cp_unified)
    mask_madsen = np.isnan(Cp_madsen)
    Cp_unified[mask_unified] = 0
    Cp_madsen[mask_madsen] = 0
    Ct_unified[mask_unified] = 0
    Ct_madsen[mask_madsen] = 0
    Ct_madsen[(Ct_unified > 1.25)] = 1.51
    Cp_unified[Cp_unified < 0.05] = 0.05
    Cp_madsen[Cp_madsen < 0.05] = 0.05

    # Plotting
    fig, axes = plt.subplots(
        1, 3, sharey=True, sharex=True, figsize=1.5 * np.array((4.5, 3))
    )
    # CP
    levels = np.arange(0.0, 0.61, 0.1)
    axes[0].contourf(pitch, tsr, Cp_unified, levels=levels, cmap="viridis")
    CS = axes[0].contour(pitch, tsr, Cp_unified, levels=levels, colors="k")
    axes[0].clabel(CS, inline=True, fontsize=10)

    axes[1].contourf(pitch, tsr, Cp_madsen, levels=levels, cmap="viridis")
    CS = axes[1].contour(pitch, tsr, Cp_madsen, levels=levels, colors="k")
    axes[1].clabel(CS, inline=True, fontsize=10)

    # diff
    Cp_diff = np.abs(Cp_unified - Cp_madsen)
    Cp_diff[np.abs(Cp_diff) < 0.01] = 0.01
    levels = np.arange(-0.3, 0.31, 0.05)
    axes[2].contourf(pitch, tsr, Cp_diff, levels=levels, cmap="RdYlBu_r")
    CS = axes[2].contour(pitch, tsr, Cp_diff, levels=levels, colors="k")
    axes[2].clabel(CS, inline=True, fontsize=10)
    print(np.max(np.abs(Cp_diff)))

    # Plot points of interest
    for key, (pitch, tsr) in points_of_interest.items():
        print(key, pitch, tsr)
        for ax in axes:
            ax.plot([pitch], [tsr], **marker_properties[key])

    axes[1].set_xlabel("Pitch [deg]")
    axes[0].set_ylabel("Tip Speed Ratio [-]")
    axes[0].set_title("$C_P$ (Unified momentum)", fontsize="small")
    axes[1].set_title("$C_P$ (Madsen (2020))", fontsize="small")
    axes[2].set_title("Absolute difference", fontsize="small")

    axes[1].set_xlabel("Blade Pitch Angle [deg]")

    [ax.set_xticks(np.arange(-10, 21, 10)) for ax in axes]

    axes[2].legend(loc="lower center", fontsize="xx-small")
    plt.savefig(fig_fn, dpi=300, bbox_inches="tight")
