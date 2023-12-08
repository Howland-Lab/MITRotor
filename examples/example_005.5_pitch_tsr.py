from pathlib import Path
import itertools
from typing import Dict


import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from foreach import foreach

from MITRotor.BEM import BEM
from MITRotor.ReferenceTurbines import IEA10MW
from MITRotor.Geometry import BEMGeometry


# Use Latex Fonts
plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

FIGDIR = Path("fig")
FIGDIR.mkdir(exist_ok=True, parents=True)

fig_fn = FIGDIR / "example_005.5_pitch_tsr.png"

REGENERATE = True
FN_PITCH_TSR = Path("pitch_tsr.csv")

rotor = IEA10MW()
geometry = BEMGeometry(Nr=100, Ntheta=20)
bem = BEM(rotor=rotor, geometry=geometry)


def func(x) -> Dict[str, any]:
    pitch, tsr, yaw = x
    sol = bem(np.deg2rad(pitch), tsr, yaw)

    return dict(
        pitch=np.round(np.rad2deg(sol.pitch), 2),
        tsr=sol.tsr,
        yaw=sol.yaw,
        Cp=sol.Cp(),
        Ct=sol.Ct(),
        Ctprime=sol.Ctprime(),
        an=sol.a(),
        niter=sol.niter,
        converged=sol.converged,
    )


def main():
    pass


if __name__ == "__main__":
    main()
    pitches = np.linspace(-15, 20, 50)
    tsrs = np.linspace(1, 20, 50)
    yaws = [0.0]

    params = list(itertools.product(pitches, tsrs, yaws))

    # Load data if already generate. Otherwise, generate.
    if FN_PITCH_TSR.exists() and not REGENERATE:
        df = pl.read_csv(FN_PITCH_TSR)
    else:
        out = foreach(func, params, parallel=True)
        df = pl.from_dicts(out)
        df.write_csv(FN_PITCH_TSR)

    print(df)
    df_Cp = df.pivot(index="tsr", columns="pitch", values="Cp", aggregate_function=None).fill_nan(None).interpolate()
    df_Ct = df.pivot(index="tsr", columns="pitch", values="Ct", aggregate_function=None).fill_nan(None).interpolate()
    tsr = df_Cp["tsr"].to_numpy()
    pitch = np.array(df_Cp.columns[1:], dtype=float)
    Cp = np.maximum(df_Cp.to_numpy()[:, 1:], 0.025)
    Ct = np.maximum(df_Ct.to_numpy()[:, 1:], 0.05)

    fig, axes = plt.subplots(1, 2, sharey=True, sharex=True, figsize=1.5 * np.array((4.5, 3)))
    # CP
    levels = np.arange(0.0, 0.61, 0.05)
    axes[0].contourf(pitch, tsr, Cp, levels=levels, cmap="viridis")
    CS = axes[0].contour(pitch, tsr, Cp, levels=levels, colors="k")
    axes[0].clabel(CS, inline=True, fontsize=10)

    levels = np.arange(0.0, 4, 0.2)
    axes[1].contourf(pitch, tsr, Ct, levels=levels, cmap="plasma")
    CS = axes[1].contour(pitch, tsr, Ct, levels=levels, colors="k")
    axes[1].clabel(CS, inline=True, fontsize=10)

    axes[1].set_xlabel("Pitch [deg]")
    axes[0].set_ylabel("Tip Speed Ratio [-]")
    axes[0].set_title("$C_P$", fontsize="small")
    axes[1].set_title("$C_T$", fontsize="small")

    axes[1].set_xlabel("Blade Pitch Angle [deg]")

    [ax.set_xticks(np.arange(-10, 21, 10)) for ax in axes]

    # axes[1].legend(loc="lower center", fontsize="xx-small")
    plt.savefig(fig_fn, dpi=500, bbox_inches="tight")
