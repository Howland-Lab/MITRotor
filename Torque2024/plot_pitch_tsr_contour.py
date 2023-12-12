from pathlib import Path
from typing import Dict


import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from MITRotor.BEM import BEM, BEMSolution
from MITRotor.ReferenceTurbines import IEA10MW
from Torque2024.cache import cache_polars
from Torque2024.shared_generation import find_optimal_setpoint, KOmega, ContourData, find_power_maximising_Ctprime_setpoints


FIGDIR = Path("fig")
FIGDIR.mkdir(exist_ok=True, parents=True)

PITCHES = np.deg2rad(np.arange(-15, 15.001, 1))
TSRS = np.arange(5, 12.001, 0.25)
YAWS = np.deg2rad([0, 45])
YAW2 = 45  # deg


def extract_output(sol: BEMSolution, data_type: str = "") -> Dict[str, float]:
    """
    Extract the output from a bem object and return a consistent dictionary.
    - Returns essential values for output dataframe.
    - If bem is not converged, returns data with nans.
    """
    if sol.converged:
        return dict(
            pitch=np.round(np.rad2deg(sol.pitch), 4),
            tsr=sol.tsr,
            yaw=np.round(np.rad2deg(sol.yaw), 4),
            Cp=sol.Cp(),
            Ct=sol.Ct(),
            Ctprime=sol.Ctprime(),
            a=sol.a(),
            u4=sol.u4(),
            v4=sol.v4(),
            data_type=data_type,
        )
    else:
        return dict(
            pitch=np.round(np.rad2deg(sol.pitch), 4),
            tsr=sol.tsr,
            yaw=np.round(np.rad2deg(sol.yaw), 4),
            Cp=np.nan,
            Ct=np.nan,
            Ctprime=np.nan,
            a=np.nan,
            u4=np.nan,
            v4=np.nan,
            data_type=data_type,
        )


def _generate_contour_and_others(yaw: float, bem: BEM, sol_opt: BEMSolution) -> pl.DataFrame:
    """
    Generates the contour grid, finds the optimal, finds the K-omega^2 set point, and the derating trajectory.
    yaw in degrees
    """
    out = []
    # generate contour
    contourdatagenerator = ContourData(BEM(IEA10MW()), yaw=np.deg2rad(yaw))
    out.extend(extract_output(x, data_type="contour") for x in contourdatagenerator(PITCHES, TSRS, parallel=True))
    
    # generate global optimal
    global_opt = find_optimal_setpoint(bem, yaw=np.deg2rad(yaw))
    out.append(extract_output(global_opt, data_type="optimal"))
    
    # generate k-omega
    out.append(extract_output(KOmega(bem, sol_opt)(np.deg2rad(yaw)), data_type="k-omega"))

    # generate trajectory
    ctprimes = np.linspace(0.01, global_opt.Ctprime(), 10)
    out.extend(
        extract_output(find_power_maximising_Ctprime_setpoints(bem, ctprime, np.deg2rad(yaw)), data_type="trajectory")
        for ctprime in ctprimes
    )

    return pl.from_dicts(out)


@cache_polars("data/pitch_tsr_contour.csv")
def generate(yaw2, regenerate=False):
    bem = BEM(IEA10MW())
    sol_opt = find_optimal_setpoint(bem, yaw=0)
    df = _generate_contour_and_others(0, bem, sol_opt)
    df2 = _generate_contour_and_others(yaw2, bem, sol_opt)

    return pl.concat([df, df2])


def plot(df, yaw2=45):
    dfs = [
        df.filter(pl.col("yaw") == 0),
        df.filter(pl.col("yaw") == yaw2),
    ]

    fig, axes = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(8, 4))

    [ax.set_xlabel("Pitch [deg]") for ax in axes]
    axes[0].set_ylabel("Tip Speed Ratio [-]")

    plt.xlim(-15, 15)
    plt.ylim(5, 12)

    for ax, _df in zip(axes, dfs):
        ## Plot surface
        df_piv = _df.filter(pl.col("data_type") == "contour").pivot(
            index="tsr", columns="pitch", values="Cp", aggregate_function=None
        )
        tsr = df_piv["tsr"].to_numpy()
        pitch = np.array(df_piv.columns[1:], dtype=float)
        Z = df_piv.to_numpy()[:, 1:]
        Z[Z < 0] = 0.01

        levels = np.arange(0, 0.60, 0.05)
        ax.contourf(pitch, tsr, Z, levels=levels, cmap="viridis")
        CS = ax.contour(pitch, tsr, Z, levels=levels, colors="k", linewidths=0.8)
        ax.clabel(CS, inline=True, fontsize=10)

        # Plot zero-yaw optimal
        dat = df.filter(pl.col("yaw") == 0).filter(pl.col("data_type") == "optimal")
        ax.plot(dat["pitch"], dat["tsr"], "o", color="tab:orange", label="zero-yaw optimal", ms=15)

        # Plot K-omega strategy
        dat = _df.filter(pl.col("data_type") == "k-omega")
        ax.plot(dat["pitch"], dat["tsr"], "P", color="blue", label="$K-\Omega^2$", ms=15)

        # Plot global optimal
        dat = _df.filter(pl.col("data_type") == "optimal")
        ax.plot(dat["pitch"], dat["tsr"], "*", color="lime", label="Global optimal", ms=10)

        # Plot zero-yaw trajectory
        dat = df.filter(pl.col("yaw") == 0).filter(pl.col("data_type") == "trajectory")
        ax.plot(dat["pitch"], dat["tsr"], ":", color="tab:orange", label="min $C_T$ ($\gamma=0°$)")

    # Plot yawed trajectory in right figure only
    dat = df.filter(pl.col("yaw") == yaw2).filter(pl.col("data_type") == "trajectory")
    ax.plot(dat["pitch"], dat["tsr"], ":", color="lime", label="min $C_T$ ($\gamma=45°$)")

    # Legend, including reordering so lines are at bottom
    handles, labels = axes[1].get_legend_handles_labels()
    order = [2, 4, 0, 3, 1]
    axes[1].legend(
        [handles[idx] for idx in order], [labels[idx] for idx in order], ncol=3, loc="lower center", bbox_to_anchor=(-0.1, 1.1)
    )

    axes[0].set_title("$C_P~(\gamma=0^o)$")
    axes[1].set_title("$C_P~(\gamma=45^o)$")

    # Titles (as text)
    axes[0].text(0.02, 0.98, "a)", fontsize=12, color="w", ha="left", va="top", transform=axes[0].transAxes)
    axes[1].text(0.02, 0.98, "b)", fontsize=12, color="w", ha="left", va="top", transform=axes[1].transAxes)

    plt.savefig(FIGDIR / "pitch_tsr_contour.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    df = generate(yaw2=YAW2, regenerate=False)
    plot(df)
