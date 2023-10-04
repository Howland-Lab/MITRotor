from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from MITRotor import MomentumTheory

FIGDIR = Path("fig")
FIGDIR.mkdir(exist_ok=True, parents=True)


momentum_theories = {
    "Limited Heck": MomentumTheory.LimitedHeck(),
    "Heck": MomentumTheory.Heck(),
    "Unified Momentum": MomentumTheory.UnifiedMomentum(),
}


def main():
    yaw = np.deg2rad(60)
    Ctprime = np.linspace(0, 100, 500)
    out = {}
    for key, model in momentum_theories.items():
        sol = model.solve(Ctprime, yaw)
        data = np.vstack(
            [
                sol.an,
                sol.Ct,
                sol.Ctprime,
                sol.u4,
                sol.v4,
                sol.dp,
                yaw * np.ones_like(sol.an),
            ]
        )
        _df = pl.DataFrame(
            data, schema=["an", "Ct", "Ctprime", "u4", "v4", "dp", "yaw"]
        ).with_columns(pl.lit(key).alias("model"))
        _df = _df.with_columns(
            (
                1
                / 2
                * pl.col("Ctprime")
                * (1 - pl.col("an"))
                * np.cos(pl.col("yaw")) ** 2
                - 1
            ).alias("blah")
        )
        out[key] = _df

    to_plot = ["u4", "v4", "blah", "dp", "Ct"]
    fig, axes = plt.subplots(1, 5, sharey=True, figsize=(10, 3))
    for key, df in out.items():
        for _key, ax in zip(to_plot, axes):
            ax.plot(df["an"], df[_key], label=key)

            ax.set_title(_key)
        # axes[1].plot(sol.Ct, sol.u4, label=key)
        # axes[2].plot(sol.Ct, sol.v4, label=key)
        # axes[3].plot(sol.Ct, sol.dp, label=key)

    [ax.grid() for ax in axes]
    axes[0].legend()
    plt.savefig(FIGDIR / "example_002_blah.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
