from pathlib import Path
import itertools


import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from MITRotor import MomentumTheory
from MITRotor.Utilities import for_each

FIGDIR = Path("fig")
FIGDIR.mkdir(exist_ok=True, parents=True)


momentum_theories = {
    "Limited Heck": MomentumTheory.LimitedHeck(),
    "Heck": MomentumTheory.Heck(),
    "Unified Momentum": MomentumTheory.UnifiedMomentum(),
}


def func(x):
    yaw, Ctprime, method = x

    model = momentum_theories[method]
    sol = model.solve(Ctprime, np.deg2rad(yaw))

    return dict(
        method=method,
        yaw=yaw,
        Ctprime=Ctprime,
        an=sol.an,
        Ct=sol.Ct,
        u4=sol.u4,
        v4=sol.v4,
        dp=sol.dp,
        niter=sol.niter,
        converged=sol.converged,
    )


def main():
    yaw = [30]
    Ctprime = np.linspace(0, 100, 1000)
    methods = list(momentum_theories.keys())

    params = list(itertools.product(yaw, Ctprime, methods))
    out = for_each(func, params, parallel=False)
    df = pl.from_dicts(out)
    print(df)

    plt.figure()

    for method, _df in df.filter(pl.col("converged")).group_by("method"):
        plt.plot(_df["Ctprime"], _df["niter"], label=method)
    plt.legend()
    plt.xlabel("$C_T'$")
    plt.ylabel("iterations")
    plt.savefig(FIGDIR / "example_003_iterations.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
