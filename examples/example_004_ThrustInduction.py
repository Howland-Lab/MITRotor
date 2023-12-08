from pathlib import Path
import itertools


import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from MITRotor import ThrustInduction
from MITRotor.Utilities import for_each

FIGDIR = Path("fig")
FIGDIR.mkdir(exist_ok=True, parents=True)


CT_a_methods = {
    # "Fixed": ThrustInduction.FixedInduction(),
    "UnifiedMomentum": ThrustInduction.CTaUnifiedMomentum(),
    "RotorAveragedUnifiedMomentum": ThrustInduction.CTaRotorAveragedUnifiedMomentum(),
    "Heck": ThrustInduction.CTaHeck(),
    "Madsen": ThrustInduction.Madsen(),
}


line_style = {
    # "Fixed": ThrustInduction.FixedInduction(),
    "UnifiedMomentum": "-",
    "RotorAveragedUnifiedMomentum": "o-",
    "Heck": "--",
    "Madsen": ":",
}


def func(x):
    yaw, Ct, method = x
    model = CT_a_methods[method]
    a = model.Ct_a(Ct, np.deg2rad(yaw))

    return dict(model=method, Ct=Ct, yaw=yaw, a=a)


def main():
    yaws = np.arange(0, 41, 10)  # np.linspace(-30, 30, 5)
    Cts = np.linspace(0, 2, 50)
    methods = list(CT_a_methods)

    params = list(itertools.product(yaws, Cts, methods))
    out = for_each(func, params, parallel=True)

    df = pl.from_dicts(out).filter(pl.col("a") < 1)

    plt.figure()
    for (method, yaw), _df in df.group_by("model", "yaw"):
        plt.plot(
            _df["a"],
            _df["Ct"],
            line_style[method],
            c=plt.cm.viridis(yaw / 40),
            label=method,
        )

    plt.xlim(0, 1)
    plt.ylim(0, 1.8)

    plt.grid()

    plt.xlabel("$a$")
    plt.ylabel("$C_T$")

    plt.savefig(FIGDIR / "example_004_ThrustInduction.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
