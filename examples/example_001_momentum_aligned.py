from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from UnifiedMomentumModel import Momentum

FIGDIR = Path("fig")
FIGDIR.mkdir(exist_ok=True, parents=True)


momentum_theories = {
    "Limited Heck": Momentum.LimitedHeck(),
    "Heck": Momentum.Heck(),
    "Unified Momentum (linear)": Momentum.UnifiedMomentum(),
}


def main():
    yaw = np.deg2rad(0.0)
    Ctprime = np.linspace(-1, 100, 500)
    out = {}
    for key, model in momentum_theories.items():
        sol = model(Ctprime, yaw)
        out[key] = sol

    fig, axes = plt.subplots(4, 1, sharex=True)
    for key, sol in out.items():
        axes[0].plot(sol.Ct, sol.an, label=key)
        axes[1].plot(sol.Ct, sol.u4, label=key)
        axes[2].plot(sol.Ct, sol.v4, label=key)
        axes[3].plot(sol.Ct, sol.dp, label=key)
    axes[0].legend()
    plt.savefig(
        FIGDIR / "example_001_momentum_aligned.png", dpi=300, bbox_inches="tight"
    )


if __name__ == "__main__":
    main()
