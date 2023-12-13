from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from MITRotor.BEM import BEM
from MITRotor.Geometry import BEMGeometry
from MITRotor.ReferenceTurbines import IEA10MW
from Torque2024.cache import cache_polars

# Use Latex Fonts
plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

FIGDIR = Path("fig")
FIGDIR.mkdir(exist_ok=True, parents=True)


# @cache_polars("data/yaw_power_loss.csv")
def generate(regenerate=False):
    df = None
    return df


def plot(df):
    pass


if __name__ == "__main__":
    df = generate()
    plot(df)
