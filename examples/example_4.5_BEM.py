from pathlib import Path

import matplotlib.pyplot as plt

from MITRotor.BEM import BEM
from MITRotor.ReferenceTurbines import IEA15MW
from MITRotor.Geometry import BEMGeometry


figdir = Path("fig_new")
figdir.mkdir(exist_ok=True, parents=True)

if __name__ == "__main__":
    rotor = IEA15MW()
    geometry = BEMGeometry(Nr=100, Ntheta=20)
    bem = BEM(rotor=rotor, geometry=geometry)
    out = bem(0.0, 7.0, 0.0)
    print(out)
    print(f"{out.Ct()=:.4f}")
    print(f"{out.Cp()=:.4f}")

    plt.plot(out.geom.mu, out.Cp(grid="radial"))
    plt.savefig(figdir / "example_4.5_bem.png", dpi=300, bbox_inches="tight")
