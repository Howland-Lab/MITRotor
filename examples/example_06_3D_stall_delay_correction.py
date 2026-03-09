import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from MITRotor import (
    BEM,
    IEA10MW,
    DefaultAerodynamics
)

figdir = Path("fig")
figdir.mkdir(exist_ok=True, parents=True)

def test():
    model = BEM(
        aerodynamic_model=DefaultAerodynamics(apply_3D_stall_correction=False),
        rotor=IEA10MW(),
    )

    aoa = np.radians(np.linspace(-5, 45, 100))
    mu = 0.3 * np.ones_like(aoa)
    tsr = 5

    Cl_uncorr, Cd_uncorr = model.rotor.clcd(
        mu, aoa, apply_3D_stall_correction=False, tsr = tsr
    )
    Cl_corr, Cd_corr = model.rotor.clcd(
        mu, aoa, apply_3D_stall_correction=True, tsr = tsr
    )


    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(np.degrees(aoa), Cl_uncorr, label="2D Airfoil")
    ax.plot(np.degrees(aoa), Cl_corr, label="3D Stall Corrected")
    ax.set_xlabel("Angle of Attack (degrees)")
    ax.set_ylabel("Cl")
    ax.set_title("Lift Coefficient with 3D Stall Correction")
    ax.legend()


    plt.savefig(figdir / "example_06_3D_stall_correction.png", dpi = 300, bbox_inches='tight')

if __name__ == "__main__":
    test()