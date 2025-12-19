import numpy as np
import matplotlib.pyplot as plt
from MITRotor import (
    BEM,
    IEA10MW,
    DefaultAerodynamics,
    UnifiedMomentum,
    BEMGeometry,
)


def test():
    model = BEM(
        aerodynamic_model=DefaultAerodynamics(apply_3D_stall_correction=False),
        rotor=IEA10MW(),
    )

    aoa = np.radians(np.linspace(-5, 30, 100))
    mu = 0.4 * np.ones_like(aoa)
    tsr = 2.0
    Cl_uncorr, Cd_uncorr = model.rotor.clcd(
        mu, aoa, tsr=tsr, apply_3D_stall_correction=False
    )
    Cl_corr, Cd_corr = model.rotor.clcd(
        mu, aoa, tsr=tsr, apply_3D_stall_correction=True
    )

    # slope = model.rotor.clcd.slope_of_cl_curve(mu[0], )

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(np.degrees(aoa), Cl_uncorr, label="2D Airfoil")
    ax[0].plot(np.degrees(aoa), Cl_corr, label="3D Stall Corrected")
    # ax[0].plot(np.degrees(aoa),  (aoa + np.radians(3)), label="Thin Airfoil Theory", linestyle='--')
    ax[0].set_xlabel("Angle of Attack (degrees)")
    ax[0].set_ylabel("Cl")
    ax[0].set_title("Lift Coefficient with 3D Stall Correction")
    ax[0].legend()

    ax[1].plot(np.degrees(aoa), Cd_uncorr, label="2D Airfoil")
    ax[1].plot(np.degrees(aoa), Cd_corr, label="3D Stall Corrected", color='orange')
    ax[1].set_xlabel("Angle of Attack (degrees)")
    ax[1].set_ylabel("Cd")
    ax[1].set_title("Drag Coefficient with 3D Stall Correction")
    ax[1].legend()

    plt.savefig("3D_stall_correction_test.png")

if __name__ == "__main__":
    test()