import numpy as np

from MITRotor import BEM, IEA10MW

if __name__ == "__main__":
    # Initialize rotor using the IEA10MW reference wind turbine model.
    rotor = IEA10MW()
    bem = BEM(rotor=rotor)

    # solve BEM for a control set point.
    pitch, tsr, yaw = np.deg2rad(0), 7.0, np.deg2rad(30.0)
    sol = bem(pitch, tsr, yaw)

    # Print various quantities in BEM solution
    if sol.converged:
        print(f"BEM solution converged in {sol.niter} iterations.")
    else:
        print("BEM solution did NOT converge!")

    print(f"Control setpoints: {sol.pitch=:2.2f}, {sol.tsr=:2.2f}, {sol.yaw=:2.2f}")
    print(f"Power coefficient: {sol.Cp():2.2f}")
    print(f"Thrust coefficient: {sol.Ct():2.2f}")
    print(f"Local thrust coefficient: {sol.Ctprime():2.2f}")
    print(f"Axial induction: {sol.a():2.2f}")
    print(f"Rotor-effective windspeed: {sol.U():2.2f}")
    print(f"Far-wake streamwise velocity: {sol.u4:2.2f}")
    print(f"Far-wake lateral velocity: {sol.v4:2.2f}")
