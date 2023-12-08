from pathlib import Path
from typing import Tuple

import numpy as np
from foreach import foreach
from scipy.optimize import minimize, minimize_scalar, root_scalar
from UnifiedMomentumModel.Utilities.FixedPointIteration import fixedpointiteration

from MITRotor.BEM import BEM, BEMSolution
from MITRotor.ReferenceTurbines import IEA15MW


class ConvergenceError(Exception):
    ...


figdir = Path("fig")
figdir.mkdir(exist_ok=True, parents=True)


bem = BEM(IEA15MW())


def find_optimal_setpoint(bem: BEM, yaw: float = 0) -> Tuple[float, float, float]:
    def to_opt(x):
        pitch, tsr = x
        return -bem(pitch, tsr, yaw).Cp()

    res = minimize(to_opt, (0, 8))
    pitch, tsr = res.x

    return np.array([pitch, tsr, yaw], dtype=float)


######################################
#   ____ _              _
#  / ___| |_ _ __  _ __(_)_ __ ___   ___
# | |   | __| '_ \| '__| | '_ ` _ \ / _ \
# | |___| |_| |_) | |  | | | | | | |  __/
#  \____|\__| .__/|_|  |_|_| |_| |_|\___|
#           |_|
######################################


def find_Ctprime_root(bem, Ctprime_target, x0, dx) -> Tuple[float, float, float]:
    def to_opt(c):
        if np.isnan(c):
            return 999
        setpoint = x0 + c * dx
        return Ctprime_target - bem.solve(*setpoint).Ctprime()

    res = root_scalar(to_opt, x0=-0.3, x1=0.01, maxiter=100)
    if res.converged:
        return x0 + res.root * dx
    else:
        return np.nan, np.nan, np.nan


def find_power_maximising_Ctprime_setpoints(bem, Ctprime_target, yaw=0) -> Tuple[float, float, float]:
    setpoints_opt = find_optimal_setpoint(bem, yaw=yaw)
    # setpoints_opt = np.array([np.deg2rad(-9), 9, yaw])

    def to_opt(angle):
        dx = np.array([0.7 * np.cos(angle), -7 * np.sin(angle), 0])
        setpoint = find_Ctprime_root(bem, Ctprime_target, setpoints_opt, dx)
        if setpoint[0] == np.nan:
            return 999

        return -bem.solve(*setpoint).Cp()

    res = minimize_scalar(to_opt, bounds=(0, np.pi / 2), method="Bounded", options=dict(maxiter=100))
    angle = res.x

    dx = np.array([1 * np.cos(angle), -9 * np.sin(angle), 0])
    setpoint = find_Ctprime_root(bem, Ctprime_target, setpoints_opt, dx)
    return setpoint


def Ctprime_frontier_setpoints(bem, Ctprime_target, yaw=0):
    setpoints_opt = find_optimal_setpoint(bem, yaw=yaw)
    angles = np.linspace(0, 2 * np.pi)

    def func(angle):
        dx = np.array([0.7 * np.cos(angle), -7 * np.sin(angle), 0])
        setpoint = find_Ctprime_root(bem, Ctprime_target, setpoints_opt, dx)
        return setpoint

    out = foreach(func, angles, parallel=False)
    return np.array(out)


def Ctprime_trajectory_setpoints(bem, Ctprime_min=0.01, yaw=0, N=10):
    setpoints_opt = find_optimal_setpoint(bem, yaw=yaw)

    Ctprime_max = bem(*setpoints_opt).Ctprime()
    Ctprime_targets = np.linspace(Ctprime_min, Ctprime_max, N)

    def func(Ctprime_target):
        return find_power_maximising_Ctprime_setpoints(bem, Ctprime_target, yaw=yaw)

    out = foreach(func, Ctprime_targets, parallel=False)
    return np.array(out)


######################################
#   ____ _
#  / ___| |_
# | |   | __|
# | |___| |_
#  \____|\__|
######################################


def find_Ct_root(bem, Ct_target, x0, dx) -> Tuple[float, float, float]:
    def to_opt(c):
        setpoint = x0 + c * dx
        return Ct_target - bem(*setpoint).Ct()

    res = root_scalar(to_opt, x0=0, x1=0.01, maxiter=100)
    if res.converged:
        return x0 + res.root * dx
    else:
        return np.nan, np.nan, np.nan


def find_power_maximising_Ct_setpoints(bem, Ct_target, yaw=0) -> Tuple[float, float, float]:
    setpoints_opt = find_optimal_setpoint(bem, yaw=yaw)
    # setpoints_opt = np.array([np.deg2rad(-9), 9, yaw])

    def to_opt(angle):
        dx = np.array([0.7 * np.cos(angle), -7 * np.sin(angle), 0])
        setpoint = find_Ct_root(bem, Ct_target, setpoints_opt, dx)
        if setpoint[0] == np.nan:
            return 999

        return -bem(*setpoint).Cp()

    res = minimize_scalar(to_opt, bounds=(0, np.pi / 2), method="Bounded", options=dict(maxiter=100))
    angle = res.x

    dx = np.array([1 * np.cos(angle), -9 * np.sin(angle), 0])
    setpoint = find_Ct_root(bem, Ct_target, setpoints_opt, dx)
    return setpoint


def Ct_frontier_setpoints(bem, Ct_target, yaw=0):
    setpoints_opt = find_optimal_setpoint(bem, yaw=yaw)
    angles = np.linspace(0, 2 * np.pi)

    def func(angle):
        dx = np.array([0.7 * np.cos(angle), -7 * np.sin(angle), 0])
        setpoint = find_Ct_root(bem, Ct_target, setpoints_opt, dx)
        return setpoint

    out = foreach(func, angles, parallel=False)
    return np.array(out)


def Ct_trajectory_setpoints(bem, Ct_min=0.01, yaw=0, N=10):
    setpoints_opt = find_optimal_setpoint(bem, yaw=yaw)
    Ct_max = bem.initialize(*setpoints_opt).Ct()
    Ct_targets = np.linspace(Ct_min, Ct_max, N)

    def func(Ct_target):
        return find_power_maximising_Ct_setpoints(bem, Ct_target, yaw=yaw)

    out = foreach(func, Ct_targets, parallel=False)
    return np.array(out)


if __name__ == "__main__":
    setpoints_opt = find_optimal_setpoint(bem, yaw=np.deg2rad(10))
    print(f"pitch: {np.rad2deg(setpoints_opt[0])} deg")
    print(f"TSR: {setpoints_opt[1]}")
    print(f"yaw: {np.rad2deg(setpoints_opt[2])} deg")

    blah = find_Ctprime_root(bem, 1.5, setpoints_opt, np.array([0.1, -0.5, 0]))
    print(blah)

    blah = find_power_maximising_Ctprime_setpoints(bem, 1.5)
    print(blah)

    blah = Ctprime_frontier_setpoints(bem, 1.5)
    print(blah)


#################################################
#  _  __
# | |/ /      ___  _ __ ___   ___  __ _  __ _
# | ' /_____ / _ \| '_ ` _ \ / _ \/ _` |/ _` |
# | . \_____| (_) | | | | | |  __/ (_| | (_| |
# |_|\_\     \___/|_| |_| |_|\___|\__, |\__,_|
#                                 |___/
#################################################


def k_omega_steady_tsr(yaw, bem, Cp_opt, pitch_opt, tsr_opt):
    def torque_control_residual(tsr):
        sol = bem(pitch_opt, tsr, yaw)

        if sol.converged:
            return np.cbrt(sol.Cp() / Cp_opt * tsr_opt**3) - tsr

    converged, tsr = fixedpointiteration(torque_control_residual, 7, relax=0.6, maxiter=200)
    if converged:
        return tsr
    else:
        return None


######################################
#   ____ _
#  / ___| |_
# | |   | __|
# | |___| |_
#  \____|\__|
######################################


def find_u4_root(bem, u4_target, x0, dx) -> Tuple[float, float, float]:
    def to_opt(c):
        setpoint = x0 + c * dx
        return u4_target - bem(*setpoint).u4()

    res = root_scalar(to_opt, x0=0, x1=0.01, maxiter=100)
    if res.converged:
        return x0 + res.root * dx
    else:
        return np.nan, np.nan, np.nan


def find_power_maximising_u4_setpoints(bem, u4_target, yaw=0) -> Tuple[float, float, float]:
    setpoints_opt = find_optimal_setpoint(bem, yaw=yaw)
    # setpoints_opt = np.array([np.deg2rad(-9), 9, yaw])

    def to_opt(angle):
        dx = np.array([0.7 * np.cos(angle), -7 * np.sin(angle), 0])
        setpoint = find_u4_root(bem, u4_target, setpoints_opt, dx)
        if setpoint[0] == np.nan:
            return 999
        return -bem(*setpoint).Cp()

    res = minimize_scalar(to_opt, bounds=(0, np.pi / 2), method="Bounded", options=dict(maxiter=100))
    angle = res.x

    dx = np.array([1 * np.cos(angle), -9 * np.sin(angle), 0])
    setpoint = find_u4_root(bem, u4_target, setpoints_opt, dx)
    return setpoint
