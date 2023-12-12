from typing import Tuple
from itertools import product

import numpy as np
from foreach import foreach
from scipy.optimize import minimize, minimize_scalar, root_scalar
from UnifiedMomentumModel.Utilities.FixedPointIteration import fixedpointiteration

from MITRotor.BEM import BEM, BEMSolution


def find_optimal_setpoint(bem: BEM, yaw: float = 0) -> BEMSolution:
    def to_opt(x):
        pitch, tsr = x
        return -bem(pitch, tsr, yaw).Cp()

    res = minimize(to_opt, (0, 8))
    pitch, tsr = res.x

    return bem(pitch, tsr, yaw)


@fixedpointiteration(max_iter=200, relaxation=0.6)
class KOmega:
    def __init__(self, bem: BEM, sol_opt: BEMSolution):
        self.bem = bem
        self.sol_opt = sol_opt
        self.cp_on_tsr3 = sol_opt.Cp() / sol_opt.tsr**3

    def initial_guess(self, yaw):
        return [7]

    def residual(self, x, yaw):
        tsr = x[0]
        return [np.cbrt(self.bem(self.sol_opt.pitch, tsr, yaw).Cp() / self.cp_on_tsr3) - tsr]

    def post_process(self, sol, yaw) -> BEMSolution:
        BEMsol = self.bem(self.sol_opt.pitch, sol.x[0], yaw)
        return BEMsol


# Unfortunately, BEM has to be global to be parallelised.
_bem, _yaw = None, None


class ContourData:
    def __init__(self, bem: BEM, yaw: float):
        global _bem, _yaw
        _bem = bem
        _yaw = yaw

    def func(self, x):
        pitch, tsr = x
        sol = _bem(pitch, tsr, _yaw)
        return sol

    def __call__(self, pitches, tsrs, parallel=True):
        params = list(product(pitches, tsrs))
        return foreach(self.func, params, parallel=parallel)


def find_power_maximising_Ctprime_setpoints(bem, Ctprime_target, yaw=0) -> BEMSolution:
    sol_opt = find_optimal_setpoint(bem, yaw=yaw)
    setpoints_opt = sol_opt.pitch, sol_opt.tsr, sol_opt.yaw

    def to_opt(angle):
        dx = np.array([0.7 * np.cos(angle), -7 * np.sin(angle), 0])
        setpoint = find_Ctprime_root(bem, Ctprime_target, setpoints_opt, dx)
        if setpoint[0] == np.nan:
            return 999

        return -bem(*setpoint).Cp()

    res = minimize_scalar(to_opt, bounds=(0, np.pi / 2), method="Bounded", options=dict(maxiter=100))
    angle = res.x

    dx = np.array([1 * np.cos(angle), -9 * np.sin(angle), 0])
    setpoint = find_Ctprime_root(bem, Ctprime_target, setpoints_opt, dx)
    return bem(*setpoint)


def find_Ctprime_root(bem, Ctprime_target, x0, dx) -> Tuple[float, float, float]:
    def to_opt(c):
        if np.isnan(c):
            return 999
        setpoint = x0 + c * dx
        return Ctprime_target - bem(*setpoint).Ctprime()

    res = root_scalar(to_opt, x0=-0.3, x1=0.01, maxiter=100)
    if res.converged:
        return x0 + res.root * dx
    else:
        return np.nan, np.nan, np.nan
