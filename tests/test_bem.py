from MITRotor import BEM, IEA15MW, UnifiedMomentum
from UnifiedMomentumModel.Utilities.Geometry import calc_eff_yaw
import numpy as np
from pytest import approx

def test_IEA15MW():
    IEA15MW()


def test_BEM_initialise():
    rotor = IEA15MW()
    BEM(rotor=rotor)


def test_default_models():
    rotor = IEA15MW()
    bem = BEM(rotor=rotor)


def test_BEM_initial_guess():
    rotor = IEA15MW()
    bem = BEM(rotor=rotor)
    bem.initial_guess(0.0, 7.0)


def test_BEM_residual():
    rotor = IEA15MW()
    bem = BEM(rotor=rotor)
    x0 = bem.initial_guess(0.0, 7.0)
    bem.residual(x0, 0.0, 7, 0)


def test_BEM_solve():
    rotor = IEA15MW()
    bem = BEM(rotor=rotor)
    sol = bem(0.0, 7.0, 0.0)

    # Check power coefficient is positive and less than Betz limit.
    assert (sol.Cp() > 0) and (sol.Cp() < 16 / 27)

def test_model_yaw_tilt_comparison():
    pitch, tsr = np.deg2rad(0), 7.0
    rotor = IEA15MW()
    momentum_model = UnifiedMomentum()
    bem = BEM(rotor=rotor, momentum_model=momentum_model)
    for (yaw, tilt) in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        eff_angle = calc_eff_yaw(yaw, tilt)
        yaw_sol = bem(pitch, tsr, yaw = eff_angle, tilt = 0.0)
        tilt_sol = bem(pitch, tsr, yaw = 0.0, tilt = eff_angle)
        yaw_tilt_sol = bem(pitch, tsr, yaw = yaw, tilt = tilt)
        # check that yaw and tilt solutions are equivalent up to a -90 degree rotation
        atol = 1e-4
        assert np.isclose(yaw_sol.u4, tilt_sol.u4, atol = atol)
        assert np.isclose(np.abs(yaw_sol.v4), np.abs(tilt_sol.w4), atol = atol)
        assert yaw_sol.w4 == 0 and tilt_sol.v4 == 0
        assert yaw_tilt_sol.v4 != 0 and yaw_tilt_sol.w4 != 0
        assert np.sign(yaw) == np.sign(-1 * yaw_tilt_sol.v4)
        assert np.sign(tilt) == np.sign(yaw_tilt_sol.w4)
        assert np.isclose(np.linalg.norm([yaw_tilt_sol.u4, yaw_tilt_sol.v4, yaw_tilt_sol.w4]), np.linalg.norm([yaw_sol.u4, yaw_sol.v4, yaw_sol.w4]), atol = atol)
        assert np.isclose(yaw_sol.Cp(), tilt_sol.Cp(), atol = atol)
        assert np.isclose(yaw_sol.Cp(), yaw_tilt_sol.Cp(), atol = atol)
        assert np.isclose(yaw_sol.Ct(), tilt_sol.Ct(), atol = atol)
        assert np.isclose(yaw_sol.Ct(), yaw_tilt_sol.Ct(), atol = atol)
        assert np.isclose(yaw_sol.Cl(), tilt_sol.Cl(), atol = atol)
        assert np.isclose(yaw_sol.Cd(), yaw_tilt_sol.Cd(), atol = atol)
        assert np.isclose(yaw_sol.aoa(), tilt_sol.aoa(), atol = atol)
        assert np.isclose(yaw_sol.aoa(), yaw_tilt_sol.aoa(), atol = atol)