from MITRotor import BEM, IEA15MW, UnifiedMomentum, BEMGeometry
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

def test_BEM_pre_process():
    rotor = IEA15MW()
    bem = BEM(rotor=rotor)
    bem.pre_process(0.0, 7.0)

def test_BEM_initial_guess():
    rotor = IEA15MW()
    bem = BEM(rotor=rotor)
    bem.pre_process(0.0, 7.0)
    bem.initial_guess(0.0, 7.0)


def test_BEM_residual():
    rotor = IEA15MW()
    bem = BEM(rotor=rotor)
    bem.pre_process(0.0, 7.0)
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

def test_model_yaw_tilt_rotor_phase():
    Nr, Ntheta = 20, 40
    rotor = IEA15MW()
    bem = BEM(rotor=rotor, momentum_model=UnifiedMomentum(averaging="rotor"), geometry = BEMGeometry(Nr=Nr, Ntheta=Ntheta))
    # solve BEM for a control set point.
    yaw = np.deg2rad(20)
    tilt = yaw
    misalignment = calc_eff_yaw(yaw, tilt)
    pitch, tsr = np.deg2rad(0), 7.0

    yaw_sol = bem(pitch, tsr, yaw = misalignment, tilt = 0)
    tilt_sol = bem(pitch, tsr, yaw = 0, tilt = misalignment)
    yaw_tilt_sol = bem(pitch, tsr, yaw = yaw, tilt = tilt)

    for idx in range(5, 10):
        theta_mesh = yaw_sol.geom.theta_mesh[idx, :]
        yaw_val = yaw_sol.Cn(grid = "sector")[idx, :]
        tilt_val = tilt_sol.Cn(grid = "sector")[idx, :]
        yaw_tilt_val = yaw_tilt_sol.Cn(grid = "sector")[idx, :]

        yaw_max_idx = np.argmax(yaw_val)
        tilt_max_idx = np.argmax(tilt_val)
        yaw_and_tilt_max_idx = np.argmax(yaw_tilt_val)

        val_atol, deg_atol = 1e-3, 2 * np.pi / Ntheta
        # all should be at a similar maximum value
        assert np.isclose(yaw_val[yaw_max_idx], tilt_val[tilt_max_idx], atol = val_atol)
        assert np.isclose(yaw_val[yaw_max_idx], yaw_tilt_val[yaw_and_tilt_max_idx], atol = val_atol)
        # tilt and yaw should be offset by 90 degrees
        assert np.isclose(np.abs(theta_mesh[yaw_max_idx] - theta_mesh[tilt_max_idx]), np.pi / 2, atol = deg_atol)
        # yaw and evenly split yaw/tilt should be offset by 45 degrees
        assert np.isclose(np.abs(theta_mesh[yaw_max_idx] - theta_mesh[yaw_and_tilt_max_idx]), np.pi / 4, atol = deg_atol)

def test_BEM_dimensionality():

    rotor = IEA15MW()
    bem = BEM(rotor=rotor)

    # Scalars
    pitch_s = 0.0
    tsr_s   = 7.0
    yaw_s   = 0.0
    tilt_s  = 0.0

    # Vectors (same length)
    pitch_v = np.array([0.0, 0.1, 0.2])
    tsr_v   = np.array([6.0, 7.0, 8.0])
    yaw_v   = np.array([0.0, 0.1, 0.2])
    tilt_v  = np.array([0.0, -0.1, -0.2])

    # Mixed scalar + vector cases
    cases = [
        (pitch_s, tsr_s, yaw_s, tilt_s),              # all scalar
        (pitch_v, tsr_s, yaw_s, tilt_s),              # one vector
        (pitch_s, tsr_v, yaw_s, tilt_s),
        (pitch_s, tsr_s, yaw_v, tilt_s),
        (pitch_s, tsr_s, yaw_s, tilt_v),
        (pitch_v, tsr_v, yaw_v, tilt_v),              # all vectors
    ]

    for pitch, tsr, yaw, tilt in cases:
        sol = bem(pitch, tsr, yaw=yaw, tilt=tilt)

        pitch_a = np.asarray(pitch)
        tsr_a   = np.asarray(tsr)
        yaw_a   = np.asarray(yaw)
        tilt_a  = np.asarray(tilt)

        # Determine expected dimensionality:
        # If any input is vector → vector length defines output dim
        vector_inputs = [x for x in [pitch_a, tsr_a, yaw_a, tilt_a] if x.ndim == 1]

        if len(vector_inputs) == 0:
            expected_ndim = 0
        else:
            # all vectors must have same length
            lengths = [x.shape[0] for x in vector_inputs]
            assert all(l == lengths[0] for l in lengths), "All vector inputs must have same length"
            expected_ndim = 1

        Cp = np.asarray(sol.Cp())
        assert Cp.ndim == expected_ndim or (expected_ndim == 0 and Cp.ndim == 0)
        # Ensure consistency across solution fields
        u4 = np.asarray(sol.u4)
        assert u4.shape == Cp.shape

test_BEM_dimensionality()