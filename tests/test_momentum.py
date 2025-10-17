from MITRotor.Momentum import ClassicalMomentum, HeckMomentum, MadsenMomentum, UnifiedMomentum
from numpy.testing import assert_almost_equal
import numpy as np
import pytest

class Test_ClassicalMomentum:
    def test_scalar(self):
        model = ClassicalMomentum()
        assert_almost_equal(model.compute_induction(8 / 9, 0.0), 1 / 3)
        assert_almost_equal(model.compute_induction(0, 0.0), 0.0)

    def test_edge(self):
        model = ClassicalMomentum()
        assert np.isnan(model.compute_induction(1.001, 0.0))

    def test_vector(self):
        model = ClassicalMomentum()
        Ct_vec = np.array([0.1, 0.2, 0.3])
        ans = model.compute_induction(Ct_vec, 0.0)
        ans_expected = [0.02565835, 0.0527864, 0.08166999]
        assert_almost_equal(ans, ans_expected)


class Test_HeckMomentum:
    def test_default_args(self):
        model = HeckMomentum()
        assert_almost_equal(model.ac, 1 / 3)
        assert_almost_equal(model.v4_correction, 1)
        model.averaging == "rotor"

    def test_scalar(self):
        model = HeckMomentum()
        assert_almost_equal(model.compute_induction(8 / 9, 0.0), 1 / 3)

    def test_scalar_yawed(self):
        model = HeckMomentum()
        u4_no_yaw, v4_no_yaw, _ = model.compute_initial_wake_velocities(8 / 9, 0.0)
        u4_yaw, v4_yaw, _ = model.compute_initial_wake_velocities(8 / 9, 0.2)
        assert u4_no_yaw > u4_yaw
        assert np.abs(v4_yaw) > np.abs(v4_no_yaw)

    def test_edge(self):
        model = HeckMomentum()
        float_sol = model.compute_induction(1.001, 0.0)
        arr_sol = model.compute_induction(np.array([1.001]), 0.0)
        assert not np.isnan(float_sol)
        assert float_sol == arr_sol[0]

    def test_vector(self):
        model = HeckMomentum()
        Ct_vec = np.array([0.1, 0.2, 0.3])
        ans = model.compute_induction(Ct_vec, 0.0)
        ans_expected = [0.02565835, 0.0527864, 0.08166999]
        assert_almost_equal(ans, ans_expected)


class Test_UnifiedMomentum:

    def test_default_args(self):
        model = UnifiedMomentum()
        assert_almost_equal(model.beta, 0.1403)
        assert model.averaging == "rotor"

    def test_scalar(self):
        model = UnifiedMomentum()
        unified_an = model.compute_induction(8 / 9, 0.0)
        # check that at low CT' we are near classical value, but with lower induction
        assert_almost_equal(unified_an, 1 / 3, decimal = 1) and unified_an < (1 / 3)

    def test_scalar_yawed(self):
        model = UnifiedMomentum()
        u4_no_yaw, v4_no_yaw, w4_no_yaw = model.compute_initial_wake_velocities(8 / 9, yaw = 0.0, tilt = 0.0)
        u4_yaw, v4_yaw, w4_yaw = model.compute_initial_wake_velocities(8 / 9, yaw = 0.2, tilt = 0.0)
        assert u4_no_yaw > u4_yaw
        assert np.abs(v4_yaw) > np.abs(v4_no_yaw)
        assert w4_no_yaw == w4_yaw

    def test_scalar_tilted(self):
        model = UnifiedMomentum()
        u4_no_tilt, v4_no_tilt, w4_no_tilt = model.compute_initial_wake_velocities(8 / 9, yaw = 0.0, tilt = 0.0)
        u4_tilt, v4_tilt, w4_tilt = model.compute_initial_wake_velocities(8 / 9, yaw = 0.0, tilt = 0.2)
        assert u4_no_tilt > u4_tilt
        assert np.abs(w4_tilt) > np.abs(w4_no_tilt)
        assert v4_no_tilt == v4_tilt

    def test_edge(self):
        model = UnifiedMomentum()
        # umm should handle above Betz limit
        assert not np.isnan(model.compute_induction(1.001, 0.0))

    def test_vector(self):
        model = UnifiedMomentum()
        Ct_vec = np.array([0.1, 0.2, 0.3])
        ans = model.compute_induction(Ct_vec, 0.0)
        ans_expected = [0.02565835, 0.0527864, 0.08166999]
        # check that at low CT' we are near classical value, but with lower induction
        assert_almost_equal(ans, ans_expected, decimal = 4) and (ans < ans_expected).all()


class Test_MadsenMomentum:
    @pytest.mark.skip("test not implemented")
    def test_scalar(self):
        model = MadsenMomentum()
        assert_almost_equal(model.compute_induction(8 / 9, 0.0), 1 / 3)

    @pytest.mark.skip("test not implemented")
    def test_scalar_yawed(self):
        ...

    @pytest.mark.skip("test not implemented")
    def test_edge(self):
        model = MadsenMomentum()
        assert not np.isnan(model.compute_induction(1.001, 0.0))

    @pytest.mark.skip("test not implemented")
    def test_vector(self):
        model = MadsenMomentum()
        Ct_vec = np.array([0.1, 0.2, 0.3])
        ans = model.compute_induction(Ct_vec, 0.0)
        ans_expected = [0.02565835, 0.0527864, 0.08166999]
        assert_almost_equal(ans, ans_expected)

@pytest.mark.parametrize("model", [ClassicalMomentum(), HeckMomentum(), UnifiedMomentum(), MadsenMomentum()])
def test_types(model):
    CT_float, yaw_float = 8 / 9, np.deg2rad(20)
    an = model.compute_induction(CT_float, yaw_float)
    u4, v4, w4 = model.compute_initial_wake_velocities(CT_float, yaw_float)
    assert isinstance(an, float)
    assert isinstance(u4, float)
    assert isinstance(v4, float)
    assert isinstance(w4, float)