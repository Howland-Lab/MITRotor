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
        print(model.compute_induction(8 / 9, yaw = 0.0))
        print(model.compute_induction(8 / 9, yaw = np.deg2rad(10)))
        print(model.compute_induction(8 / 9, yaw = np.deg2rad(20)))
        # assert_almost_equal(model.compute_induction(8 / 9, 0.1), 1 / 3)
        assert False

    def test_edge(self):
        model = HeckMomentum()
        assert np.isnan(model.compute_induction(1.001, 0.0))

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
        print(model.compute_induction(8 / 9, 0.0))
        print(model.compute_induction(8 / 9, 0.1))
        print(model.compute_induction(8 / 9, 0.2))
        assert False

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
