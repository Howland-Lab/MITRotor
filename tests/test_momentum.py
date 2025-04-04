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

    @pytest.mark.skip("test not implemented")
    def test_scalar_yawed(self):
        ...

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
    @pytest.mark.skip("test not implemented")
    def test_default_args(self):
        model = UnifiedMomentum()
        assert_almost_equal(model.ac, 1 / 3)
        assert_almost_equal(model.v4_correction, 1)
        model.averaging == "rotor"

    @pytest.mark.skip("test not implemented")
    def test_scalar(self):
        model = UnifiedMomentum()
        assert_almost_equal(model.compute_induction(8 / 9, 0.0), 1 / 3)

    @pytest.mark.skip("test not implemented")
    def test_scalar_yawed(self):
        ...

    @pytest.mark.skip("test not implemented")
    def test_edge(self):
        model = UnifiedMomentum()
        assert np.isnan(model.compute_induction(1.001, 0.0))

    @pytest.mark.skip("test not implemented")
    def test_vector(self):
        model = UnifiedMomentum()
        Ct_vec = np.array([0.1, 0.2, 0.3])
        ans = model.compute_induction(Ct_vec, 0.0)
        ans_expected = [0.02565835, 0.0527864, 0.08166999]

        assert_almost_equal(ans, ans_expected)


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
        assert np.isnan(model.compute_induction(1.001, 0.0))

    @pytest.mark.skip("test not implemented")
    def test_vector(self):
        model = MadsenMomentum()
        Ct_vec = np.array([0.1, 0.2, 0.3])
        ans = model.compute_induction(Ct_vec, 0.0)
        ans_expected = [0.02565835, 0.0527864, 0.08166999]

        assert_almost_equal(ans, ans_expected)
