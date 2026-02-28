import numpy as np
import polars as pl
import pytest
from pathlib import Path

from MITRotor.CachedLUT import CachedLUT
from MITRotor.Momentum import ThrustBasedUnifiedLUT, UnifiedMomentumLUT, UnifiedMomentum
from UnifiedMomentumModel import Momentum as UMM


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

class DummyLUT(CachedLUT):
    """Minimal concrete LUT for testing cache + interpolation."""
    calls = 0

    def generate_table(self) -> pl.DataFrame:
        DummyLUT.calls += 1

        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 6)

        rows = []
        for xi in x:
            for yi in y:
                rows.append(
                    {
                        "x": xi,
                        "y": yi,
                        "z": xi + yi,
                        "w": xi * yi,   # second surface
                    }
                )
        return pl.DataFrame(rows)

# ------------------------------------------------------------------
# CachedLUT caching behavior
# ------------------------------------------------------------------

def test_lut_generates_and_saves(tmp_path):
    cache_file = tmp_path / "lut.csv"

    lut = DummyLUT(
        key1="x",
        key2="y",
        to_interp=["z", "w"],
        cache_fn=cache_file,
        regenerate=True,
    )

    assert cache_file.exists()
    assert isinstance(lut.df, pl.DataFrame)


def test_lut_loads_from_cache(tmp_path):
    cache_file = tmp_path / "lut.csv"

    DummyLUT(
        key1="x",
        key2="y",
        to_interp=["z", "w"],
        cache_fn=cache_file,
        regenerate=True,
    )

    lut = DummyLUT(
        key1="x",
        key2="y",
        to_interp=["z", "w"],
        cache_fn=cache_file,
        regenerate=False,
    )

    assert isinstance(lut.df, pl.DataFrame)

# ------------------------------------------------------------------
# Interpolator creation
# ------------------------------------------------------------------

def test_interpolators_created_for_all_fields(tmp_path):
    cache_file = tmp_path / "lut.csv"

    lut = DummyLUT(
        key1="x",
        key2="y",
        to_interp=["z", "w"],
        cache_fn=cache_file,
        regenerate=True,
    )

    assert set(lut.interpolators.keys()) == {"z", "w"}

# ------------------------------------------------------------------
# Interpolation correctness
# ------------------------------------------------------------------

def test_interpolator_reproduces_surfaces(tmp_path):
    cache_file = tmp_path / "lut.csv"

    lut = DummyLUT(
        key1="x",
        key2="y",
        to_interp=["z", "w"],
        cache_fn=cache_file,
        regenerate=True,
        s=0.0,
    )

    z_val = lut.interpolators["z"](0.3, 0.4, grid=False)
    w_val = lut.interpolators["w"](0.3, 0.4, grid=False)

    assert np.isclose(z_val, 0.3 + 0.4, atol=1e-3)
    assert np.isclose(w_val, 0.3 * 0.4, atol=1e-3)

# ------------------------------------------------------------------
# Regenerate flag behavior
# ------------------------------------------------------------------

def test_regenerate_forces_new_generation(tmp_path):
    cache_file = tmp_path / "lut.csv"

    DummyLUT.calls = 0

    DummyLUT(
        key1="x",
        key2="y",
        to_interp=["z"],
        cache_fn=cache_file,
        regenerate=True,
    )

    DummyLUT(
        key1="x",
        key2="y",
        to_interp=["z"],
        cache_fn=cache_file,
        regenerate=True,
    )

    assert DummyLUT.calls == 2


def test_no_regenerate_uses_cache(tmp_path):
    cache_file = tmp_path / "lut.csv"

    DummyLUT.calls = 0

    DummyLUT(
        key1="x",
        key2="y",
        to_interp=["z"],
        cache_fn=cache_file,
        regenerate=True,
    )

    DummyLUT(
        key1="x",
        key2="y",
        to_interp=["z"],
        cache_fn=cache_file,
        regenerate=False,
    )

    assert DummyLUT.calls == 1

# ------------------------------------------------------------------
# Test ThrustBasedUnifiedLUT behavior and correctness
# ------------------------------------------------------------------
# @pytest.fixture(scope="module")
# def lut(tmp_path_factory):
#     tmp_dir = tmp_path_factory.mktemp("lut_cache")
#     cache_file = tmp_dir / "lut.csv"

#     return ThrustBasedUnifiedLUT(
#         cache_fn=cache_file,
#         regenerate=True,
#         s=0,
#         LUT_Cts=np.linspace(-0.5, 1.5, 30),
#         LUT_yaws=np.linspace(-15, 15, 30),
#     )

def test_generate_table_shape(tmp_path):
    model = ThrustBasedUnifiedLUT(
        cache_fn=tmp_path / "lut.csv",
        regenerate=True,
        LUT_Cts=np.linspace(0, 1, 5),
        LUT_yaws=np.linspace(-10.0, 10.1, 5),
    )

    df = model.generate_table()

    assert df.shape[0] == 25
    assert set(df.columns) == {"eff_yaw", "Ctprime", "Cp", "Ct", "an", "u4", "v4", "w4", "x0", "dp", "dp_NL"}

# @pytest.mark.parametrize("Ct,yaw,tilt", [
#     (0.2, np.deg2rad(0.0), np.deg2rad(0.0)),
#     (0.2, np.deg2rad(0.0), np.deg2rad(10.0)),
#     (0.8, np.deg2rad(8.0), np.deg2rad(0.0)),
#     (1.0, np.deg2rad(-8.0), np.deg2rad(6.0)),
#     (1.0, np.deg2rad(-6.0), np.deg2rad(-8.0)),
# ])
# def test_lut_matches_unified_model(lut, Ct, yaw, tilt):
#     ref_model = UMM.ThrustBasedUnified()
#     sol_ref = ref_model(Ct, yaw, tilt = tilt)
#     sol_lut = lut(Ct, yaw, tilt = tilt)
#     atol = 1e-2
#     np.testing.assert_allclose(sol_lut.an, sol_ref.an, atol=atol)
#     np.testing.assert_allclose(sol_lut.Cp, sol_ref.Cp, atol=atol)
#     np.testing.assert_allclose(sol_lut.u4, sol_ref.u4, atol=atol)
#     np.testing.assert_allclose(sol_lut.v4, sol_ref.v4, atol=atol)

# def test_compute_induction_rotor(tmp_path):
#     model = UnifiedMomentum()
#     modelLUT = UnifiedMomentumLUT(
#         s = 0,
#         cache_fn=tmp_path / "lut.csv",
#         regenerate=False, # uses the same LUT as previous test
#     )

#     Ct = np.array([0.8])
#     yaw = np.deg2rad(12.0)
#     tilt = np.deg2rad(4.0)

#     an_ref = model.compute_induction(Ct, yaw, tilt=tilt)
#     an_lut = modelLUT.compute_induction(Ct, yaw, tilt=tilt)

#     u4_ref, v4_ref, w4_ref = model.compute_initial_wake_velocities(Ct, yaw, tilt = tilt)
#     u4_lut, v4_lut, w4_lut = modelLUT.compute_initial_wake_velocities(Ct, yaw, tilt = tilt)

#     atol = 1e-2
#     np.testing.assert_allclose(an_lut, an_ref, atol=atol)
#     np.testing.assert_allclose(u4_lut, u4_ref, atol=atol)
#     np.testing.assert_allclose(v4_lut, v4_ref, atol=atol)
#     np.testing.assert_allclose(w4_lut, w4_ref, atol=atol)