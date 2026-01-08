import numpy as np
import pytest
import os
from numpy.testing import assert_almost_equal, assert_allclose
from floris import FlorisModel, TimeSeries
from MITRotor.FlorisInterface import FlorisInterface
from MITRotor.FlorisInterface.FlorisInterface import default_bem_factory, default_pitch_csv, default_tsr_csv
from MITRotor.FlorisInterface.FlorisInterface import MITRotorTurbine, csv_to_interp


def test_pitch_tsr_interpolation():
    # get 15MW CSV file paths
    module_dir = os.path.dirname(__file__)  # tests/
    pitch_csv = os.path.join(module_dir, "..", "MITRotor", "FlorisInterface", "pitch_15mw.csv")
    tsr_csv   = os.path.join(module_dir, "..", "MITRotor", "FlorisInterface", "tsr_15mw.csv")
    pitch_csv = os.path.abspath(pitch_csv)
    tsr_csv   = os.path.abspath(tsr_csv)

    # create interpolators
    tsr_interp = csv_to_interp(tsr_csv)
    pitch_interp = csv_to_interp(pitch_csv)

    # load raw CSV data
    tsr_data = np.loadtxt(tsr_csv, delimiter=",", skiprows=1)
    pitch_data = np.loadtxt(pitch_csv, delimiter=",", skiprows=1)
    tsr_ws, tsr_vals = tsr_data[:, 0], tsr_data[:, 1]
    pitch_ws, pitch_vals = pitch_data[:, 0], pitch_data[:, 1]

    # interpolator reproduces raw data
    assert_allclose(tsr_interp(tsr_ws), tsr_vals, rtol=1e-12, atol=1e-12)
    assert_allclose(pitch_interp(pitch_ws), pitch_vals, rtol=1e-12, atol=1e-12)

    # reasonable values
    x_interp_vals = np.linspace(0.0, 25.0, 100)
    tsr_interp_vals = tsr_interp(x_interp_vals)
    pitch_interp_vals = pitch_interp(x_interp_vals)
    assert np.all(np.isfinite(tsr_interp_vals))
    assert np.all(np.isfinite(pitch_interp_vals))
    assert np.all(tsr_interp_vals > 0.0)
    assert np.all(pitch_interp_vals >= -10.0)   # deg, loose bound
    assert np.all(pitch_interp_vals <= 40.0)


# compute MITRotor BEM outputs directly
def compute_mitrotor_cp_ct_a(wind_speeds, yaw_deg = 0.0, tilt_deg = 0.0):
    bem_model = default_bem_factory() # default BEM (IEA15MW) used in floris interface
    pitch_interp = csv_to_interp(default_pitch_csv()) # IEA15MW pitch curve
    tsr_interp = csv_to_interp(default_tsr_csv()) # IEA15MW tsr curve

    n = len(wind_speeds)
    Ct = np.empty(n)
    a = np.empty(n) 
    for i, ws in enumerate(wind_speeds):
        pitch = np.deg2rad(pitch_interp(ws))
        tsr = tsr_interp(ws)
        sol = bem_model(pitch, tsr, yaw=np.deg2rad(yaw_deg), tilt = np.deg2rad(tilt_deg))
        Ct[i] = sol.Ct()
        a[i] = sol.a()

    return Ct, a

@pytest.mark.parametrize("n_turbines", [1, 2])
def test_mitrotor_floris_wind_speeds(n_turbines):
    wind_speeds = np.array([6.0, 8.0, 10.0, 12.0])
    wind_dirs = np.full_like(wind_speeds, 270.0)
    turbulence_intensity = np.zeros_like(wind_speeds)

    time_series = TimeSeries(
        wind_speeds=wind_speeds,
        wind_directions=wind_dirs,
        turbulence_intensities=turbulence_intensity,
    )

    layout_x = np.linspace(0.0, 500.0 * (n_turbines - 1), n_turbines)
    layout_y = np.zeros_like(layout_x)

    fmodel = FlorisModel("defaults")
    fmodel.set(layout_x=layout_x, layout_y=layout_y, wind_data=time_series)
    fmodel.set_operation_model(MITRotorTurbine())

    fmodel.run()

    floris_Ct = fmodel.get_turbine_thrust_coefficients()
    floris_a = fmodel.get_turbine_axial_induction_factors()

    mit_Ct, mit_a = compute_mitrotor_cp_ct_a(wind_speeds)

    # First turbine matches MITRotor BEM (<1% error)
    assert_almost_equal(floris_Ct[:, 0], mit_Ct, decimal=2)
    assert_almost_equal(floris_a[:, 0], mit_a, decimal=2)

    if n_turbines > 1:
        # first turbine produces more power than the second (wake effects)
        floris_power = fmodel.get_turbine_powers()
        assert np.all(floris_power[:, 0] > floris_power[:, 1])

        # yawing the first turbine decreases first turbine power and increases second turbine power
        fmodel = FlorisModel("defaults")
        yaw_angles = np.tile(np.array([5.0, 0.0]), (4,1))
        fmodel.set(layout_x=layout_x, layout_y=layout_y, wind_data=time_series, yaw_angles = yaw_angles)
        fmodel.set_operation_model(MITRotorTurbine())
        fmodel.run()
        yawed_floris_power = fmodel.get_turbine_powers()
        assert np.all(floris_power[:, 0] > yawed_floris_power[:, 0])
        assert np.all(floris_power[:, 1] < yawed_floris_power[:, 1])

        # TODO: uncomment if tilt is able to be set for Floris
        # tilting the first turbine decreases first turbine power and increases second turbine power
        # fmodel = FlorisModel("defaults")
        # tilt_angles = np.tile(np.array([5.0, 0.0]), (4,1))
        # fmodel.set(layout_x=layout_x, layout_y=layout_y, wind_data=time_series, tilt_angles = tilt_angles)
        # fmodel.set_operation_model(MITRotorTurbine())
        # fmodel.run()
        # tilted_floris_power = fmodel.get_turbine_powers()
        # assert np.all(floris_power[:, 0] > tilted_floris_power[:, 0])
        # assert np.all(floris_power[:, 1] < tilted_floris_power[:, 1])

        # # yawing and tilting turbines is equivalent
        # assert_almost_equal(yawed_floris_power[:, 0], tilted_floris_power[:, 0], decimal=4)
        # assert_almost_equal(yawed_floris_power[:, 1], tilted_floris_power[:, 1], decimal=4)
