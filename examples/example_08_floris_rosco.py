# Python modules
import os
import matplotlib.pyplot as plt
import tempfile
from pathlib import Path
from ruamel.yaml import YAML
import numpy as np
import pandas as pd
import polars as pl
from scipy.interpolate import interp1d

# FLORIS modules
from floris import FlorisModel, TimeSeries

# ROSCO toolbox modules 
from rosco.toolbox import controller as ROSCO_controller
from rosco.toolbox import turbine as ROSCO_turbine
from rosco.toolbox.inputs.validation import load_rosco_yaml

# MITRotor modules
from MITRotor.Momentum import UnifiedMomentumLUT
from MITRotor import BEM, BEMGeometry, IEA15MW
from MITRotor.FlorisInterface.FlorisInterface import MITRotorTurbine, default_bem_factory, default_pitch_interp, default_tsr_interp
from MITRotor.FlorisInterface.ROSCOInterface import get_rosco_control_interps

figdir = Path("fig")

def change_control_param(param_key, param_value, template_yaml, bem):
    yaml = YAML()
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_yaml = Path(tmpdir) / "rosco_temp.yaml"

        with open(template_yaml) as f:
            data = yaml.load(f)

        data["controller_params"][param_key] = param_value

        with open(temp_yaml, "w") as f:
            yaml.dump(data, f)

        pitch_interp, tsr_interp = get_rosco_control_interps(temp_yaml, bem)
    return pitch_interp, tsr_interp

def get_turbine_power_coefficent(bem, fmodel, wind_speeds):
    rotor_area = np.pi * bem.rotor.R**2 
    floris_power = np.squeeze(fmodel.get_turbine_powers())
    floris_Cp =  floris_power / (0.5 * 1.225 * rotor_area * (wind_speeds)**3)
    return floris_Cp

def main():
    # wind condtions
    wind_speeds = np.linspace(3, 25, 25)
    wind_dirs = np.full_like(wind_speeds, 270.0)
    turbulence_intensity = np.zeros_like(wind_speeds)
    time_series = TimeSeries(
        wind_speeds=wind_speeds,
        wind_directions=wind_dirs,
        turbulence_intensities=turbulence_intensity,
    )
    
    # load example trajectories from ROSCO Paper Figure 2:
    #   Abbas, N. J., Zalkind, D. S., Pao, L., & Wright, A. (2022).
    #   A reference open-source controller for fixed and floating offshore wind turbines.
    #   Wind Energy Science, 7(1), 53-73.
    abbas_pitch = pd.read_csv("examples/examples_in/IEA15_pitch.csv") # note that this is in degrees
    abbas_tsr = pd.read_csv("examples/examples_in/IEA15_TSR.csv")
    abbas_pitch_interp = interp1d(abbas_pitch["x"], abbas_pitch["y"], kind="linear", fill_value="extrapolate", bounds_error=False)
    abbas_tsr_interp = interp1d(abbas_tsr["x"], abbas_tsr["y"], kind="linear", fill_value="extrapolate", bounds_error=False)

    # create new MITRotor BEM model with a LUT for UMM
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True, parents=True)
    cache_file = cache_dir / "lut.csv"
    lut_model = UnifiedMomentumLUT(
        cache_fn=cache_file,
        regenerate=False,
        LUT_Cts=np.linspace(-0.5,1.5,40),
        LUT_yaws=np.linspace(0.0,20.1,20),
    )
    bem = BEM(rotor = IEA15MW(), momentum_model = lut_model, geometry = BEMGeometry(Nr=10, Ntheta=20))

    # Load IEA15MW ROSCO parameters to make controllers
    rosco_yaml_PS_Mode_3 = "MITRotor/ReferenceTurbines/ROSCO_IEA15MW.yaml"
    rosco_PS_Mode_3_pitch_interp, rosco_PS_Mode_3_tsr_interp = get_rosco_control_interps(rosco_yaml_PS_Mode_3, bem)
    rosco_PS_Mode_0_pitch_interp, rosco_PS_Mode_0_tsr_interp = change_control_param("PS_Mode", 0, rosco_yaml_PS_Mode_3, bem)
        
    # Plot IEA15MW control from ROSCO paper, ROSOC control with PS_Mode = 3, and ROSOC control with PS_Mode = 0
    pitch_abbas = abbas_pitch_interp(wind_speeds)
    pitch_ps0 = np.rad2deg(rosco_PS_Mode_0_pitch_interp(wind_speeds))
    pitch_ps3 = np.rad2deg(rosco_PS_Mode_3_pitch_interp(wind_speeds))

    tsr_abbas = abbas_tsr_interp(wind_speeds)
    tsr_ps0 = rosco_PS_Mode_0_tsr_interp(wind_speeds)
    tsr_ps3 = rosco_PS_Mode_3_tsr_interp(wind_speeds)

    # Create figure
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    # Plot pitch
    ax[0].plot(wind_speeds, pitch_ps0, label="ROSCO: PS_Mode = 0", lw=3)
    ax[0].plot(wind_speeds, pitch_ps3, label="ROSCO: PS_Mode = 3", lw=3, linestyle = "dashed")
    ax[0].plot(wind_speeds, pitch_abbas, label="Abbas et al. Fig 2", lw=3, linestyle = "dotted")

    ax[0].set_xlabel("Wind Speed [m/s]")
    ax[0].set_ylabel("Pitch [deg]")
    ax[0].set_title("Pitch Schedule")
    ax[0].grid(True)
    ax[0].legend()

    # Plot TSR
    ax[1].plot(wind_speeds, tsr_ps0, label="ROSCO: PS_Mode = 0", lw=3)
    ax[1].plot(wind_speeds, tsr_ps3, label="ROSCO: PS_Mode = 3", lw=3, linestyle = "dashed")
    ax[1].plot(wind_speeds, tsr_abbas, label="Abbas et al. Fig 2", lw=3, linestyle = "dotted")

    ax[1].set_xlabel("Wind Speed [m/s]")
    ax[1].set_ylabel("TSR [-]")
    ax[1].set_title("Tip-Speed Ratio (TSR) Schedule")
    ax[1].grid(True)
    ax[1].legend()

    plt.savefig(figdir / "example_8_IEA15mw_controls.png", dpi=300)

    # make FLORIS turbines
    floris_default_turbine = MITRotorTurbine(
        bem_model = bem,
    )

    floris_abbas_turbine = MITRotorTurbine(
        bem_model =  bem,
        pitch_interp = abbas_pitch_interp,
        pitch_rad = False,
        tsr_interp = abbas_tsr_interp,
    )

    floris_PS_Mode_0_turbine = MITRotorTurbine(
        bem_model = bem,
        pitch_interp = rosco_PS_Mode_0_pitch_interp,
        pitch_rad = True,
        tsr_interp = rosco_PS_Mode_0_tsr_interp,
    )

    floris_PS_Mode_3_turbine = MITRotorTurbine(
        bem_model = bem,
        pitch_interp = rosco_PS_Mode_3_pitch_interp,
        pitch_rad = True,
        tsr_interp = rosco_PS_Mode_3_tsr_interp,
    )

    fmodel_default = FlorisModel("defaults")
    fmodel_default.set(layout_x = [0.0], layout_y = [0.0], wind_data = time_series)
    fmodel_default.set_operation_model(floris_default_turbine)
    fmodel_default.run()
    Ct_default = fmodel_default.get_turbine_thrust_coefficients()
    Cp_default = get_turbine_power_coefficent(bem, fmodel_default, wind_speeds)

    fmodel_abbas = FlorisModel("defaults")
    fmodel_abbas.set(layout_x = [0.0], layout_y = [0.0], wind_data = time_series)
    fmodel_abbas.set_operation_model(floris_abbas_turbine)
    fmodel_abbas.run()
    Ct_abbas = fmodel_abbas.get_turbine_thrust_coefficients()
    Cp_abbas = get_turbine_power_coefficent(bem, fmodel_abbas, wind_speeds)

    fmodel_PS_Mode_0 = FlorisModel("defaults")
    fmodel_PS_Mode_0.set(layout_x = [0.0], layout_y = [0.0], wind_data = time_series)
    fmodel_PS_Mode_0.set_operation_model(floris_PS_Mode_0_turbine)
    fmodel_PS_Mode_0.run()
    Ct_PS_Mode_0 = fmodel_PS_Mode_0.get_turbine_thrust_coefficients()
    Cp_PS_Mode_0 = get_turbine_power_coefficent(bem, fmodel_PS_Mode_0, wind_speeds)

    fmodel_PS_Mode_3 = FlorisModel("defaults")
    fmodel_PS_Mode_3.set(layout_x = [0.0], layout_y = [0.0], wind_data = time_series)
    fmodel_PS_Mode_3.set_operation_model(floris_PS_Mode_3_turbine)
    fmodel_PS_Mode_3.run() 
    Ct_PS_Mode_3 = fmodel_PS_Mode_3.get_turbine_thrust_coefficients()
    Cp_PS_Mode_3 = get_turbine_power_coefficent(bem, fmodel_PS_Mode_3, wind_speeds)

    # plot CT values against one another and against IEA15MW from figure 3.1-C (https://docs.nrel.gov/docs/fy20osti/75698.pdf)
    fig, (ax1, ax2) = plt.subplots(ncols = 2, sharey = True, figsize = (12, 6))
    ax1.plot(
        wind_speeds, Ct_default, label="Default Control",
        linewidth=2, linestyle = "solid", zorder = 1,
    )
    ax1.plot(
        wind_speeds, Ct_abbas, label="Abbas et al. Control",
        linewidth=2, linestyle = "solid", zorder = 1,
    )
    ax1.plot(
        wind_speeds, Ct_PS_Mode_0, label="ROSCO PS_Mode = 0",
        linewidth=2, linestyle = "dashed", zorder = 1,
    )
    ax1.plot(wind_speeds, Ct_PS_Mode_3, label="ROSCO PS_Mode = 3",
        linewidth=2, linestyle = "dotted", zorder = 1,
    )
    ax1.set_xlabel("Wind Speed [m/s]")
    ax1.set_ylabel("$C_T$")
    ax1.tick_params()
    ax1.set_title("$C_T$")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(
        wind_speeds, Cp_default, label="Default Control",
        linewidth=2, linestyle = "solid", zorder = 1,
    )
    ax2.plot(
        wind_speeds, Cp_abbas, label="Abbas et al. Control",
        linewidth=2, linestyle = "solid", zorder = 1,
    )
    ax2.plot(
        wind_speeds, Cp_PS_Mode_0, label="ROSCO PS_Mode = 0",
        linewidth=2, linestyle = "dashed", zorder = 1,
    )
    ax2.plot(wind_speeds, Cp_PS_Mode_3, label="ROSCO PS_Mode = 3",
        linewidth=2, linestyle = "dotted", zorder = 1,
    )
    ax2.set_xlabel("Wind Speed [m/s]")
    ax2.set_ylabel("$C_P$")
    ax2.tick_params()
    ax2.set_title("$C_P$")
    ax2.grid(True)
    ax2.legend()

    plt.savefig(figdir / "example_8_IEA15mw_CT_CP.png", dpi=300)



if __name__ == "__main__":
    main()