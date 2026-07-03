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
import time

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
from MITRotor.FlorisInterface.ROSCOInterface import get_rosco_control_interps, query_controls

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

        pitch_interp, tsr_interp, rated_rotor_speed = get_rosco_control_interps(temp_yaml, bem)
    return pitch_interp, tsr_interp, rated_rotor_speed

def get_turbine_power_coefficent(fturbine, fmodel, wind_speeds):
    rotor_area = np.pi * fturbine.bem_model.rotor.R**2 
    floris_power = np.squeeze(fmodel.get_turbine_powers())
    floris_Cp =  floris_power / (0.5 * 1.225 * rotor_area * (wind_speeds)**3 * fturbine.eff_ratio)
    return floris_Cp, floris_power / 1e6 # MW

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
    print("Generating new LUT for BEM.")
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True, parents=True)
    cache_file = cache_dir / "lut.csv"
    lut_model = UnifiedMomentumLUT(
        cache_fn=cache_file,
        regenerate=False,
        LUT_Cts=np.linspace(-0.5,1.5,40),
        LUT_yaws=np.linspace(0.0,40.0,40),
    )
    print("LUT for BEM done generating. ")
    bem = BEM(rotor = IEA15MW(), momentum_model = lut_model, geometry = BEMGeometry(Nr=10, Ntheta=20))

    # Load IEA15MW ROSCO parameters to make controllers
    start = time.time()
    rosco_yaml_PS_Mode_3 = "MITRotor/ReferenceTurbines/ROSCO_IEA15MW.yaml"
    rosco_PS_Mode_3_pitch_interp, rosco_PS_Mode_3_tsr_interp, rosco_PS_Mode_3_rated_rotorspeed = get_rosco_control_interps(
        rosco_yaml_PS_Mode_3, bem,
        regenerate = False, save_control_file = "control.csv")
    end = time.time()
    print(f"Time to make control CSV: {end - start}")

    # rosco_PS_Mode_0_pitch_interp, rosco_PS_Mode_0_tsr_interp, rosco_PS_Mode_0_rated_rotorspeed = change_control_param("PS_Mode", 0, rosco_yaml_PS_Mode_3, bem)
        
    # Plot IEA15MW control from ROSCO paper, ROSOC control with PS_Mode = 3, and ROSOC control with PS_Mode = 0
    pitch_abbas = abbas_pitch_interp(wind_speeds)
    # pitch_ps0 = np.rad2deg(rosco_PS_Mode_0_pitch_interp(wind_speeds))
    pitch_ps3 = np.rad2deg(query_controls(rosco_PS_Mode_3_pitch_interp, wind_speeds, 0.0))

    tsr_abbas = abbas_tsr_interp(wind_speeds)
    # tsr_ps0 = rosco_PS_Mode_0_tsr_interp(wind_speeds)
    tsr_ps3 = query_controls(rosco_PS_Mode_3_tsr_interp, wind_speeds, 0.0)

    # Create figure
    fig, ax = plt.subplots(1, 2, figsize = (10,4), sharey = True, constrained_layout=True)
    fig.suptitle(fr"Setpoint Trajectories")
    # Plot pitch
    # ax[0].plot(wind_speeds, pitch_ps0, label="ROSCO: PS_Mode = 0", lw=3)
    ax[0].plot(wind_speeds, pitch_ps3, label="ROSCO: PS_Mode = 3", lw=3, linestyle = "dashed")
    ax[0].plot(wind_speeds, pitch_abbas, label="Abbas et al. Fig 2", lw=3, linestyle = "dotted")

    ax[0].set_xlabel("Wind Speed [m/s]")
    ax[0].set_ylabel("Pitch [deg]")
    ax[0].set_title("Pitch Schedule")
    ax[0].grid(True)

    # Plot TSR
    # ax[1].plot(wind_speeds, tsr_ps0, label="MITRotor+FLORIS+ROSCO: PS_Mode = 0", lw=3)
    ax[1].plot(wind_speeds, tsr_ps3, label="MITRotor+FLORIS+ROSCO: PS_Mode = 3", lw=3, linestyle = "dashed")
    ax[1].plot(wind_speeds, tsr_abbas, label="Abbas et al. ROSCO Control", lw=3, linestyle = "dotted")

    ax[1].set_xlabel("Wind Speed [m/s]")
    ax[1].set_ylabel("TSR [-]")
    ax[1].set_title("Tip-Speed Ratio (TSR) Schedule")
    ax[1].grid(True)
    ax[1].legend()

    plt.savefig(figdir / "example_8_IEA15mw_controls.png", dpi=300)

    # make FLORIS turbines
    # floris_default_turbine = MITRotorTurbine(
    #     bem_model = bem,
    # )

    # floris_abbas_turbine = MITRotorTurbine(
    #     bem_model =  bem,
    #     pitch_interp = abbas_pitch_interp,
    #     pitch_rad = False,
    #     tsr_interp = abbas_tsr_interp,
    # )

    # floris_PS_Mode_0_turbine = MITRotorTurbine(
    #     bem_model = bem,
    #     pitch_interp = rosco_PS_Mode_0_pitch_interp,
    #     pitch_rad = True,
    #     tsr_interp = rosco_PS_Mode_0_tsr_interp,
    #     rated_rotor_speed = rosco_PS_Mode_0_rated_rotorspeed,
    # )

    floris_PS_Mode_3_turbine = MITRotorTurbine(
        bem_model = bem,
        pitch_interp = rosco_PS_Mode_3_pitch_interp,
        pitch_rad = True,
        tsr_interp = rosco_PS_Mode_3_tsr_interp,
        rated_rotor_speed = rosco_PS_Mode_3_rated_rotorspeed,
    )
    yaw = 0.0 # degrees
    yaw_angles = [[yaw] for _ in np.arange(len(wind_speeds))]
    # fmodel_default = FlorisModel("defaults")
    # fmodel_default.set(layout_x = [0.0], layout_y = [0.0], wind_data = time_series, yaw_angles = yaw_angles)
    # fmodel_default.set_operation_model(floris_default_turbine)
    # fmodel_default.run()
    # Ct_default = fmodel_default.get_turbine_thrust_coefficients()
    # Cp_default, P_default = get_turbine_power_coefficent(floris_default_turbine, fmodel_default, wind_speeds)

    # fmodel_abbas = FlorisModel("defaults")
    # fmodel_abbas.set(layout_x = [0.0], layout_y = [0.0], wind_data = time_series, yaw_angles = yaw_angles)
    # fmodel_abbas.set_operation_model(floris_abbas_turbine)
    # fmodel_abbas.run()
    # Ct_abbas = fmodel_abbas.get_turbine_thrust_coefficients()
    # Cp_abbas, P_abbas = get_turbine_power_coefficent(floris_abbas_turbine, fmodel_abbas, wind_speeds)


    # fmodel_PS_Mode_0 = FlorisModel("defaults")
    # fmodel_PS_Mode_0.set(layout_x = [0.0], layout_y = [0.0], wind_data = time_series, yaw_angles = yaw_angles)
    # fmodel_PS_Mode_0.set_operation_model(floris_PS_Mode_0_turbine)
    # fmodel_PS_Mode_0.run()
    # Ct_PS_Mode_0 = fmodel_PS_Mode_0.get_turbine_thrust_coefficients()
    # Cp_PS_Mode_0, P_PS_Mode_0 = get_turbine_power_coefficent(floris_PS_Mode_0_turbine, fmodel_PS_Mode_0, wind_speeds)

    fmodel_PS_Mode_3 = FlorisModel("defaults")
    fmodel_PS_Mode_3.set(layout_x = [0.0], layout_y = [0.0], wind_data = time_series, yaw_angles = yaw_angles)
    fmodel_PS_Mode_3.set_operation_model(floris_PS_Mode_3_turbine)
    fmodel_PS_Mode_3.run() 
    Ct_PS_Mode_3 = fmodel_PS_Mode_3.get_turbine_thrust_coefficients()
    Cp_PS_Mode_3, P_PS_Mode_3 = get_turbine_power_coefficent(floris_PS_Mode_3_turbine, fmodel_PS_Mode_3, wind_speeds)

    # plot CT values against one another and against IEA15MW from figure 3.1-C (https://docs.nlr.gov/docs/fy20osti/75698.pdf)
    fig, (ax1, ax2, ax3) = plt.subplots(ncols = 3, sharey = False, figsize = (16,4), constrained_layout=True)
    fig.suptitle(fr"$C_T$ and $C_P$ with Yaw = {yaw} for Different Control Strategies")
    # ax1.plot(
    #     wind_speeds, Ct_default, label="FLORIS+MITROTOR + Gaertner et al. Control",
    #     lw=3, linestyle = "solid", zorder = 1,
    # )
    # ax1.plot(
    #     wind_speeds, Ct_abbas, label="FLORIS+MITROTOR Abbas et al. Control",
    #     lw=3, linestyle = "solid", zorder = 1,
    # )
    # ax1.plot(
    #     wind_speeds, Ct_PS_Mode_0, label="FLORIS+MITROTOR+ROSCO: PS_Mode = 0",
    #     lw=3, linestyle = "dashed", zorder = 1,
    # )
    ax1.plot(wind_speeds, Ct_PS_Mode_3, label="FLORIS+MITROTOR+ROSCO: PS_Mode = 3",
        lw=3, linestyle = "dotted", zorder = 1,
    )
    ax1.set_xlabel("Wind Speed [m/s]")
    ax1.set_ylabel("$C_T [-]$")
    ax1.set_title("$C_T$")
    ax1.grid(True)
    ax1.legend()

    # ax2.plot(
    #     wind_speeds, Cp_default, label="Gaertner et al. IEA15MW Ref",
    #     lw=3, linestyle = "solid", zorder = 1,
    # )
    # ax2.plot(
    #     wind_speeds, Cp_abbas, label="Abbas et al. ROSCO Control",
    #     lw=3, linestyle = "solid", zorder = 1,
    # )
    # ax2.plot(
    #     wind_speeds, Cp_PS_Mode_0, label="FLORIS + MITROTOR + ROSCO: PS_Mode = 0",
    #     lw=3, linestyle = "dashed", zorder = 1,
    # )
    ax2.plot(wind_speeds, Cp_PS_Mode_3, label="FLORIS + MITROTOR + ROSCO: PS_Mode = 3",
        lw=3, linestyle = "dotted", zorder = 1,
    )
    ax2.set_xlabel("Wind Speed [m/s]")
    ax2.set_ylabel("$C_P [-]$")
    ax2.set_title("$C_P$")
    ax2.grid(True)
    ax2.legend()

    # ax3.plot(
    #     wind_speeds, P_default, label="Gaertner et al. IEA15MW Ref",
    #     lw=3, linestyle = "solid", zorder = 1,
    # )
    # ax3.plot(
    #     wind_speeds, P_abbas, label="Abbas et al. ROSCO Control",
    #     lw=3, linestyle = "solid", zorder = 1,
    # )
    # ax3.plot(
    # #     wind_speeds, P_PS_Mode_0, label="FLORIS + MITROTOR + ROSCO: PS_Mode = 0",
    # #     lw=3, linestyle = "dashed", zorder = 1,
    # # )
    ax3.plot(wind_speeds, P_PS_Mode_3, label="FLORIS + MITROTOR + ROSCO: PS_Mode = 3",
        lw=3, linestyle = "dotted", zorder = 1,
    )
    ax3.axhline(y = 15, lw=3, linestyle = "dotted", zorder = 0, color = "k", label = "15MW")
    ax3.set_xlabel("Wind Speed [m/s]")
    ax3.set_ylabel("$Power [MW]$")
    ax3.set_title("$Power$")
    ax3.grid(True)
    ax3.legend(loc = 'lower left')

    plt.savefig(figdir / "example_8_IEA15mw_CT_CP.png", dpi=300)

    # plot the difference in CT/CP curves for different yaw values
    yaw_list = [0.0, 10.0, 20.0] # degrees
    fig, (ax1, ax2, ax3) = plt.subplots(ncols = 3, sharey = False, figsize = (10,4), constrained_layout=True)
    fig.suptitle(fr"$C_T$ and $C_P$ under Yaw for FLORIS + MITROTOR + ROSCO: PS_Mode = 3")
    for (i, yaw) in enumerate(yaw_list):
        yaw_angles = [[yaw] for _ in np.arange(len(wind_speeds))]
        fmodel_PS_Mode_3 = FlorisModel("defaults")
        fmodel_PS_Mode_3.set(layout_x = [0.0], layout_y = [0.0], wind_data = time_series, yaw_angles = yaw_angles)
        fmodel_PS_Mode_3.set_operation_model(floris_PS_Mode_3_turbine)
        fmodel_PS_Mode_3.run() 
        ct = fmodel_PS_Mode_3.get_turbine_thrust_coefficients()
        cp, p = get_turbine_power_coefficent(floris_PS_Mode_3_turbine, fmodel_PS_Mode_3, wind_speeds)

        ax1.plot(wind_speeds, ct, label=fr"Yaw = ${yaw}^\circ$",
            lw=3, linestyle = "solid", zorder = 1,
        )
        ax2.plot(wind_speeds, cp, label=fr"Yaw = ${yaw}^\circ$",
            lw=3, linestyle = "solid", zorder = 1,
        )
        ax3.plot(wind_speeds, p, label=fr"Yaw = ${yaw}^\circ$",
            lw=3, linestyle = "solid", zorder = 1,
        )

    ax1.set_xlabel("Wind Speed [m/s]")
    ax1.set_ylabel("$C_T [-]$")
    ax1.set_title("$C_T$")
    ax1.grid(True)

    ax2.set_xlabel("Wind Speed [m/s]")
    ax2.set_ylabel("$C_P [-]$")
    ax2.set_title("$C_P$")
    ax2.grid(True)

    ax3.axhline(y = 15, lw=3, linestyle = "dotted", zorder = 0, color = "k", label = "15MW")
    ax3.set_xlabel("Wind Speed [m/s]")
    ax3.set_ylabel("$Power [MW]$")
    ax3.set_title("$Power$")
    ax3.grid(True)
    ax3.legend()
    plt.savefig(figdir / "example_8_IEA15mw_CT_CP_yawed.png", dpi=300)


if __name__ == "__main__":
    main()