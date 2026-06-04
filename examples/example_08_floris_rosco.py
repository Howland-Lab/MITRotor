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
    return floris_Cp, floris_power

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

    # # create new MITRotor BEM model with a LUT for UMM
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True, parents=True)
    cache_file = cache_dir / "lut.csv"
    lut_model = UnifiedMomentumLUT(
        cache_fn=cache_file,
        regenerate=True,
        LUT_Cts=np.linspace(-0.5,1.5,40),
        LUT_yaws=np.linspace(0.0,25.1,25),
    )
    bem = BEM(rotor = IEA15MW(), momentum_model = lut_model, geometry = BEMGeometry(Nr=10, Ntheta=20))

    # # Load IEA15MW ROSCO parameters to make controllers
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
    fig, ax = plt.subplots(1, 2, figsize = (10,4), sharey = True, constrained_layout=True)
    fig.suptitle(fr"Setpoint Trajectories")
    # Plot pitch
    ax[0].plot(wind_speeds, pitch_ps0, label="ROSCO: PS_Mode = 0", lw=3)
    ax[0].plot(wind_speeds, pitch_ps3, label="ROSCO: PS_Mode = 3", lw=3, linestyle = "dashed")
    ax[0].plot(wind_speeds, pitch_abbas, label="Abbas et al. Fig 2", lw=3, linestyle = "dotted")

    ax[0].set_xlabel("Wind Speed [m/s]")
    ax[0].set_ylabel("Pitch [deg]")
    ax[0].set_title("Pitch Schedule")
    ax[0].grid(True)
    # ax[0].legend()

    # Plot TSR
    ax[1].plot(wind_speeds, tsr_ps0, label="MITRotor+FLORIS+ROSCO: PS_Mode = 0", lw=3)
    ax[1].plot(wind_speeds, tsr_ps3, label="MITRotor+FLORIS+ROSCO: PS_Mode = 3", lw=3, linestyle = "dashed")
    ax[1].plot(wind_speeds, tsr_abbas, label="Abbas et al. ROSCO Control", lw=3, linestyle = "dotted")

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
    yaw = 0.0 # degrees
    yaw_angles = [[yaw] for _ in np.arange(len(wind_speeds))]
    fmodel_default = FlorisModel("defaults")
    fmodel_default.set(layout_x = [0.0], layout_y = [0.0], wind_data = time_series, yaw_angles = yaw_angles)
    fmodel_default.set_operation_model(floris_default_turbine)
    fmodel_default.run()
    Ct_default = fmodel_default.get_turbine_thrust_coefficients()
    Cp_default, P_default = get_turbine_power_coefficent(bem, fmodel_default, wind_speeds)

    fmodel_abbas = FlorisModel("defaults")
    fmodel_abbas.set(layout_x = [0.0], layout_y = [0.0], wind_data = time_series, yaw_angles = yaw_angles)
    fmodel_abbas.set_operation_model(floris_abbas_turbine)
    fmodel_abbas.run()
    Ct_abbas = fmodel_abbas.get_turbine_thrust_coefficients()
    Cp_abbas, P_abbas = get_turbine_power_coefficent(bem, fmodel_abbas, wind_speeds)


    fmodel_PS_Mode_0 = FlorisModel("defaults")
    fmodel_PS_Mode_0.set(layout_x = [0.0], layout_y = [0.0], wind_data = time_series, yaw_angles = yaw_angles)
    fmodel_PS_Mode_0.set_operation_model(floris_PS_Mode_0_turbine)
    fmodel_PS_Mode_0.run()
    Ct_PS_Mode_0 = fmodel_PS_Mode_0.get_turbine_thrust_coefficients()
    Cp_PS_Mode_0, P_PS_Mode_0 = get_turbine_power_coefficent(bem, fmodel_PS_Mode_0, wind_speeds)

    fmodel_PS_Mode_3 = FlorisModel("defaults")
    fmodel_PS_Mode_3.set(layout_x = [0.0], layout_y = [0.0], wind_data = time_series, yaw_angles = yaw_angles)
    fmodel_PS_Mode_3.set_operation_model(floris_PS_Mode_3_turbine)
    fmodel_PS_Mode_3.run() 
    Ct_PS_Mode_3 = fmodel_PS_Mode_3.get_turbine_thrust_coefficients()
    Cp_PS_Mode_3, P_PS_Mode_3 = get_turbine_power_coefficent(bem, fmodel_PS_Mode_3, wind_speeds)

    # plot CT values against one another and against IEA15MW from figure 3.1-C (https://docs.nlr.gov/docs/fy20osti/75698.pdf)
    fig, (ax1, ax2, ax3) = plt.subplots(ncols = 3, sharey = False, figsize = (15,4), constrained_layout=True)
    fig.suptitle(fr"$C_T$ and $C_P$ with Yaw = 0 for Different Control Strategies")
    ax1.plot(
        wind_speeds, Ct_default, label="FLORIS+MITROTOR + Gaertner et al. Control",
        lw=3, linestyle = "solid", zorder = 1,
    )
    ax1.plot(
        wind_speeds, Ct_abbas, label="FLORIS+MITROTOR Abbas et al. Control",
        lw=3, linestyle = "solid", zorder = 1,
    )
    ax1.plot(
        wind_speeds, Ct_PS_Mode_0, label="FLORIS+MITROTOR+ROSCO: PS_Mode = 0",
        lw=3, linestyle = "dashed", zorder = 1,
    )
    ax1.plot(wind_speeds, Ct_PS_Mode_3, label="FLORIS+MITROTOR+ROSCO: PS_Mode = 3",
        lw=3, linestyle = "dotted", zorder = 1,
    )
    ax1.set_xlabel("Wind Speed [m/s]")
    ax1.set_ylabel("$C_T [-]$")
    ax1.set_title("$C_T$")
    ax1.grid(True)
    # ax1.legend()

    ax2.plot(
        wind_speeds, Cp_default, label="Gaertner et al. IEA15MW Ref",
        lw=3, linestyle = "solid", zorder = 1,
    )
    ax2.plot(
        wind_speeds, Cp_abbas, label="Abbas et al. ROSCO Control",
        lw=3, linestyle = "solid", zorder = 1,
    )
    ax2.plot(
        wind_speeds, Cp_PS_Mode_0, label="FLORIS + MITROTOR + ROSCO: PS_Mode = 0",
        lw=3, linestyle = "dashed", zorder = 1,
    )
    ax2.plot(wind_speeds, Cp_PS_Mode_3, label="FLORIS + MITROTOR + ROSCO: PS_Mode = 3",
        lw=3, linestyle = "dotted", zorder = 1,
    )
    ax2.set_xlabel("Wind Speed [m/s]")
    ax2.set_ylabel("$C_P [-]$")
    ax2.set_title("$C_P$")
    ax2.grid(True)
    # ax2.legend()

    ax3.plot(
        wind_speeds, P_default, label="Gaertner et al. IEA15MW Ref",
        lw=3, linestyle = "solid", zorder = 1,
    )
    ax3.plot(
        wind_speeds, P_abbas, label="Abbas et al. ROSCO Control",
        lw=3, linestyle = "solid", zorder = 1,
    )
    ax3.plot(
        wind_speeds, P_PS_Mode_0, label="FLORIS + MITROTOR + ROSCO: PS_Mode = 0",
        lw=3, linestyle = "dashed", zorder = 1,
    )
    ax3.plot(wind_speeds, P_PS_Mode_3, label="FLORIS + MITROTOR + ROSCO: PS_Mode = 3",
        lw=3, linestyle = "dotted", zorder = 1,
    )
    ax3.set_xlabel("Wind Speed [m/s]")
    ax3.set_ylabel("$Power [MW]$")
    ax3.set_title("$Power$")
    ax3.grid(True)
    ax3.legend()

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
        cp, p = get_turbine_power_coefficent(bem, fmodel_PS_Mode_3, wind_speeds)

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

    ax3.set_xlabel("Wind Speed [m/s]")
    ax3.set_ylabel("$Power [MW]$")
    ax3.set_title("$Power$")
    ax3.grid(True)
    ax3.legend()
    plt.savefig(figdir / "example_8_IEA15mw_CT_CP_yawed.png", dpi=300)

    # VALUES FROM THE LAST YAW EXAMPLE
    # yaw_angles = [0.0, 10.0, 20.0]
    # wind_speeds = np.array([ 2.99013218,  3.90378368,  4.81743518,  5.73108668,  6.64473818,  7.55838968,
    # 8.47204118,  9.38569268, 10.29934418, 11.21299568, 12.12664717, 13.04029867,
    # 13.95395017, 14.86760167, 15.78125317, 16.69490467, 17.60855617, 18.52220767,
    # 19.43585917, 20.34951067, 21.26316217, 22.17681367, 23.09046517, 24.00411667,
    # 24.91776817])

    # pitch_data = [
    #     np.array([0.06674022, 0.05708731, 0.04527502, 0.02821344, 0.00638814, 0.,
    # 0.,         0.,         0.03561096, 0.06201665, 0.10170293, 0.14064172,
    # 0.17480852, 0.20334416, 0.2279438,  0.24969219, 0.26945148, 0.28830328,
    # 0.3064061,  0.32385969, 0.34076384, 0.35709961, 0.37304778, 0.38862888,
    # 0.40381301]),
    #     np.array([0.0670117,  0.05749029, 0.04596395, 0.02940459, 0.00804692, 0.,
    # 0.,         0.,         0.03159737, 0.05959471, 0.09666413, 0.13512453,
    # 0.17020744, 0.19918221, 0.22414017, 0.2461998,  0.26590456, 0.2847307,
    # 0.30278677, 0.32017919, 0.33707697, 0.35336143, 0.3692476,  0.38476091,
    # 0.39988875]),
    #     np.array([0.06782725, 0.05861179, 0.0480336,  0.03267438, 0.01290243, 0.,
    # 0.,         0.,         0.01790879, 0.05231887, 0.08152841, 0.11886783,
    # 0.15571155, 0.18608686, 0.21210147, 0.23480731, 0.25495804, 0.27378792,
    # 0.29175,    0.30904482, 0.32577372, 0.34195922, 0.35764598, 0.37297828,
    # 0.38797132])
    # ]

    # tsr_data = [
    #     np.array([21.17700633, 16.24293034, 13.15445983, 11.05613061,  9.53582497,  9.,
    # 9.,          9.,          9.,          8.54102737,  7.89840549,  7.34547759,
    # 6.86473567,  6.44293918,  6.06989257,  5.73761929,  5.43979028,  5.17132019,
    # 4.92807656,  4.7066669,   4.50428035,  4.31856823,  4.14761354,  3.98985351,
    # 3.84362968]),
    #     np.array([21.36151077, 16.39563759, 13.28451444, 11.1591271,   9.62802523,  9.,
    # 9.,          9.,          9.,          8.62614707,  7.97636669,  7.41738871,
    # 6.93146656,  6.50518492,  6.12821732,  5.79248716,  5.49158768,  5.22037222,
    # 4.97523891,  4.7520206,   4.54791367,  4.36057184,  4.18801658,  4.02856807,
    # 3.88079097]),
    #     np.array([21.91578937, 16.88295332, 13.67521769, 11.49774122,  9.91488218,  9.00015098,
    # 9.,          9.,          9.,          8.88185921,  8.21215035,  7.63791414,
    # 7.13816236,  6.6993949,   6.31116194,  5.96525606,  5.65514567,  5.37646321,
    # 5.12399624,  4.89401575,  4.6836678,   4.49055784,  4.31266275,  4.14835247,
    # 3.99643498])
    # ]

    # # Create figure
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # # Colors for yaw angles
    # colors = plt.cm.viridis(np.linspace(0, 1, len(yaw_angles)))

    # # Plot pitch vs wind speed
    # for i, yaw in enumerate(yaw_angles):
    #     ax1.plot(wind_speeds, np.rad2deg(pitch_data[i]), marker='o', linewidth=2, 
    #             label=f'Yaw = {yaw}°', color=colors[i])
    # ax1.set_xlabel('Wind Speed (m/s)', fontsize=12)
    # ax1.set_ylabel('Pitch Angle (deg)', fontsize=12)
    # ax1.set_title('Pitch vs Wind Speed', fontsize=13, fontweight='bold')
    # ax1.legend()
    # ax1.grid(True, alpha=0.3)

    # # Plot TSR vs wind speed
    # for i, yaw in enumerate(yaw_angles):
    #     ax2.plot(wind_speeds, tsr_data[i], marker='s', linewidth=2,
    #             label=f'Yaw = {yaw}°', color=colors[i])
    # ax2.set_xlabel('Wind Speed (m/s)', fontsize=12)
    # ax2.set_ylabel('Tip Speed Ratio', fontsize=12)
    # ax2.set_title('TSR vs Wind Speed', fontsize=13, fontweight='bold')
    # ax2.legend()
    # ax2.grid(True, alpha=0.3)

    # plt.tight_layout()
    # plt.savefig(figdir / "example_8_IEA15mw_setpoints_yawed.png", dpi=300)

if __name__ == "__main__":
    main()