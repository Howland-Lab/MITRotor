# Python modules
import matplotlib.pyplot as plt 
import os 
# ROSCO toolbox modules 
from rosco.toolbox import controller as ROSCO_controller
from rosco.toolbox import turbine as ROSCO_turbine
from rosco.toolbox.utilities import write_DISCON
from rosco.toolbox.inputs.validation import load_rosco_yaml
from rosco.toolbox.utilities import write_rotor_performance

def main():
    from MITRotor.Momentum import UnifiedMomentumLUT
    from MITRotor import BEM, BEMGeometry, NREL5MW, IEA15MW, IEA10MW
    from pathlib import Path
    import pandas as pd

    # load CSVs
    pitch_15 = pd.read_csv("examples/Tune_Cases/IEA15_pitch.csv")
    tsr_15 = pd.read_csv("examples/Tune_Cases/IEA15_TSR.csv")
    pitch_5 = pd.read_csv("examples/Tune_Cases/NLR5_pitch.csv")
    tsr_5 = pd.read_csv("examples/Tune_Cases/NLR5_TSR.csv")

    # calcualte new
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True, parents=True)
    cache_file = cache_dir / "lut.csv"

    # TODO: right now we are using 2 input files that have some shared parameters...
    Nr, Ntheta = 10, 20
    lut_model = UnifiedMomentumLUT(
        cache_fn=cache_file,
        regenerate=False,
        LUT_Cts=np.linspace(-0.5,1.5,40),
        LUT_yaws=np.linspace(0.0,20.1,20),
    )
    # NLR 5MW
    inps_5 = load_rosco_yaml("examples/Tune_Cases/NLR_5MW.yaml")
    turbine_params_5      = inps_5['turbine_params']
    controller_params_5   = inps_5['controller_params']
    rotor_5 = NREL5MW()
    lut_bem_5 = BEM(rotor=rotor_5, momentum_model=lut_model, geometry = BEMGeometry(Nr=Nr, Ntheta=Ntheta))
    turbine_5 = ROSCO_turbine.Turbine(turbine_params_5)
    turbine_5 = load_from_mitrotor(turbine_5, lut_bem_5, generator_inertia=534.116, GenEff=94.4)
    controller_5 = ROSCO_controller.Controller(controller_params_5)
    controller_5.tune_controller(turbine_5)


    # IEA 15MW
    inps_15 = load_rosco_yaml("examples/Tune_Cases/IEA15MW.yaml")
    turbine_params_15      = inps_15['turbine_params']
    controller_params_15   = inps_15['controller_params']
    rotor_15 = IEA15MW()
    lut_bem_15 = BEM(rotor=rotor_15, momentum_model=lut_model, geometry = BEMGeometry(Nr=Nr, Ntheta=Ntheta))
    turbine_15 = ROSCO_turbine.Turbine(turbine_params_15)
    turbine_15 = load_from_mitrotor(turbine_15, lut_bem_15)
    controller_15 = ROSCO_controller.Controller(controller_params_15)
    controller_15.tune_controller(turbine_15)


    plt.figure()
    plt.plot(pitch_5["x"], pitch_5["y"], label = "NLR 5MW (Abbas et al.)", linestyle = "dashed")
    plt.plot(pitch_15["x"], pitch_15["y"], label = "IEA 15MW (Abbas et al.)", linestyle = "dashed")
    plt.plot(controller_5.v, np.rad2deg(np.maximum(controller_5.pitch_op, controller_5.ps_min_bld_pitch)), label='NLR 5MW (Generated)')
    plt.plot(controller_15.v, np.rad2deg(np.maximum(controller_15.pitch_op, controller_15.ps_min_bld_pitch)), label='IEA 15MW (Generated)')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Pitch')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(tsr_5["x"], tsr_5["y"], label = "NLR 5MW (Abbas et al.)", linestyle = "dashed")
    plt.plot(tsr_15["x"], tsr_15["y"], label = "IEA 15MW (Abbas et al.)", linestyle = "dashed")
    plt.plot(controller_5.v, controller_5.TSR_op, label='NLR 5MW (Generated)')
    plt.plot(controller_15.v, controller_15.TSR_op, label='IEA 15MW (Generated)')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('TSR')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()