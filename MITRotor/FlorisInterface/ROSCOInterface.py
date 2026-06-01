import numpy as np
from scipy.interpolate import interp1d

from rosco.toolbox import controller as ROSCO_controller
from rosco.toolbox import turbine as ROSCO_turbine
from rosco.toolbox.inputs.validation import load_rosco_yaml

def load_from_mitrotor(
    turbine, bem, refine_cp_surface=False,
    TurbineName = "IEA15MW", GenEff = 95.756, generator_inertia = 1836784,
):
    '''
    Loads rotor performance information by running MITRotor aerodynamic analysis.

    Based on ROSCO function load_from_ccblade:
    https://github.com/NatLabRockies/ROSCO/blob/main/rosco/toolbox/turbine.py#L272

    Args:
        turbine (rosco.toolbox.turbine.Turbine): ROSCO turbine object
        bem (MITRotor.BEM): MITRotor BEM object
        refine_cp_surface (boolean): if true then smooth cp surface, else false
        TurbineName (string): string turbine name; default to "IEA15MW"
        GenEff (float): generator efficiency (0-100); defaults to IEA15MW value 95.756
        generator_inertia (float): generator_inertia; defaults to IEA15MW value 1836784 [kg m^2]

    Returns: turbine with new fields needed for ROSCO controller, including Ct, Cp, and Cq surfaces
    '''
    turbine.TurbineName = TurbineName
    turbine.rotor_performance_filename = 'Cp_Ct_Cq.txt'

    (tower_height, gearbox_efficiency, gearbox_ratio, air_density) = bem.rotor.rosco_values
    turbine.TowerHt = tower_height
    turbine.GBoxEff = gearbox_efficiency
    turbine.Ng = gearbox_ratio
    turbine.rho = air_density

    turbine.GenEff = GenEff
    turbine.generator_inertia = generator_inertia

    turbine.yaw = 0.0
    turbine.rotor_radius = bem.rotor.R
    turbine.J = turbine.rotor_inertia + turbine.generator_inertia * turbine.Ng**2
    turbine.rated_torque = turbine.rated_power/(turbine.GenEff/100*turbine.rated_rotor_speed*turbine.Ng)

    # Generate the look-up tables, mesh the grid and flatten the arrays for cc_rotor aerodynamic analysis
    TSR_initial = np.arange(0.5, 25., 0.5)
    pitch_initial = np.arange(0.0, 31., 1.0)
    pitch_initial_rad = np.deg2rad(pitch_initial)

    tsr_mesh, pitch_rad_mesh = np.meshgrid(TSR_initial, pitch_initial_rad)

    tsr_flat = tsr_mesh.ravel()
    pitch_rad_flat = pitch_rad_mesh.ravel()

    # Get values from MITRotor
    print('Running MITRotor aerodynamic analysis, this may take a minute...')

    mit_sols = bem(
        pitch_rad_flat,
        tsr_flat,
        yaw=np.zeros_like(tsr_flat),
        tilt=np.zeros_like(tsr_flat),
    )

    print('MITRotor aerodynamic analysis run successfully.')

    # Reshape directly into lookup tables
    n_pitch = len(pitch_initial_rad)
    n_tsr = len(TSR_initial)

    Cp_table = mit_sols.Cp().reshape(n_pitch, n_tsr).T
    Ct_table = mit_sols.Ct().reshape(n_pitch, n_tsr).T
    Cq_table = mit_sols.Cq().reshape(n_pitch, n_tsr).T

    # Store necessary metrics for analysis
    turbine.pitch_initial_rad = pitch_initial_rad
    turbine.TSR_initial = TSR_initial

    turbine.Cp_table = Cp_table
    turbine.Ct_table = Ct_table
    turbine.Cq_table = Cq_table

    turbine.Cp = ROSCO_turbine.RotorPerformance(turbine.Cp_table,turbine.pitch_initial_rad,turbine.TSR_initial, refine=refine_cp_surface)
    turbine.Ct = ROSCO_turbine.RotorPerformance(turbine.Ct_table,turbine.pitch_initial_rad,turbine.TSR_initial, refine=refine_cp_surface)
    turbine.Cq = ROSCO_turbine.RotorPerformance(turbine.Cq_table,turbine.pitch_initial_rad,turbine.TSR_initial, refine=refine_cp_surface)
    return turbine

def get_rosco_control_interps(rosco_yaml, bem, TurbineName = None, generator_inertia = None, GenEff = None):
    """
    Creates pitch and tsr 1D interpolators using ROSCO controller tuning.

    Args:
        rosco_yaml (string): yaml file that defineds turbine and control parameters;
            [see MITRotor/ReferenceTurbines/ROSCO_IEA15MW.yaml for example]
        bem (MITRotor.BEM): MITRotor BEM object
        TurbineName (string): string turbine name; default to "IEA15MW"
        GenEff (float): generator efficiency (0-100); defaults to IEA15MW value 95.756
        generator_inertia (float): generator_inertia; defaults to IEA15MW value 1836784 [kg m^2]

    Returns:
        pitch_interp (string): 1D pitch interpolator from wind speed tuned with ROSCO control paramters
        tsr_interp (string): 1D tsr interpolator from wind speed tuned with ROSCO control paramters
    """
    # load ROSCO inputs
    inps = load_rosco_yaml(rosco_yaml)
    turbine_params = inps['turbine_params']
    controller_params = inps['controller_params']

    # make turbine
    turbine = ROSCO_turbine.Turbine(turbine_params)
    turbine = load_from_mitrotor(
        turbine, bem,
        TurbineName = TurbineName, generator_inertia = generator_inertia, GenEff = GenEff,
    )

    # make controller
    controller = ROSCO_controller.Controller(controller_params)
    controller.tune_controller(turbine)

    # get interpolators
    pitch_interp = rosco_pitch_interp(controller)
    tsr_interp = rosco_tsr_interp(controller)

    return pitch_interp, tsr_interp

def rosco_pitch_interp(controller):
    """
    Return 1D pitch interpolator from tuned ROSCO controller
    
    Args:
        controller (rosco.toolbox.controller.Controller)
    """
    wind_table = controller.v
    pitch_table =  np.maximum(controller.pitch_op, controller.ps_min_bld_pitch)
    return interp1d(wind_table, pitch_table, kind="linear", fill_value="extrapolate", bounds_error=False)

def rosco_tsr_interp(controller):
    """
    Return 1D tsr interpolator from tuned ROSCO controller
    
    Args:
        controller (rosco.toolbox.controller.Controller)
    """
    wind_table = controller.v
    tsr_table =  controller.TSR_op
    return interp1d(wind_table, tsr_table, kind="linear", fill_value="extrapolate", bounds_error=False)
