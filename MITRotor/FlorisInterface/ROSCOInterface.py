# Python modules
import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import warnings
# Parallelization modules
from types import SimpleNamespace
# ROSCO toolbox modules 
from rosco import discon_lib_path as lib_name
from rosco.toolbox import controller as ROSCO_controller
from rosco.toolbox import turbine as ROSCO_turbine
from rosco.toolbox.utilities import write_rotor_performance, write_DISCON
from rosco.toolbox.inputs.validation import load_rosco_yaml
# MITRotor Imports
import MITRotor.FlorisInterface.InterfaceUtilities as iu

# -----------------------------
# Create Ct/Cp surfaces
# -----------------------------
def load_from_mitrotor(
    turbine, bem,
    refine_cp_surface=False,
    TurbineName = "IEA15MW",
    rotor_performance_filename = 'Cp_Ct_Cq.txt',
    GenEff = 95.756,
    generator_inertia = 1836784,
    yaw = 0.0,
    tilt = 0.0,
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
        yaw (float): yaw with which to make Cp/Ct surface [deg]
        tilt (float): tilt with which to make Cp/Ct surface [deg]
        rotor_performance_filename (string): file name (including path) for saved Cp/Ct/Cq surface

    Returns: turbine with new fields needed for ROSCO controller, including Ct, Cp, and Cq surfaces
    '''
    # Set turbine parameters
    turbine.TurbineName = TurbineName
    turbine.rotor_performance_filename = rotor_performance_filename

    # Save needed paramters that are read into MITRotor turbine definition through ReferenceTurbines files
    (tower_height, gearbox_efficiency, gearbox_ratio, air_density) = bem.rotor.rosco_values
    turbine.TowerHt = tower_height
    turbine.GBoxEff = gearbox_efficiency
    turbine.Ng = gearbox_ratio
    turbine.rho = air_density

    # Save remaining two needed generator parameters --> user needs to pass in! 
    turbine.GenEff = GenEff
    turbine.generator_inertia = generator_inertia

    # yaw and tilt for turbine/controller default Cp/Ct surface
    turbine.yaw = yaw
    turbine.tilt = tilt

    # Calcualte remining needed values
    turbine.rotor_radius = bem.rotor.R
    turbine.J = turbine.rotor_inertia + turbine.generator_inertia * turbine.Ng**2
    turbine.rated_torque = turbine.rated_power/(turbine.GenEff/100*turbine.rated_rotor_speed*turbine.Ng)

    # Generate the look-up tables; mesh the grid and flatten the arrays for aerodynamic analysis
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
        yaw=np.ones_like(tsr_flat) * turbine.yaw,
        tilt=np.ones_like(tsr_flat) * turbine.tilt,
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

    # if TSR_operational is not reccomended, find max Cp TSR from Cp surface
    if not turbine.TSR_operational:
        turbine.TSR_operational = turbine.Cp.TSR_opt

    return turbine

# -----------------------------
# Generate control scheme
# -----------------------------
def get_rosco_control_interps(
    rosco_yaml, bem,
    TurbineName="IEA15MW",
    rotor_performance_filename='Cp_Ct_Cq.txt',
    SimName="Sim1",
    GenEff=95.756,
    generator_inertia=1836784,
    regenerate=False,
    save_control_file=None,
    save_dir=".",
    dt=0.05,
    yaw_grid_deg=np.arange(0.0, 25.0, 1.0),  # degrees
    n_jobs=-1, # parallel workers across wind speeds
):
    """
    Creates pitch and tsr 2D interpolators using ROSCO controller tuning.
    Parallelized over yaws; wind speeds for each yaw are solved sequentially in one worker with warm-start chaining.

    Args:
        rosco_yaml (str): Path to ROSCO controller YAML configuration file.
        bem (MITRotor.BEM): MITRotor BEM object used to generate rotor performance surfaces.
        TurbineName (str): Turbine name. Defaults to "IEA15MW".
        rotor_performance_filename (str): File name (including path) for saved Cp/Ct/Cq surface. Defaults to "Cp_Ct_Cq.txt".
        SimName (str): Simulation name used in ROSCO configuration. Defaults to "Sim1".
        GenEff (float): Generator efficiency (0-100%). Defaults to IEA15MW value of 95.756.
        generator_inertia (float): Generator inertia [kg m^2]. Defaults to IEA15MW value of 1836784.
        regenerate (bool): If True, regenerate the rotor performance file even if one exists. Defaults to False.
        save_control_file (str): Optional file name for saving the generated ROSCO control file. Defaults to None.
        save_dir (str): Directory in which to save generated files. Defaults to ".".
        dt (float): Controller timestep [s]. Defaults to 0.05.
        yaw_grid_deg (array-like): Yaw angles [deg] used when generating Cp/Ct/Cq performance surfaces. Defaults to np.arange(0.0, 25.0, 1.0).
        n_jobs (int): Number of parallel workers used when generating rotor performance surfaces across wind speeds. Defaults to -1 (all available workers).

    Returns: pitch_rad_interp, tsr_interp, turbine.rated_rotor_speed
    """

    # Load ROSCO inputs and make basic turbine
    inps = load_rosco_yaml(rosco_yaml)
    turbine_params = inps['turbine_params']
    turbine = ROSCO_turbine.Turbine(turbine_params)

    # If control csv exists and do not want to regenerate, load old csv
    if not regenerate and save_control_file is not None:
        pitch_rad_interp, tsr_interp = iu.load_control_interps_from_csv(save_control_file)
        return pitch_rad_interp, tsr_interp, turbine.rated_rotor_speed

    # Load and check control parameters
    controller_params = inps['controller_params']
    if (controller_params["WE_Mode"] != 0) and (controller_params["WE_Mode"] != 2):
        warnings.warn(
            "We suggest using WE_Mode = 0 or WE_Mode = 2 within the control parameter file.",
            UserWarning,
        )
    if (controller_params["VS_ControlMode"] == 2):
        warnings.warn(
                    "We suggest NOT using VS_ControlMode = 2 within the control parameter file. Mismatched TSR definitions between MITRotor and" \
                    "ROSCO lead to biased optimization.",
                    UserWarning,
                )

    # Generate turbine Cp/Ct surfaces
    turbine = load_from_mitrotor(
        turbine, bem,
        TurbineName=TurbineName, rotor_performance_filename=rotor_performance_filename,
        generator_inertia=generator_inertia, GenEff=GenEff,
        yaw=0.0, # Cp/Ct surface generated with yaw = 0
    )

    # Save turbine Cp/Ct surface
    cp_filename = turbine.rotor_performance_filename
    write_rotor_performance(turbine, txt_filename=cp_filename)

    # Make and tune controller
    controller = ROSCO_controller.Controller(controller_params)
    controller.tune_controller(turbine)

    # Write parameter input file
    param_filename = os.path.join(save_dir, 'DISCON.IN')
    write_DISCON(
        turbine, controller,
        param_file=param_filename,
        txt_filename=cp_filename
    )

    # Tuned initial setpoints
    init_pitch_rad_list = np.maximum(controller.pitch_op, controller.ps_min_bld_pitch)
    init_tsr_list = controller.TSR_op

    # Create parameter tables
    v_grid = controller.v.copy()
    yaw_grid_deg = np.sort(np.asarray(yaw_grid_deg, dtype=float))
    yaw_grid_rad = np.deg2rad(yaw_grid_deg)

    pitch_tbl = np.full((len(v_grid), len(yaw_grid_rad)), np.nan, dtype=float)
    tsr_tbl   = np.full_like(pitch_tbl, np.nan)
    power_tbl = np.full_like(pitch_tbl, np.nan)

    # Lightweight turbine object for worker pickling
    turbine_sim = SimpleNamespace(
        rotor_radius=float(turbine.rotor_radius),
        rho=float(turbine.rho),
        Ng=float(turbine.Ng),
        J=float(turbine.J),
        GBoxEff=float(turbine.GBoxEff),
        GenEff=float(turbine.GenEff),
    )

    # run simulations in parallel or serial depending on n_jobs
    def yaw_row_func(i, yaw_rad):
        return _run_one_yaw_row(
            i, v_grid, yaw_rad, init_pitch_rad_list, init_tsr_list,
            turbine_sim, bem, param_filename, dt, SimName
        )
    cols = iu.run_yaw_row_sim(v_grid, yaw_grid_rad, yaw_row_func, n_jobs)

    # Collect setpoints
    for i, pitch_col, tsr_col, power_col in cols:
        pitch_tbl[:, i] = pitch_col
        tsr_tbl[:, i]   = tsr_col
        power_tbl[:, i] = power_col

    # Optional CSV save
    iu.save_control_interps_to_csv(save_control_file, v_grid, yaw_grid_rad, pitch_tbl, tsr_tbl, power_tbl)

    # Make interpolators
    pitch_rad_interp = RegularGridInterpolator(
        (v_grid, yaw_grid_rad), pitch_tbl,
        method="linear", bounds_error=False, fill_value=None
    )
    tsr_interp = RegularGridInterpolator(
        (v_grid, yaw_grid_rad), tsr_tbl,
        method="linear", bounds_error=False, fill_value=None
    )
    return pitch_rad_interp, tsr_interp, turbine.rated_rotor_speed

def _run_one_yaw_row(
    i, v_grid, yaw_rad, init_pitch_rad_list, init_tsr_list,
    turbine_sim, bem, param_filename, dt, SimName
):
    """
    Worker: solve all yaw points for a single wind speed index i.
    Uses sequential warm-start across yaw in this row.
    Returns row arrays (pitch/tsr/power), with NaN on fail/non-convergence.
    """
    print(f"Starting ROSCO simulation for yaw = {yaw_rad}")
    nv = len(v_grid)
    pitch_col = np.full(nv, np.nan, dtype=float)
    tsr_col   = np.full(nv, np.nan, dtype=float)
    power_col = np.full(nv, np.nan, dtype=float)\

    j = 0
    init_pitch_rad = init_pitch_rad_list[j]
    init_pitch_deg = np.rad2deg(init_pitch_rad)
    init_tsr = init_tsr_list[j]
    init_rot_speed = init_tsr * v_grid[j] / turbine_sim.rotor_radius
    init_gen_speed   = init_rot_speed * turbine_sim.Ng

    controller_int = iu.WarmStartControllerInterface(
        lib_name,
        param_filename=param_filename,
        sim_name=f"{SimName}_{i}_{j}",
        DT=dt,
        init_ws=v_grid[j],
        init_rot_speed=init_rot_speed,
        init_gen_speed=init_gen_speed,
        init_pitch_deg=init_pitch_deg,   # deg
        init_torque=0.0,
        init_nac_imu=yaw_rad,            # rad
    )

    # lightweight sim object compatible with sim_ws_mitrotor
    sim = SimpleNamespace(turbine=turbine_sim, controller_int=controller_int)

    for j, v in enumerate(v_grid):
        controller_int = None
        try:
            converged = sim_ws_mitrotor(
                sim=sim, bem=bem, ws=v, dt=dt,
                init_tsr=init_tsr_list[j],
                init_pitch_rad=init_pitch_rad_list[j],   # rad
                init_yaw_rad=yaw_rad,            # rad
                wd=0.0,                      # rad
                verbose=False,
            )

            # Save only converged points; else remain NaN
            if converged:
                pitch_col[j] = sim.bld_pitch   # rad
                tsr_col[j]   = sim.tsr
                power_col[j] = sim.gen_power

        except Exception:
            continue

    # Kill controller
    sim.controller_int.kill_discon()

    return i, pitch_col, tsr_col, power_col


# -----------------------------
# Steady-state simulation
# -----------------------------
def sim_ws_mitrotor(
    sim, bem, ws, dt, init_tsr, init_pitch_rad,
    wd=0.0, init_yaw_rad=0.0,
    max_iter=20000,
    conv_settings=None,
    verbose=True,
):
    """
    Steady-state single-wind-speed simulation (no time history storage),
    with rolling-window convergence checks.

    ws: wind speed, (m/s)
    init_tsr: tip speed ratio, (-)
    init_pitch: initial blade pitch angle, (rad)
    wd: wind direction, (rad)
    yaw_init: initial "north" (or constant) yaw angle, (rad)
    max_iter: maximum number of iterations to try for convergence (int),
    conv_settings: settings that define convergence to steady state, (ConvergenceSettings)
    verbose: print extra information if True, (bool)

    """
    if conv_settings is None:
        conv_settings = iu.ConvergenceSettings()

    # Turbine constants
    R = sim.turbine.rotor_radius
    rho = sim.turbine.rho
    Ng = sim.turbine.Ng
    J = sim.turbine.J
    gbox_eff = sim.turbine.GBoxEff / 100.0
    gen_eff = sim.turbine.GenEff / 100.0

    # States
    bld_pitch = init_pitch_rad                    # rad
    rot_speed = (init_tsr * ws / R)           # rad/s
    gen_speed = rot_speed * Ng                # rad/s
    gen_torque = 0.0                          # Nm
    gen_power = 0.0                           # W
    nac_yaw = init_yaw_rad                        # rad
    nac_yawrate = 0.0                         # rad/s

    # Convergence tracker
    tracker = iu.init_convergence_tracker(dt, conv_settings)

    n_iter = 0
    t = 0.0
    converged = False
    last_metrics = None
    # Begin iteration
    t = 0.0
    while n_iter < max_iter:
        t += dt

        tsr = rot_speed * R / max(ws, 1e-6) 
        yaw_err = wd - nac_yaw

        # BEM call
        sol = bem(
            bld_pitch, tsr,
            yaw = yaw_err,
            tilt = 0.0 # yaw here is "effective yaw" or total y-z plane offset
        )
        cp = float(np.ravel(sol.Cp())[0])

        # Rotor dynamics calculations (same as used in ROSCO's sim_ws_series function)
        aero_torque = 0.5 * rho * (np.pi * R**3) * (cp / max(tsr, 1e-6)) * ws**2
        rot_speed = rot_speed + (dt / J) * (aero_torque - Ng * gen_torque / gbox_eff)
        gen_speed = rot_speed * Ng

        # Controller input state
        turbine_state = {
            "iStatus": 1,
            "t": t,
            "dt": dt,
            "ws": ws,
            "bld_pitch": bld_pitch, # rad
            "gen_torque": gen_torque,
            "gen_speed": gen_speed,
            "gen_eff": gen_eff,
            "rot_speed": rot_speed,
            "Yaw_fromNorth": nac_yaw, # rad
            "Y_MeasErr": yaw_err, # rad
        }

        # Controller call; torque, blade pitch, and yaw rate outputs
        gen_torque, bld_pitch, nac_yawrate = sim.controller_int.call_controller(turbine_state)

        # Calculate power
        gen_power = gen_speed * gen_torque * gen_eff  # W

        # Yaw dynamics update for if yaw-control is on
        nac_yaw += nac_yawrate * dt

        # Convergence tracking
        iu.update_convergence_tracker(tracker, gen_power, rot_speed, bld_pitch)
        converged, metrics = iu.check_convergence(n_iter, dt, tracker, conv_settings)
        if metrics is not None:
            last_metrics = metrics

        # Check if converged
        if converged:
            if verbose and last_metrics is not None:
                print(
                    f"Converged after {n_iter} steps "
                    f"(t={n_iter*dt:.1f}s): "
                    f"P={last_metrics['p_mean']/1000:.1f} kW, "
                    f"std(P)/mean(P)={last_metrics['rel_p_std']:.2e}, "
                    f"max|dω/dt|={last_metrics['max_domega_dt']:.2e} rad/s², "
                    f"max|dθ/dt|={np.rad2deg(last_metrics['max_dpitch_dt']):.2e} deg/s"
                )
            break

        n_iter += 1

    if not converged and verbose:
        print(f"WARNING: hit max_iter={max_iter} without convergence")

    # Save outputs
    sim.bld_pitch = bld_pitch               # rad
    sim.tsr = rot_speed * R / max(ws, 1e-6) # TSR w.r.t. freestream
    sim.rot_speed = rot_speed               # rad/s
    sim.gen_speed = gen_speed               # rad/s
    sim.gen_torque = gen_torque             # Nm
    sim.gen_power = gen_power               # W
    sim.nac_yaw = nac_yaw                   # rad
    sim.n_iter = n_iter
    sim.converged = converged

    return converged