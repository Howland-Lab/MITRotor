# Python modules
import os
from collections import deque
from ctypes import cdll, POINTER, c_float, c_int32, c_char_p, create_string_buffer
from dataclasses import dataclass
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d, RegularGridInterpolator
import warnings
import sys
from contextlib import contextmanager
# Parallelization modules
from types import SimpleNamespace
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
# ROSCO toolbox modules 
from rosco import discon_lib_path as lib_name
from rosco.toolbox import controller as ROSCO_controller
from rosco.toolbox import turbine as ROSCO_turbine
from rosco.toolbox import control_interface as ROSCO_ci
from rosco.toolbox import sim as ROSCO_sim
from rosco.toolbox.utilities import write_rotor_performance, write_DISCON
from rosco.toolbox.inputs.validation import load_rosco_yaml

# -----------------------------
# Create Ct/Cp surfaces
# -----------------------------
def load_from_mitrotor(
    turbine, bem, refine_cp_surface=False,
    TurbineName = "IEA15MW", rotor_performance_filename = 'Cp_Ct_Cq.txt',
    GenEff = 95.756, generator_inertia = 1836784,
    yaw = 0.0
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
    # Set turbine parameters
    turbine.TurbineName = TurbineName
    turbine.rotor_performance_filename = rotor_performance_filename

    (tower_height, gearbox_efficiency, gearbox_ratio, air_density) = bem.rotor.rosco_values
    turbine.TowerHt = tower_height
    turbine.GBoxEff = gearbox_efficiency
    turbine.Ng = gearbox_ratio
    turbine.rho = air_density

    turbine.GenEff = GenEff
    turbine.generator_inertia = generator_inertia

    turbine.yaw = yaw
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
        tilt=np.ones_like(tsr_flat) * np.deg2rad(5.0),
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

# -----------------------------
# Generate control scheme
# -----------------------------
def get_rosco_control_interps(
    rosco_yaml, bem,
    TurbineName="IEA15MW", rotor_performance_filename='Cp_Ct_Cq.txt', SimName="Sim1",
    GenEff=95.756, generator_inertia=1836784,
    regenerate=False,
    save_control_file=None,
    save_dir=".",
    dt=0.05,
    yaw_grid_deg=np.arange(0.0, 25.0, 1.0),  # degrees
    n_jobs=-1,                                 # parallel workers across wind speeds
):
    """
    Creates pitch and tsr 2D interpolators using ROSCO controller tuning.
    Parallelized over wind speeds; yaws for each wind are solved sequentially
    in one worker with warm-start chaining.
    """

    # Load ROSCO inputs and make basic turbine
    inps = load_rosco_yaml(rosco_yaml)
    turbine_params = inps['turbine_params']
    turbine = ROSCO_turbine.Turbine(turbine_params)

    # If control csv exists and do not want to regenerate, load old csv
    if not regenerate and save_control_file is not None:
        pitch_interp, tsr_interp = load_control_interps_from_csv(save_control_file)
        return pitch_interp, tsr_interp, turbine.rated_rotor_speed

    # Load and check control parameters
    controller_params = inps['controller_params']
    if (controller_params["WE_Mode"] != 0) or (controller_params["WE_Mode"] != 2):
        warnings.warn(
            "We suggest using WE_Mode = 0 or WE_Mode = 2 within the control parameter file.",
            UserWarning,
        )

    # Generate turbine Cp/Ct surfaces
    turbine = load_from_mitrotor(
        turbine, bem,
        TurbineName=TurbineName, rotor_performance_filename=rotor_performance_filename,
        generator_inertia=generator_inertia, GenEff=GenEff,
        yaw=0.0,
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
    init_pitch_list = np.rad2deg(np.maximum(controller.pitch_op, controller.ps_min_bld_pitch))
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

    # Calculate total number of cases
    n_wind = len(v_grid)
    n_yaw = len(yaw_grid_rad)
    total_cases = n_wind * n_yaw

    # Determine effective n_jobs
    if n_jobs is None:
        n_jobs_eff = 1
    elif n_jobs == -1:
        n_jobs_eff = min(os.cpu_count() or 1, n_wind)
    else:
        n_jobs_eff = max(1, min(int(n_jobs), n_wind))

    # Run parallel (fallback to serial on failure, e.g., pickling issue)
    print(
        f"Starting control LUT sweep: {n_wind} wind-row tasks "
        f"({total_cases} wind-yaw cases), using {n_jobs_eff} worker process(es)."
    )

    def _run_serial(desc):
        out = []
        for i, v in enumerate(tqdm(v_grid, total=n_wind, desc=desc, dynamic_ncols=True)):
            out.append(
                _run_one_wind_row(
                    i, v, yaw_grid_rad, init_pitch_list[i], init_tsr_list[i],
                    turbine_sim, bem, param_filename, dt, SimName
                )
            )
        return out

    rows = None
    if n_jobs_eff > 1:
        bar = tqdm(total=n_wind, desc="Control LUT sweep (wind rows)", dynamic_ncols=True)
        try:
            with tqdm_joblib(bar):
                rows = Parallel(n_jobs=n_jobs_eff, backend="loky", verbose=0, batch_size=1)(
                    delayed(_run_one_wind_row)(
                        i, v, yaw_grid_rad, init_pitch_list[i], init_tsr_list[i],
                        turbine_sim, bem, param_filename, dt, SimName
                    )
                    for i, v in enumerate(v_grid)
                )
        except Exception as e:
            warnings.warn(
                f"Parallel execution failed ({type(e).__name__}: {e}). "
                "Falling back to serial execution.",
                RuntimeWarning,
            )
        finally:
            bar.close()
    if rows is None:
        desc = "Control LUT sweep (serial fallback)" if n_jobs_eff > 1 else "Control LUT sweep (serial)"
        rows = _run_serial(desc)

    # Collect setpoints
    for i, pitch_row, tsr_row, power_row in rows:
        pitch_tbl[i, :] = pitch_row
        tsr_tbl[i, :]   = tsr_row
        power_tbl[i, :] = power_row

    # Optional CSV save
    if save_control_file is not None:
        save_control_file = Path(save_control_file)
        save_control_file.parent.mkdir(parents=True, exist_ok=True)

        vv, yy = np.meshgrid(v_grid, yaw_grid_rad, indexing="ij")
        data = np.column_stack([
            vv.ravel(),            # wind_speed_mps
            yy.ravel(),            # yaw_rad
            pitch_tbl.ravel(),     # pitch_rad
            tsr_tbl.ravel(),       # tsr
            power_tbl.ravel(),     # generated power
        ])

        header = "wind_speed_mps,yaw_rad,pitch_rad,tsr,gen_power"
        np.savetxt(save_control_file, data, delimiter=",", header=header, comments="")

    # Make interpolators
    pitch_interp = RegularGridInterpolator(
        (v_grid, yaw_grid_rad), pitch_tbl,
        method="linear", bounds_error=False, fill_value=None
    )
    tsr_interp = RegularGridInterpolator(
        (v_grid, yaw_grid_rad), tsr_tbl,
        method="linear", bounds_error=False, fill_value=None
    )

    return pitch_interp, tsr_interp, turbine.rated_rotor_speed

def _run_one_wind_row(
    i, v, yaw_grid_rad, init_pitch_deg0, init_tsr0,
    turbine_sim, bem, param_filename, dt, SimName
):
    """
    Worker: solve all yaw points for a single wind speed index i.
    Uses sequential warm-start across yaw in this row.
    Returns row arrays (pitch/tsr/power), with NaN on fail/non-convergence.
    """
    ny = len(yaw_grid_rad)
    pitch_row = np.full(ny, np.nan, dtype=float)
    tsr_row   = np.full(ny, np.nan, dtype=float)
    power_row = np.full(ny, np.nan, dtype=float)

    # Warm-start seeds for first yaw in this wind row
    prev_pitch_deg = float(init_pitch_deg0)
    prev_tsr = float(init_tsr0)

    for j, yaw_rad in enumerate(yaw_grid_rad):
        controller_int = None
        try:
            init_omega = prev_tsr * v / turbine_sim.rotor_radius
            init_gen   = init_omega * turbine_sim.Ng

            controller_int = WarmStartControllerInterface(
                lib_name,
                param_filename=param_filename,
                sim_name=f"{SimName}_{i}_{j}",
                DT=dt,
                init_ws=v,
                init_rot_speed=init_omega,
                init_gen_speed=init_gen,
                init_pitch_deg=prev_pitch_deg,   # deg
                init_torque=0.0,
                init_nac_imu=yaw_rad,            # rad
            )

            # lightweight sim object compatible with sim_ws_mitrotor
            sim = SimpleNamespace(turbine=turbine_sim, controller_int=controller_int)

            converged = sim_ws_mitrotor(
                sim=sim, bem=bem, ws=v, dt=dt,
                init_tsr=prev_tsr,
                init_pitch=prev_pitch_deg,   # deg
                yaw_init=yaw_rad,            # rad
                wd=0.0,
                verbose=False,
            )

            # Save only converged points; else remain NaN
            if converged:
                pitch_row[j] = sim.bld_pitch   # rad
                tsr_row[j]   = sim.tsr
                power_row[j] = sim.gen_power

                # update warm-start for next yaw
                prev_pitch_deg = np.rad2deg(sim.bld_pitch)
                prev_tsr = sim.tsr

        except Exception:
            # leave NaN and continue to next yaw
            if controller_int is not None:
                try:
                    controller_int.kill_discon()
                except Exception:
                    pass
            continue

    return i, pitch_row, tsr_row, power_row


def load_control_interps_from_csv(csv_path):
    """
    Load wind/yaw control LUT CSV and return 2D interpolators.

    Expected CSV columns:
      wind_speed_mps, yaw_rad, pitch_rad, tsr, gen_power
    """
    data = np.genfromtxt(csv_path, delimiter=",", names=True)

    ws = np.asarray(data["wind_speed_mps"], dtype=float)
    yaw = np.asarray(data["yaw_rad"], dtype=float)
    pitch = np.asarray(data["pitch_rad"], dtype=float)
    tsr = np.asarray(data["tsr"], dtype=float)

    v_grid = np.unique(ws)
    yaw_grid = np.unique(yaw)

    # sort rows by (ws, yaw), then reshape into [n_ws, n_yaw]
    order = np.lexsort((yaw, ws))
    n_ws, n_yaw = len(v_grid), len(yaw_grid)

    pitch_tbl = pitch[order].reshape(n_ws, n_yaw)
    tsr_tbl = tsr[order].reshape(n_ws, n_yaw)

    pitch_interp = RegularGridInterpolator(
        (v_grid, yaw_grid), pitch_tbl,
        method="linear", bounds_error=False, fill_value=None
    )
    tsr_interp = RegularGridInterpolator(
        (v_grid, yaw_grid), tsr_tbl,
        method="linear", bounds_error=False, fill_value=None
    )

    return pitch_interp, tsr_interp


def query_controls(
    interp,
    ws,
    yaw_rad,
    *,
    kind="generic",               # "pitch" or "tsr"
    rated_rotor_speed=None,       # rad/s (for 1D tsr scheme)
    rotor_radius=None,            # m   (required if rated_rotor_speed is set)
):
    """
    Supports:
      - 2D RegularGridInterpolator: interp(ws, yaw)
      - 1D interp1d/callable: interp(ws), yaw ignored (warn if |yaw| > 0)

    For 1D TSR and rated_rotor_speed provided:
      applies simple above-rated cap via omega = tsr*ws/R.
    """
    ws_arr = np.asarray(ws, dtype=float)
    yaw_arr = np.asarray(yaw_rad, dtype=float)
    ws_b, yaw_b = np.broadcast_arrays(ws_arr, yaw_arr)

    # 2D LUT path
    if isinstance(interp, RegularGridInterpolator) and len(interp.grid) == 2:
        pts = np.column_stack((ws_b.ravel(), yaw_b.ravel()))
        vals = np.asarray(interp(pts)).reshape(ws_b.shape)
        return float(vals) if vals.shape == () else vals

    # 1D legacy path
    if np.any(np.abs(yaw_b) > 0):
        warnings.warn(
            "Using 1D control curve with nonzero yaw/tilt; yaw/tilt is ignored for lookup.",
            UserWarning,
        )

    vals = np.asarray(interp(ws_b)).reshape(ws_b.shape)

    if kind == "tsr":
        if rated_rotor_speed is not None:
            if rotor_radius is None:
                raise ValueError("rotor_radius is required when rated_rotor_speed is set.")
            omega_lookup = vals * ws_b / np.maximum(rotor_radius, 1e-12)
            tsr_from_rated = rated_rotor_speed * rotor_radius / np.maximum(ws_b, 1e-6)
            vals = np.where(omega_lookup <= rated_rotor_speed, vals, tsr_from_rated)

        vals = np.maximum(vals, 0.0)

    return float(vals) if vals.shape == () else vals


# -----------------------------
# Steady-state simulation
# -----------------------------
def sim_ws_mitrotor(
    sim, bem, ws, dt, init_tsr, init_pitch,
    wd=0.0, yaw_init=0.0,
    max_iter=20000,
    conv_settings=None,
    verbose=True,
):
    """
    Steady-state single-wind-speed simulation (no time history storage),
    with rolling-window convergence checks.
    """
    if conv_settings is None:
        conv_settings = ConvergenceSettings()

    # Turbine constants
    R = sim.turbine.rotor_radius
    rho = sim.turbine.rho
    Ng = sim.turbine.Ng
    J = sim.turbine.J
    gbox_eff = sim.turbine.GBoxEff / 100.0
    gen_eff = sim.turbine.GenEff / 100.0

    # States
    bld_pitch = np.deg2rad(init_pitch) # rad (init_pitch is deg)
    rot_speed = (init_tsr * ws / R)           # rad/s
    gen_speed = rot_speed * Ng                # rad/s
    gen_torque = 0.0                          # Nm
    gen_power = 0.0                           # W
    nac_yaw = yaw_init                        # deg
    nac_yawrate = 0.0                         # deg/s

    # Convergence tracker
    tracker = init_convergence_tracker(dt, conv_settings)

    n_iter = 0
    t = 0.0
    converged = False
    last_metrics = None
    # Begin iteration
    while n_iter < max_iter:
        t += dt

        tsr = rot_speed * R / max(ws, 1e-6)
        yaw_err = wd - nac_yaw

        # BEM call
        sol = bem(
            bld_pitch, tsr,
            yaw = yaw_err,
            tilt = 0.0
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
            "bld_pitch": bld_pitch,
            "gen_torque": gen_torque,
            "gen_speed": gen_speed,
            "gen_eff": gen_eff,
            "rot_speed": rot_speed,
            "Yaw_fromNorth": nac_yaw,
            "Y_MeasErr": yaw_err,
        }

        # Controller call; torque, blade pitch, and yaw rate outputs
        gen_torque, bld_pitch, nac_yawrate = sim.controller_int.call_controller(turbine_state)

        # Calculate power
        gen_power = gen_speed * gen_torque * gen_eff  # W

        # Yaw dynamics update for if yaw-control is on
        nac_yaw += nac_yawrate * dt

        # Convergence tracking
        update_convergence_tracker(tracker, gen_power, rot_speed, bld_pitch)
        converged, metrics = check_convergence(n_iter, dt, tracker, conv_settings)
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

    # Kill controller
    sim.controller_int.kill_discon()

    # Save outputs
    sim.bld_pitch = bld_pitch               # rad
    sim.tsr = rot_speed * R / max(ws, 1e-6)
    sim.rot_speed = rot_speed               # rad/s
    sim.gen_speed = gen_speed               # rad/s
    sim.gen_torque = gen_torque             # Nm
    sim.gen_power = gen_power               # W
    sim.nac_yaw = nac_yaw                   # rad
    sim.n_iter = n_iter
    sim.converged = converged

    return converged

# -----------------------------
# Controller Interface -> child of ROSCOS's ControllerInterface
# -----------------------------
class WarmStartControllerInterface(ROSCO_ci.ControllerInterface):
    """
    ROSCO controller interface with configurable warm-start initial conditions.

    This subclass overrides the default `ControllerInterface` initialization so
    the first DISCON call (`iStatus=0`) is seeded with user-provided initial
    wind speed, rotor/generator speeds, pitch, torque, and nacelle IMU angle,
    instead of hard-coded defaults.

    Notes
    -----
    - This is useful for steady-state sweeps, where reducing startup transients
      improves convergence speed and robustness.
    - Units are expected to match ROSCO avrSWAP channel conventions.
    """

    def __init__(
        self,
        lib_name,
        param_filename="DISCON.IN",
        init_ws=10.0,             # m/s
        init_rot_speed=1.0,       # rad/s
        init_gen_speed=1.0,       # rad/s
        init_pitch_deg=0.0,       # deg
        init_torque=0.0,          # Nm
        init_nac_imu=0.0,         # rad
        **kwargs
    ):
        """
        Initialize warm-start settings and construct the ROSCO controller interface.

        Parameters
        ----------
        lib_name : str
            Path to the ROSCO dynamic library (.dll/.so/.dylib).
        param_filename : str, optional
            Path to DISCON input file, by default "DISCON.IN".
        init_ws : float, optional
            Initial wind speed used during DISCON init, in m/s.
        init_rot_speed : float, optional
            Initial low-speed shaft rotor speed, in rad/s.
        init_gen_speed : float, optional
            Initial generator speed (high-speed shaft), in rad/s.
        init_pitch_deg : float, optional
            Initial collective blade pitch, in degrees.
        init_torque : float, optional
            Initial generator torque, in Nm.
        init_nac_imu : float, optional
            Initial nacelle IMU angle/state, in radians.
        **kwargs
            Additional keyword arguments forwarded to
            `ROSCO_ci.ControllerInterface` (e.g., `DT`, `sim_name`, etc.).
        """
        # Set starting conditions
        self.init_ws = float(init_ws)
        self.init_rot_speed = float(init_rot_speed)
        self.init_gen_speed = float(init_gen_speed)
        self.init_torque = float(init_torque)
        self.init_nac_imu = float(init_nac_imu)

        # Parent uses self.pitch for initial blade pitch
        super().__init__(
            lib_name,
            param_filename=param_filename,
            pitch=init_pitch_deg,
            **kwargs
        )

    def init_discon(self):
        """
        Initialize DISCON with warm-start avrSWAP values.

        This method allocates and populates the avrSWAP array, sets the first-call
        status (`iStatus=0`), invokes DISCON once to initialize internal ROSCO states,
        then switches to normal run mode (`iStatus=1`) for subsequent calls.

        Raises
        ------
        ValueError
            If ROSCO returns a negative `aviFAIL` error code during initialization.
        """
        self.torque = self.init_torque

        # Load library + allocate swap
        self.discon = cdll.LoadLibrary(self.lib_name)
        self.avrSWAP = np.zeros(self.avr_size)

        # Required channels
        self.avrSWAP[2]  = self.DT
        self.avrSWAP[60] = self.num_blade

        # --- warm-start initial states (replaces hard-coded values) ---
        self.avrSWAP[19] = self.init_gen_speed   # gen speed [rad/s]
        self.avrSWAP[20] = self.init_rot_speed   # rot speed [rad/s]
        self.avrSWAP[82] = self.init_nac_imu     # nac IMU
        self.avrSWAP[26] = self.init_ws          # wind speed [m/s]
        self.avrSWAP[22] = self.init_torque      # gen torque [Nm]

        # Initial blade pitch (all blades)
        pitch_rad = np.deg2rad(self.pitch)
        self.avrSWAP[3]  = pitch_rad
        self.avrSWAP[32] = pitch_rad
        self.avrSWAP[33] = pitch_rad

        self.avrSWAP[27] = 1  # IPC flag

        # First-call init
        self.avrSWAP[0] = 0

        self.aviFAIL = c_int32()
        self.accINFILE = self.param_name.encode("utf-8")
        self.avcOUTNAME = self.sim_name.encode("utf-8")
        self.avcMSG = create_string_buffer(1000)

        self.discon.DISCON.argtypes = [
            POINTER(c_float),
            POINTER(c_int32),
            c_char_p,
            c_char_p,
            c_char_p,
        ]

        self.avrSWAP[48] = self.char_buffer
        self.avrSWAP[49] = len(self.param_name)
        self.avrSWAP[50] = len(self.avcOUTNAME)
        self.avrSWAP[51] = self.char_buffer

        with suppress_fortran_output():
            self.call_discon()   # iStatus=0 init call

        self.avrSWAP[0] = 1    # subsequent calls are normal

        if self.aviFAIL.value < 0:
            raise ValueError("ROSCO dynamic library has returned an error")
        
    def kill_discon(self):
        """
        Silent DISCON shutdown (suppresses ROSCO shutdown print spam).
        Removes ROSCO FORTRAN output clean up terminal so MITRotor interface
        outputs are visible.
        """
        try:
            with suppress_fortran_output():   # context manager
                super().kill_discon()
        except Exception:
            pass
        

@contextmanager
def suppress_fortran_output():
    """Suppress Fortran prints (stdout/stderr) in this process."""
    sys.stdout.flush()
    sys.stderr.flush()

    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(old_stdout)
        os.close(old_stderr)
        os.close(devnull)


# -----------------------------
# Convergence helper structures
# -----------------------------
@dataclass
class ConvergenceSettings:
    """
    Tunable parameters for steady-state convergence detection.

    Attributes
    ----------
    warmup_time : float
        Time to ignore before evaluating convergence, in seconds.
    window_time : float
        Length of rolling analysis window, in seconds.
    hold_time : float
        Duration for which all criteria must remain satisfied, in seconds.
    tol_rel_power_std : float
        Threshold on relative power variability: std(P)/|mean(P)|.
    tol_domega_dt : float
        Threshold on max rotor acceleration magnitude, in rad/s^2.
    tol_dpitch_dt : float
        Threshold on max pitch-rate magnitude, in rad/s.
    """
    warmup_time: float = 60.0
    window_time: float = 20.0
    hold_time: float = 5.0
    tol_rel_power_std: float = 1e-3
    tol_domega_dt: float = 1e-4
    tol_dpitch_dt: float = np.deg2rad(1e-3)


def init_convergence_tracker(dt: float, settings: ConvergenceSettings):
    """
    Create and initialize rolling-window state for convergence checks.

    Parameters
    ----------
    dt : float
        Simulation timestep in seconds.
    settings : ConvergenceSettings
        Convergence timing and tolerance settings.

    Returns
    -------
    dict
        Tracker containing step counts, rolling histories, and stable counter.
        Keys include: `warmup_steps`, `window_steps`, `hold_steps`,
        `stable_count`, `p_hist`, `omega_hist`, `pitch_hist`.
    """
    warmup_steps = int(np.ceil(settings.warmup_time / dt))
    window_steps = int(np.ceil(settings.window_time / dt))
    hold_steps = int(np.ceil(settings.hold_time / dt))

    tracker = {
        "warmup_steps": warmup_steps,
        "window_steps": window_steps,
        "hold_steps": hold_steps,
        "stable_count": 0,
        "p_hist": deque(maxlen=window_steps),      # W
        "omega_hist": deque(maxlen=window_steps),  # rad/s
        "pitch_hist": deque(maxlen=window_steps),  # rad
    }
    return tracker


def update_convergence_tracker(tracker, gen_power, rot_speed, bld_pitch):
    """
    Append latest simulation values to rolling convergence histories.

    Parameters
    ----------
    tracker : dict
        Tracker dictionary created by `init_convergence_tracker`.
    gen_power : float
        Generator electrical power, in W.
    rot_speed : float
        Rotor speed, in rad/s.
    bld_pitch : float
        Collective blade pitch, in rad.
    """
    tracker["p_hist"].append(float(gen_power))
    tracker["omega_hist"].append(float(rot_speed))
    tracker["pitch_hist"].append(float(bld_pitch))


def compute_window_metrics(tracker, dt):
    """
    Compute window-based convergence metrics from rolling histories.

    Parameters
    ----------
    tracker : dict
        Convergence tracker containing `p_hist`, `omega_hist`, `pitch_hist`.
    dt : float
        Simulation timestep in seconds.

    Returns
    -------
    dict
        Dictionary with:
        - `p_mean` : mean power over window [W]
        - `rel_p_std` : relative power std, std(P)/max(|mean(P)|, 1)
        - `max_domega_dt` : max |d(omega)/dt| over window [rad/s^2]
        - `max_dpitch_dt` : max |d(pitch)/dt| over window [rad/s]
    """
    p = np.asarray(tracker["p_hist"])
    om = np.asarray(tracker["omega_hist"])
    th = np.asarray(tracker["pitch_hist"])

    p_mean = float(np.mean(p))
    rel_p_std = float(np.std(p) / max(abs(p_mean), 1.0))
    max_domega_dt = float(np.max(np.abs(np.diff(om))) / dt)
    max_dpitch_dt = float(np.max(np.abs(np.diff(th))) / dt)

    return {
        "p_mean": p_mean,
        "rel_p_std": rel_p_std,
        "max_domega_dt": max_domega_dt,
        "max_dpitch_dt": max_dpitch_dt,
    }


def check_convergence(n_iter, dt, tracker, settings: ConvergenceSettings):
    """
    Evaluate steady-state convergence criteria.

    Criteria
    --------
    After warmup and once the rolling window is full, convergence is considered
    satisfied at a given step if:
      1) relative power std < `tol_rel_power_std`
      2) max rotor acceleration < `tol_domega_dt`
      3) max pitch rate < `tol_dpitch_dt`

    These conditions must hold for `hold_steps` consecutive evaluations.

    Parameters
    ----------
    n_iter : int
        Current simulation iteration index.
    dt : float
        Simulation timestep in seconds.
    tracker : dict
        Convergence tracker dictionary.
    settings : ConvergenceSettings
        Convergence timing and tolerance settings.

    Returns
    -------
    tuple[bool, dict | None]
        `(converged, metrics)` where:
        - `converged` is True if hold condition is met.
        - `metrics` is None until checks become active; otherwise window metrics
          from `compute_window_metrics`.
    """
    # not ready yet
    if n_iter < tracker["warmup_steps"]:
        return False, None
    if len(tracker["p_hist"]) < tracker["window_steps"]:
        return False, None

    metrics = compute_window_metrics(tracker, dt)

    ok_power = metrics["rel_p_std"] < settings.tol_rel_power_std
    ok_omega = metrics["max_domega_dt"] < settings.tol_domega_dt
    ok_pitch = metrics["max_dpitch_dt"] < settings.tol_dpitch_dt

    if ok_power and ok_omega and ok_pitch:
        tracker["stable_count"] += 1
    else:
        tracker["stable_count"] = 0

    converged = tracker["stable_count"] >= tracker["hold_steps"]
    return converged, metrics