from collections import deque
from contextlib import contextmanager
from ctypes import cdll, POINTER, c_float, c_int32, c_char_p, create_string_buffer
from dataclasses import dataclass
import numpy as np
import sys
from scipy.interpolate import RegularGridInterpolator
import os
import warnings
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
from pathlib import Path

# ROSCO imports
from rosco.toolbox import control_interface as ROSCO_ci

# -----------------------------
# Control Interpolator Helpers
# -----------------------------
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
    Gets TRS or pitch values for given wind speeds and yaw from interpolator

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
    else: # 1D legacy path
        vals = np.asarray(interp(ws_b)).reshape(ws_b.shape)
        # warning if non-zero yaw values
        if np.any(np.abs(yaw_b) > 0):
            warnings.warn(
                "Using 1D control curve with nonzero yaw/tilt; yaw/tilt is ignored for lookup.",
                UserWarning,
            )
        # basic rated rotor speed tracking for above rated conditions
        if kind == "tsr":
            if rated_rotor_speed is not None:
                if rotor_radius is None:
                    raise ValueError("rotor_radius is required when rated_rotor_speed is set.")
                omega_lookup = vals * ws_b / np.maximum(rotor_radius, 1e-12)
                tsr_from_rated = rated_rotor_speed * rotor_radius / np.maximum(ws_b, 1e-6)
                vals = np.where(omega_lookup <= rated_rotor_speed, vals, tsr_from_rated)
            vals = np.maximum(vals, 0.0)

    return float(vals) if vals.shape == () else vals

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

    # sort cols by (ws, yaw), then reshape into [n_ws, n_yaw]
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

def save_control_interps_to_csv(save_control_file, v_grid, yaw_grid_rad, pitch_tbl, tsr_tbl, power_tbl):
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

# -----------------------------
# Parallelization Helpers
# -----------------------------
def get_num_jobs(n_jobs, n_wind):
# Determine effective n_jobs
    if n_jobs is None:
        n_jobs_eff = 1
    elif n_jobs == -1:
        n_jobs_eff = min(os.cpu_count() or 1, n_wind)
    else:
        n_jobs_eff = max(1, min(int(n_jobs), n_wind))
    return n_jobs_eff

def run_yaw_row_sim(v_grid, yaw_grid_rad, yaw_row_func, n_jobs):
    # Calculate total number of cases
    n_wind = len(v_grid)
    n_yaw = len(yaw_grid_rad)
    total_cases = n_wind * n_yaw

    # Determine effective n_jobs
    n_jobs_eff = get_num_jobs(n_jobs, n_wind)

    # Run parallel (fallback to serial on failure, e.g., pickling issue)
    print(
        f"Starting control LUT sweep: {n_yaw} yaw tasks "
        f"({total_cases} wind-yaw cases), using {n_jobs_eff} worker process(es)."
    )
    
    def _run_serial(desc):
        out = []
        for i, yaw_rad in enumerate(tqdm(yaw_grid_rad, total=n_yaw, desc=desc, dynamic_ncols=True)):
            out.append(yaw_row_func(i, yaw_rad))
        return out

    cols = None
    if n_jobs_eff > 1:
        bar = tqdm(total=n_yaw, desc="Control LUT sweep (wind cols)", dynamic_ncols=True)
        try:
            with tqdm_joblib(bar):
                cols = Parallel(n_jobs=n_jobs_eff, backend="loky", verbose=0, batch_size=1)(
                    delayed(yaw_row_func)(i, yaw_rad)
                    for i, yaw_rad in enumerate(yaw_grid_rad)
                )
        except Exception as e:
            warnings.warn(
                f"Parallel execution failed ({type(e).__name__}: {e}). "
                "Falling back to serial execution.",
                RuntimeWarning,
            )
        finally:
            bar.close()
    if cols is None:
        desc = "Control LUT sweep (serial fallback)" if n_jobs_eff > 1 else "Control LUT sweep (serial)"
        cols = _run_serial(desc)

    return cols

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
        init_yaw_rad=0.0,         # rad
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
        init_yaw_rad : float, optional
            Initial yaw angle/state, in radians.
        **kwargs
            Additional keyword arguments forwarded to
            `ROSCO_ci.ControllerInterface` (e.g., `DT`, `sim_name`, etc.).
        """
        # Set starting conditions
        self.init_ws = float(init_ws)
        self.init_rot_speed = float(init_rot_speed)
        self.init_gen_speed = float(init_gen_speed)
        self.init_torque = float(init_torque)
        self.init_yaw_rad = float(init_yaw_rad)

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
        self.avrSWAP[82] = 0  # HARD CODE initial nacIMU = 0
        self.avrSWAP[26] = self.init_ws          # wind speed [m/s]
        self.avrSWAP[22] = self.init_torque      # gen torque [Nm]

        # Initial blade pitch (all blades)
        # passing in deg to swap to rad with standard ControllerInterface behavior
        pitch_rad = np.deg2rad(self.pitch)
        self.avrSWAP[3]  = pitch_rad
        self.avrSWAP[32] = pitch_rad
        self.avrSWAP[33] = pitch_rad

        # Initial yaw
        self.avrSWAP[23] = self.init_yaw_rad
        self.avrSWAP[36] = self.init_yaw_rad

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
# ROSCO Convergence helper structures
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
    tol_domega_dt: float = 1e-3
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