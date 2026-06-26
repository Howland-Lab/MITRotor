# Python modules
import os
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import warnings
from ctypes import cdll, POINTER, c_float, c_int32, c_char_p, create_string_buffer
from collections import deque
from dataclasses import dataclass
# ROSCO toolbox modules 
from rosco import discon_lib_path as lib_name
from rosco.toolbox import controller as ROSCO_controller
from rosco.toolbox import turbine as ROSCO_turbine
from rosco.toolbox import control_interface as ROSCO_ci
from rosco.toolbox import sim as ROSCO_sim
from rosco.toolbox.utilities import write_rotor_performance, write_DISCON
from rosco.toolbox.inputs.validation import load_rosco_yaml

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

def get_rosco_control_interps(
    rosco_yaml, bem,
    TurbineName = "IEA15MW", rotor_performance_filename = 'Cp_Ct_Cq.txt', SimName = "Sim1",
    GenEff = 95.756, generator_inertia = 1836784,
):
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

    if controller_params["WE_Mode"] != 0:
        warnings.warn(
            "Using wind speed estimators in this simple simulation is known to cause problems. We suggest using WE_Mode = 0.",
            UserWarning,
        )

    # make turbine
    turbine = ROSCO_turbine.Turbine(turbine_params)
    turbine = load_from_mitrotor(
        turbine, bem,
        TurbineName = TurbineName, rotor_performance_filename = rotor_performance_filename,
        generator_inertia = generator_inertia, GenEff = GenEff,
        yaw = 0.0,
    )
    cp_filename = turbine.rotor_performance_filename
    write_rotor_performance(turbine, txt_filename=cp_filename)

    # make controller
    controller = ROSCO_controller.Controller(controller_params)
    controller.tune_controller(turbine)

    # Write parameter input file
    # param_filename = os.path.join(this_dir,'DISCON.IN')
    param_filename = 'DISCON.IN'
    write_DISCON(
        turbine,controller,
        param_file=param_filename, 
        txt_filename=cp_filename
    )

    R = turbine.rotor_radius
    GBRatio = turbine.Ng
    dt = 0.025
    # sec2steady = 5 * 60 # length of time to simulate (s)
    # t_list = np.arange(0, sec2steady, dt)
    # ws_list = np.ones_like(t_list)
    init_pitch_list = np.rad2deg(np.maximum(controller.pitch_op, controller.ps_min_bld_pitch))
    init_tsr_list = controller.TSR_op
    # ss_win = 20

    yaw_grid = np.array([0])
    v_grid = controller.v.copy()

    pitch_tbl = np.zeros((len(v_grid), len(yaw_grid)))
    tsr_tbl   = np.zeros_like(pitch_tbl)
    start = 0
    for (ii, v) in enumerate(v_grid[start:]):
        i = start + ii
        # ws_list.fill(v)
        init_pitch = init_pitch_list[i]
        init_tsr = init_tsr_list[i]
        # init_rmp = (init_tsr * v / R) * 60 / (2 * np.pi)
        for (j, yaw) in enumerate(yaw_grid):
            # Load controller library
            # controller_int = ROSCO_ci.ControllerInterface(
            #     lib_name, param_filename = param_filename, sim_name=f"{SimName}_{i}_{j}",
            # )

            omega0 = init_tsr * v / R   # rad/s
            gen0   = omega0 * GBRatio
            controller_int = WarmStartControllerInterface(
                lib_name,
                param_filename=param_filename,
                sim_name=f"{SimName}_{i}_{j}",
                DT=dt,
                init_ws=v,
                init_rot_speed=omega0,
                init_gen_speed=gen0,
                init_pitch_deg=init_pitch,   # init_pitch is in deg
                init_torque=0.0,
                init_nac_imu=np.deg2rad(yaw),
            )

            # Load the simulator
            sim = ROSCO_sim.Sim(turbine, controller_int)
            
            sim_ws_mitrotor(
                sim, bem, v, dt, init_tsr, init_pitch, yaw_init = yaw,
            )

            pitch_tbl[i, j] = sim.bld_pitch
            tsr_tbl[i, j]   = sim.tsr  

            # Run the simulation
            # sim.sim_ws_series(
            #     t_list, ws_list,
            #     init_pitch = init_pitch, rotor_rpm_init = init_rmp, yaw_init = np.deg2rad(yaw),
            #     make_plots=False,
            # )

            # nss = int(ss_win/dt) # number of steady-state steps
            # pitch_ss = np.mean(sim.bld_pitch[-nss:])
            # omega_ss = np.mean(sim.rot_speed[-nss:])
            # tsr_ss = omega_ss * R / v

            # pitch_tbl[i, j] = pitch_ss
            # tsr_tbl[i, j]   = tsr_ss   

    # plot_pitch_tsr_vs_wind(v_grid, yaw_grid, pitch_tbl, tsr_tbl, pitch_in_rad=True)

    # get interpolators
    # pitch_interp = rosco_pitch_interp(controller)
    # tsr_interp = rosco_tsr_interp(controller)

    wind_tbl = controller.v
    pitch_interp =  interp1d(wind_tbl, pitch_tbl[:, 0], kind="linear", fill_value="extrapolate", bounds_error=False)
    tsr_interp = interp1d(wind_tbl, tsr_tbl[:, 0], kind="linear", fill_value="extrapolate", bounds_error=False)

    return pitch_interp, tsr_interp, turbine.rated_rotor_speed

def plot_pitch_tsr_vs_wind(wind_speeds, yaw_vals, pitch_tbl, tsr_tbl, pitch_in_rad=True):
    sns.set_theme(style="whitegrid")

    n_yaw = len(yaw_vals)
    colors = sns.color_palette("viridis", n_colors=n_yaw)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # If stored in radians, convert to degrees for plotting
    pitch_plot = np.rad2deg(pitch_tbl) if pitch_in_rad else pitch_tbl

    # Plot one line per yaw
    for j, yaw in enumerate(yaw_vals):
        sns.lineplot(
            x=wind_speeds, y=pitch_plot[:, j],
            ax=axes[0], color=colors[j], linewidth=2
        )
        sns.lineplot(
            x=wind_speeds, y=tsr_tbl[:, j],
            ax=axes[1], color=colors[j], linewidth=2
        )

    axes[0].set_ylabel("Pitch (deg)" if pitch_in_rad else "Pitch")
    axes[0].set_title("Steady-State Pitch vs Wind Speed")

    axes[1].set_xlabel("Wind Speed (m/s)")
    axes[1].set_ylabel("TSR (-)")
    axes[1].set_title("Steady-State TSR vs Wind Speed")

    # One shared legend
    legend_handles = [
        Line2D([0], [0], color=colors[j], lw=2, label=f"{yaw:g}°")
        for j, yaw in enumerate(yaw_vals)
    ]
    fig.legend(handles=legend_handles, title="Yaw", loc="center right")
    plt.tight_layout(rect=[0, 0, 0.88, 1])  # leave room for legend
    plt.savefig("pitch_tsr_plots.png")


# Example call:
# plot_pitch_tsr_vs_wind(U_grid, yaw_grid, pitch_tbl, tsr_tbl, pitch_in_rad=True)

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


deg2rad = np.deg2rad(1)
rad2deg = np.rad2deg(1)
rpm2RadSec = 2.0*(np.pi)/60.0


def sim_ws_mitrotor(
        sim, bem, ws, dt, init_tsr, init_pitch,
        wd=0.0, yaw_init=0.0,
        max_iter = 10000, tol = 1e-2,
):
    # Store turbine data for convenience
    R = sim.turbine.rotor_radius
    rho = sim.turbine.rho
    GBRatio = sim.turbine.Ng

    # Declare output arrays
    bld_pitch = init_pitch * deg2rad
    tsr = init_tsr
    rot_speed = (tsr * ws / R) #rad / s
    gen_speed = rot_speed * GBRatio 
    aero_torque = 1000.0
    gen_torque = 1.0
    gen_power = 0.0
    nac_yaw =  yaw_init
    nac_yawerr = 0.0
    nac_yawrate = 0.0

    # Loop through time
    n_iter = 0
    t = 0.0
    while n_iter < max_iter:
        t += dt
        tsr = rot_speed * R / ws
        gamma = wd - nac_yaw
        sol = bem(bld_pitch, tsr, yaw = gamma)
        cp = sol.Cp()

        # Update the turbine state
        # -- 1DOF model: rotor speed and generator speed (scaled by Ng)
        aero_torque = 0.5 * rho * (np.pi * R**3) * (cp/tsr) * ws**2
        rot_speed = rot_speed + (dt/sim.turbine.J)*(aero_torque - sim.turbine.Ng * gen_torque / (sim.turbine.GBoxEff/100))
        gen_speed = rot_speed * sim.turbine.Ng

        # populate turbine state dictionary
        turbine_state = {}
        turbine_state['iStatus'] = 1
        turbine_state['t'] = t
        turbine_state['dt'] = dt
        turbine_state['ws'] = ws
        turbine_state['bld_pitch'] = bld_pitch
        turbine_state['gen_torque'] = gen_torque
        turbine_state['gen_speed'] = gen_speed
        turbine_state['gen_eff'] = sim.turbine.GenEff/100
        turbine_state['rot_speed'] = rot_speed
        turbine_state['Yaw_fromNorth'] = nac_yaw
        turbine_state['Y_MeasErr'] = gamma

        # Define outputs
        gen_torque, bld_pitch, nac_yawrate = sim.controller_int.call_controller(turbine_state)

        # Calculate the power
        gen_power_old = gen_power
        gen_power = gen_speed * gen_torque * sim.turbine.GenEff / 100
        # Calculate the nacelle position
        nac_yaw += nac_yawrate * dt

        if (gen_power_old - gen_power < tol) and (n_iter > 60 / dt):
            print(f"Converged after {n_iter} with Power: {gen_power} kW")
            break
        n_iter += 1

    sim.controller_int.kill_discon()

    # Save these values
    sim.bld_pitch = bld_pitch
    sim.tsr = rot_speed * R / ws
    # self.rot_speed = rot_speed
    # self.gen_speed = gen_speed
    # self.aero_torque = aero_torque
    # self.gen_torque = gen_torque
    # self.gen_power = gen_power
    # self.ws = ws
    # self.wd = wd
    # self.nac_yaw = nac_yaw


# from rosco.toolbox.control_interface import ControllerInterface
# # -------------------------------------------------------------
# # Try generating 2D lookup table with ROSCO!!
# # -------------------------------------------------------------

# def _parse_rosco_outputs(out):
#     """
#     Handle slight API differences in ROSCO toolbox versions.
#     Expected: pitch command (rad, collective) and generator torque command (N-m).
#     """
#     # dict-like
#     if isinstance(out, dict):
#         if "bld_pitch_cmd" in out:
#             beta_cmd = np.atleast_1d(out["bld_pitch_cmd"])[0]
#         elif "pitch_cmd" in out:
#             beta_cmd = np.atleast_1d(out["pitch_cmd"])[0]
#         else:
#             raise KeyError("Could not find pitch command in ROSCO output dict.")

#         if "gen_torque_cmd" in out:
#             Tg_cmd = out["gen_torque_cmd"]
#         elif "torque_cmd" in out:
#             Tg_cmd = out["torque_cmd"]
#         else:
#             raise KeyError("Could not find torque command in ROSCO output dict.")
#         return float(beta_cmd), float(Tg_cmd)

#     # tuple/list-like fallback
#     if isinstance(out, (tuple, list)) and len(out) >= 2:
#         beta_cmd = np.atleast_1d(out[0])[0]
#         Tg_cmd = out[1]
#         return float(beta_cmd), float(Tg_cmd)

#     raise TypeError("Unknown ROSCO controller return type.")


# def _call_rosco(ci, t, dt, beta, omega, Tg, Ng, U_meas):
#     """
#     Version-tolerant ROSCO call wrapper.
#     You may need to tweak keyword names once for your installed ROSCO version.
#     """
#     # Try common signature 1
#     try:
#         out = ci.call_controller(
#             t=t,
#             dt=dt,
#             bld_pitch=np.array([beta, beta, beta]),
#             gen_speed=omega * Ng,
#             gen_torque=Tg,
#             rot_speed=omega,
#             wind_speed=U_meas,
#         )
#         return _parse_rosco_outputs(out)
#     except TypeError:
#         pass

#     # Try common signature 2
#     out = ci.call_controller(
#         t, dt,
#         np.array([beta, beta, beta]),
#         omega * Ng,
#         Tg,
#         omega,
#         U_meas,
#     )
#     return _parse_rosco_outputs(out)


# def generate_rosco_2d_tables_full_rosco(
#     rosco_yaml,
#     bem,
#     wind_table,
#     yaw_table,
#     discon_lib_path,      # compiled ROSCO shared lib (e.g., libdiscon.so / discon.dll)
#     discon_in_path,       # DISCON.IN from ROSCO tuning
#     TurbineName="IEA15MW",
#     GenEff=95.756,
#     generator_inertia=1836784,
#     dt=0.05,
#     t_max=220.0,
#     tilt_deg=5.0,
#     min_ws=0.25,
#     omega_eps=0.2,
#     tau_pitch=0.35,
#     tau_torque=0.20,
#     pitch_rate_max=np.deg2rad(8.0),
#     torque_rate_max=3e6,
#     t_min_check=20.0,
#     conv_window=4.0,
#     avg_window=5.0,
#     tol_omega=2e-3,
#     tol_pitch=np.deg2rad(0.03),
#     tol_torque=2e4,
#     save_npz=None,
# ):
#     """
#     Fully ROSCO-controlled trim-map generation (no hand-coded region logic).
#     """
#     # Load turbine properties via your existing pipeline
#     inps = load_rosco_yaml(rosco_yaml)
#     turbine = ROSCO_turbine.Turbine(inps["turbine_params"])
#     turbine = load_from_mitrotor(
#         turbine, bem,
#         TurbineName=TurbineName,
#         GenEff=GenEff,
#         generator_inertia=generator_inertia,
#         yaw=0.0,
#     )

#     # For initialization only (NOT control logic)
#     pitch_interp, tsr_interp, rated_rotor_speed = get_rosco_control_interps(
#         rosco_yaml, bem,
#         TurbineName=TurbineName, GenEff=GenEff, generator_inertia=generator_inertia
#     )

#     # ROSCO DLL interface
#     ci = ControllerInterface(discon_lib_path, discon_in_path)

#     R = float(bem.rotor.R)
#     A = np.pi * R**2
#     rho = float(turbine.rho)
#     Ng = float(turbine.Ng)
#     J = float(turbine.J)

#     gbox_eff = float(turbine.GBoxEff)
#     if gbox_eff > 1.5:  # percent -> fraction
#         gbox_eff /= 100.0
#     gbox_eff = np.clip(gbox_eff, 1e-3, 1.0)

#     wind_table = np.asarray(wind_table, dtype=float)
#     yaw_table = np.asarray(yaw_table, dtype=float)

#     nW, nY = len(wind_table), len(yaw_table)
#     tabs = {
#         "wind_table": wind_table,
#         "yaw_table": yaw_table,
#         "pitch": np.zeros((nW, nY)),
#         "omega": np.zeros((nW, nY)),
#         "tsr_bem": np.zeros((nW, nY)),
#         "tsr_normal": np.zeros((nW, nY)),
#         "Cp": np.zeros((nW, nY)),
#         "Ct": np.zeros((nW, nY)),
#         "a": np.zeros((nW, nY)),
#         "power": np.zeros((nW, nY)),
#         "converged": np.zeros((nW, nY), dtype=bool),
#     }

#     n_steps = int(t_max / dt)
#     n_conv = max(3, int(conv_window / dt))
#     n_avg = max(3, int(avg_window / dt))

#     for j, yaw_deg in enumerate(yaw_table):
#         yaw = np.deg2rad(yaw_deg)
#         tilt = np.deg2rad(tilt_deg)

#         # warm start along wind
#         state = None

#         for i, U_in in enumerate(wind_table):
#             U = max(float(U_in), min_ws)
#             U_n = max(U * np.cos(yaw) * np.cos(tilt), min_ws)

#             # init state
#             if state is None:
#                 lam0 = max(float(tsr_interp(U_n)), 0.1)
#                 omega = max(lam0 * U_n / R, omega_eps)
#                 beta = float(pitch_interp(U_n))
#                 Tg = float(turbine.rated_torque * (omega / max(rated_rotor_speed, 1e-3))**2)
#             else:
#                 omega, beta, Tg = state["omega"], state["beta"], state["Tg"]

#             omega_hist, beta_hist, Tg_hist = [], [], []
#             cp_hist, ct_hist, a_hist = [], [], []
#             converged = False

#             for k in range(n_steps):
#                 t = k * dt
#                 omega = max(omega, omega_eps)

#                 # --- ROSCO gives commands (fully controller-managed) ---
#                 beta_cmd, Tg_cmd = _call_rosco(
#                     ci=ci, t=t, dt=dt,
#                     beta=beta, omega=omega, Tg=Tg, Ng=Ng,
#                     U_meas=U_n,   # rotor-normal measured wind
#                 )

#                 # actuator dynamics / limits (plant-side, not control policy)
#                 dbeta = np.clip((beta_cmd - beta) / tau_pitch, -pitch_rate_max, pitch_rate_max)
#                 dTg = np.clip((Tg_cmd - Tg) / tau_torque, -torque_rate_max, torque_rate_max)
#                 beta += dbeta * dt
#                 Tg += dTg * dt

#                 # MITRotor aero
#                 tsr_bem = omega * R / U
#                 sol = bem(
#                     np.array([beta]),
#                     np.array([tsr_bem]),
#                     yaw=np.array([yaw]),
#                     tilt=np.array([tilt]),
#                 )
#                 Cp = float(sol.Cp()[0])
#                 Ct = float(sol.Ct()[0])
#                 a = float(sol.a()[0])

#                 P = 0.5 * rho * A * Cp * U**3
#                 Qaero = P / max(omega, omega_eps)
#                 Qgen_rotor = Tg * Ng / gbox_eff
#                 domega = (Qaero - Qgen_rotor) / J
#                 omega = max(omega + domega * dt, omega_eps)

#                 omega_hist.append(omega)
#                 beta_hist.append(beta)
#                 Tg_hist.append(Tg)
#                 cp_hist.append(Cp)
#                 ct_hist.append(Ct)
#                 a_hist.append(a)

#                 if t >= t_min_check and len(omega_hist) >= n_conv:
#                     if (np.ptp(omega_hist[-n_conv:]) < tol_omega and
#                         np.ptp(beta_hist[-n_conv:]) < tol_pitch and
#                         np.ptp(Tg_hist[-n_conv:]) < tol_torque):
#                         converged = True
#                         break

#             # steady averages
#             omega_ss = float(np.mean(omega_hist[-n_avg:]))
#             beta_ss = float(np.mean(beta_hist[-n_avg:]))
#             Cp_ss = float(np.mean(cp_hist[-n_avg:]))
#             Ct_ss = float(np.mean(ct_hist[-n_avg:]))
#             a_ss = float(np.mean(a_hist[-n_avg:]))

#             tabs["pitch"][i, j] = beta_ss
#             tabs["omega"][i, j] = omega_ss
#             tabs["tsr_bem"][i, j] = omega_ss * R / U
#             tabs["tsr_normal"][i, j] = omega_ss * R / U_n
#             tabs["Cp"][i, j] = Cp_ss
#             tabs["Ct"][i, j] = Ct_ss
#             tabs["a"][i, j] = a_ss
#             tabs["power"][i, j] = 0.5 * rho * A * Cp_ss * U**3
#             tabs["converged"][i, j] = converged

#             state = {"omega": omega, "beta": beta, "Tg": Tg}

#             if not converged:
#                 print(f"[warn] not converged at U={U:.2f}, yaw={yaw_deg:.2f}")

#     if save_npz is not None:
#         np.savez(save_npz, **tabs)

#     return tabs

class WarmStartControllerInterface(ROSCO_ci.ControllerInterface):
    def __init__(
        self,
        lib_name,
        param_filename="DISCON.IN",
        init_ws=10.0,             # m/s
        init_rot_speed=1.0,       # rad/s (LSS)
        init_gen_speed=1.0,       # rad/s (HSS)
        init_pitch_deg=0.0,       # deg
        init_torque=0.0,          # Nm
        init_nac_imu=0.0,         # rad
        **kwargs
    ):
        self.init_ws = float(init_ws)
        self.init_rot_speed = float(init_rot_speed)
        self.init_gen_speed = float(init_gen_speed)
        self.init_torque = float(init_torque)
        self.init_nac_imu = float(init_nac_imu)

        # parent uses self.pitch for initial blade pitch
        super().__init__(
            lib_name,
            param_filename=param_filename,
            pitch=init_pitch_deg,
            **kwargs
        )

    def init_discon(self):
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
        pitch_rad = self.pitch * np.deg2rad(1.0)
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

        self.call_discon()     # iStatus=0 init call
        self.avrSWAP[0] = 1    # subsequent calls are normal

        if self.aviFAIL.value < 0:
            raise ValueError("ROSCO dynamic library has returned an error")
        


