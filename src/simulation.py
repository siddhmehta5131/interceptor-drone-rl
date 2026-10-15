"""
simulation.py
-------------
Real-time physics-based quadrotor flight simulator with live visualisation.

Implements a full 9-stage physics pipeline derived from first principles:

    Stage A   Rate PID controller    (gamepad → desired body rates → torque cmd)
    Stage B   Mixer                  (torque cmd + collective thrust → per-motor cmd)
    Stage 2   ESC / Battery model    (motor cmd → steady-state motor speed)
    Stage 3   Motor dynamics         (first-order lag on motor speed)
    Stage 4   Propeller forces       (motor speed → thrust + drag torque per prop)
    Stage 5   Force aggregation      (sum over 4 propellers)
    Stage 6   Gyroscopic torques     (reaction + inertial / Coriolis terms)
    Stage 7   Aerodynamic model      (polynomial drag — fitted from PX4 flight logs)
    Stage 8   Rigid-body dynamics    (Newton-Euler equations of motion)
    Stage 9   Euler integration      (propagate state to next time step)

Physical parameters are calibrated to the Crazyflie 2.0/2.1 nano-quadrotor.
Aerodynamic coefficients are fitted by least-squares regression against real
PX4 flight logs (see `docs/coefficient_fitting.md`).

Author  : Siddh Mehta
Project : RL-Based Autonomous Interceptor Drone
Date    : 2025

Variable taxonomy (project-wide)
----------------------------------
TRUE INPUT  : thrust, roll, pitch, yaw  — gamepad stick positions in [-1, 1]
TRUE OUTPUT : p_WB (3,) position, q_WB (4,) attitude quaternion

All loop variables (v_WB, omega_B, Omega, …) are in SI units.

Dependencies
------------
    numpy, matplotlib, pygame
    visualiser  (src/visualiser.py — plot_trajectory, plot_input_bars, read_gamepad)

Usage
-----
    python src/simulation.py
    # Close either matplotlib window to exit.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from visualiser import plot_trajectory, plot_input_bars, read_gamepad


# ============================================================================
# Math helpers
# ============================================================================

def _quat2rotm(q: np.ndarray) -> np.ndarray:
    """Quaternion [w, x, y, z] → 3×3 rotation matrix (body → world)."""
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2),  2*(x*y - w*z),        2*(x*z + w*y)      ],
        [2*(x*y + w*z),         1 - 2*(x**2 + z**2),   2*(y*z - w*x)     ],
        [2*(x*z - w*y),         2*(y*z + w*x),         1 - 2*(x**2 + y**2)],
    ])


def _quat_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions in [w, x, y, z] convention."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


# ============================================================================
# Stage functions  (A, B, 2 … 9)
# ============================================================================

def pid_controller(
    omega_cmd: np.ndarray,
    omega_B_prev: np.ndarray,
    omega_B_prev2: np.ndarray,
    I_prev: np.ndarray,
    throttle_cut_flag: bool,
    Kp: np.ndarray,
    Ki: np.ndarray,
    Kd: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Stage A — Rate PID controller.

    Computes the per-axis torque correction command from the error between
    desired and actual body rates.  The integrator is hard-reset to zero
    whenever the throttle is cut to prevent integrator windup on landing.

    Parameters
    ----------
    omega_cmd        : (3,) desired body rates [wx, wy, wz]  rad/s
    omega_B_prev      : (3,) body rates at iteration i-1
    omega_B_prev2     : (3,) body rates at iteration i-2  (needed for D-term)
    I_prev            : (3,) integrator state at i-1
    throttle_cut_flag : bool  — True resets integrator
    Kp, Ki, Kd        : (3,) per-axis PID gains
    dt                : float  step time (s)

    Returns
    -------
    u     : (3,) torque-correction command [u_x, u_y, u_z]
    I_new : (3,) updated integrator state for iteration i+1
    """
    e = omega_cmd - omega_B_prev
    P = Kp * e
    I_new = np.zeros(3) if throttle_cut_flag else I_prev + Ki * e * dt
    D = -Kd * (omega_B_prev - omega_B_prev2) / dt
    return P + I_new + D, I_new


def mixer(
    u: np.ndarray,
    c_cmd: float,
    cmd_min: float,
    cmd_max: float,
    motor_geometry: np.ndarray,
) -> np.ndarray:
    """Stage B — Mixer: torque + collective thrust → per-motor commands.

    Scales down the torque components proportionally if any motor would
    saturate, preserving the collective thrust exactly.

    Parameters
    ----------
    u              : (3,) torque-correction command [u_x, u_y, u_z]
    c_cmd          : scalar collective thrust command (normalised)
    cmd_min        : scalar lower saturation limit
    cmd_max        : scalar upper saturation limit
    motor_geometry : (4, 3) sign matrix  — layout/spin pattern of 4 motors

    Returns
    -------
    cmd : (4,) individual motor commands, clipped to [cmd_min, cmd_max]

    Bug fixed (vs original)
    -----------------------
    Division by zero when cmd_max_val == c_cmd (all motors identical) is
    now guarded with an abs(denom) > 1e-9 check.
    """
    cmd_raw     = c_cmd + motor_geometry @ u
    cmd_max_val = float(np.max(cmd_raw))
    denom       = cmd_max_val - c_cmd

    if cmd_max_val > cmd_max and abs(denom) > 1e-9:
        scale   = (cmd_max - c_cmd) / denom
        u       = scale * u
        cmd_raw = c_cmd + motor_geometry @ u

    return np.clip(cmd_raw, cmd_min, cmd_max)


def esc_battery_model(
    cmd: np.ndarray,
    U_bat_prev: float,
    Omega_prev: np.ndarray,
    eta: float,
    battery_coeffs: np.ndarray,
    c_d: float,
) -> tuple[np.ndarray, float]:
    """Stage 2 — ESC / Battery model.

    Maps normalised motor commands to steady-state motor speeds via a
    fitted polynomial that captures ESC non-linearity and battery sag.

    The fitted polynomial uses an *inverted* command convention: cmd=0
    in the raw coefficients means full throttle.  Inverting the command
    here aligns it with the simulation convention (cmd=0 → motors off,
    cmd=1 → full throttle).

    Parameters
    ----------
    cmd            : (4,) normalised motor commands in [0, 1]
    U_bat_prev     : scalar battery voltage (V) at previous step
    Omega_prev     : (4,) motor speeds at previous step (rad/s)
    eta            : scalar motor efficiency (dimensionless)
    battery_coeffs : (5,) polynomial coefficients [c0..c4]
    c_d            : scalar propeller drag coefficient (N·m/(rad/s)²)

    Returns
    -------
    Omega_ss  : (4,) steady-state target motor speed (rad/s)
    U_bat_new : float updated battery voltage (simple model: constant)
    """
    # Power draw — kept for future battery-voltage-dynamics integration
    _ = (c_d * Omega_prev**3) / eta  # noqa: F841

    # Constant-voltage placeholder until a full battery model is fitted
    U_bat_new = U_bat_prev

    cmd_esc = 1.0 - np.clip(cmd, 0.0, 1.0)  # invert: cmd=1 → esc_cmd=0 (max speed)
    sqrt_esc = np.sqrt(cmd_esc)

    Omega_ss = (
        battery_coeffs[0]
        + battery_coeffs[1] * U_bat_prev
        + battery_coeffs[2] * sqrt_esc
        + battery_coeffs[3] * cmd_esc
        + battery_coeffs[4] * U_bat_prev * sqrt_esc
    )
    return Omega_ss, U_bat_new


def motor_dynamics(
    Omega_ss: np.ndarray,
    Omega_prev: np.ndarray,
    k_mot: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Stage 3 — First-order motor lag.

    Parameters
    ----------
    Omega_ss   : (4,) steady-state target speed (rad/s)
    Omega_prev : (4,) actual speed at previous step (rad/s)
    k_mot      : float motor time constant (s)
    dt         : float step time (s)

    Returns
    -------
    Omega     : (4,) actual motor speed this step
    Omega_dot : (4,) speed derivative (reused in Stage 6)
    """
    Omega_dot = (Omega_ss - Omega_prev) / k_mot
    return Omega_prev + dt * Omega_dot, Omega_dot


def propeller_force_torque(
    Omega: np.ndarray,
    c_l: float,
    c_d: float,
    spin_sign: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Stage 4 — Per-propeller thrust and reaction-drag torque.

    Parameters
    ----------
    Omega     : (4,) motor speeds (rad/s)
    c_l       : float lift coefficient  N·(rad/s)⁻²
    c_d       : float drag coefficient  N·m·(rad/s)⁻²
    spin_sign : (4,) +1 or -1 per motor (CW / CCW)

    Returns
    -------
    f_props   : (4, 3) per-propeller force vectors
    tau_props : (4, 3) per-propeller torque vectors

    Note
    ----
    Without per-motor spin_sign, the four drag torques cancel and the
    vehicle can never produce a net yaw torque — a common simulation bug.
    """
    f_props   = np.zeros((4, 3))
    tau_props = np.zeros((4, 3))
    for j in range(4):
        f_props[j]   = [0.0, 0.0, c_l * Omega[j]**2]
        tau_props[j] = [0.0, 0.0, spin_sign[j] * c_d * Omega[j]**2]
    return f_props, tau_props


def aggregate_forces(
    f_props: np.ndarray,
    tau_props: np.ndarray,
    r_P: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Stage 5 — Sum propeller contributions into total body-frame wrench.

    Parameters
    ----------
    f_props   : (4, 3) per-propeller forces
    tau_props : (4, 3) per-propeller torques
    r_P       : (4, 3) propeller positions relative to CoM (body frame)

    Returns
    -------
    f_prop   : (3,) total propeller force
    tau_prop : (3,) total propeller torque (including moment-arm cross terms)
    """
    f_prop   = np.sum(f_props, axis=0)
    tau_prop = np.zeros(3)
    for j in range(4):
        tau_prop += tau_props[j] + np.cross(r_P[j], f_props[j])
    return f_prop, tau_prop


def motor_reaction_inertial_torque(
    Omega_dot: np.ndarray,
    omega_B: np.ndarray,
    J_mp: float,
    zeta: np.ndarray,
    J: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Stage 6 — Gyroscopic and inertial coupling torques.

    Parameters
    ----------
    Omega_dot : (4,) motor speed derivatives
    omega_B   : (3,) body angular rate
    J_mp      : float combined motor + propeller rotational inertia (kg·m²)
    zeta      : (4, 3) motor spin-axis unit vectors
    J         : (3, 3) vehicle inertia matrix (kg·m²)

    Returns
    -------
    tau_mot  : (3,) motor reaction torque
    tau_iner : (3,) inertial / gyroscopic torque
    """
    tau_mot  = J_mp * np.sum(zeta * Omega_dot[:, None], axis=0)
    tau_iner = -np.cross(omega_B, J @ omega_B)
    return tau_mot, tau_iner


def aerodynamic_force_torque(
    v_B: np.ndarray,
    Omega: np.ndarray,
    fx_coeffs: np.ndarray,
    fy_coeffs: np.ndarray,
    fz_coeffs: np.ndarray,
    taux_coeffs: np.ndarray,
    tauy_coeffs: np.ndarray,
    tauz_coeffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Stage 7 — Polynomial aerodynamic drag model.

    Coefficients are fitted from PX4 flight-log data by least-squares
    regression (see docs/coefficient_fitting.md).

    Parameters
    ----------
    v_B : (3,) body-frame velocity [vx, vy, vz] (m/s)
    Omega : (4,) motor speeds (rad/s)
    fx/fy/fz/taux/tauy/tauz coeffs : fitted polynomial coefficients

    Returns
    -------
    f_aero   : (3,) aerodynamic force in body frame (N)
    tau_aero : (3,) aerodynamic torque in body frame (N·m)
    """
    vx, vy, vz = v_B
    v_xy = np.sqrt(vx**2 + vy**2)
    Ob_sq = np.mean(Omega**2)

    f_x = fx_coeffs @ [vx,  vx*abs(vx),  Ob_sq,  vx*Ob_sq]
    f_y = fy_coeffs @ [vy,  vy*abs(vy),  Ob_sq,  vy*Ob_sq]
    f_z = fz_coeffs @ [vz,  vz*abs(vz),  v_xy**2, v_xy*Ob_sq,
                        vz*Ob_sq,  v_xy*vz*Ob_sq]

    tau_x = taux_coeffs @ [vy, vy*abs(vy), Ob_sq, vy*Ob_sq, vy*abs(vy)*Ob_sq]
    tau_y = tauy_coeffs @ [vx, vx*abs(vx), Ob_sq, vx*Ob_sq, vx*abs(vx)*Ob_sq]
    tau_z = tauz_coeffs @ [vx, vy]

    return np.array([f_x, f_y, f_z]), np.array([tau_x, tau_y, tau_z])


def rigid_body_dynamics(
    f_prop: np.ndarray,
    f_aero: np.ndarray,
    tau_prop: np.ndarray,
    tau_mot: np.ndarray,
    tau_aero: np.ndarray,
    tau_iner: np.ndarray,
    q_WB: np.ndarray,
    v_WB: np.ndarray,
    omega_B: np.ndarray,
    Omega_ss: np.ndarray,
    Omega: np.ndarray,
    m: float,
    J: np.ndarray,
    g_W: np.ndarray,
    k_mot: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stage 8 — Newton-Euler equations of motion (state derivative).

    Parameters
    ----------
    f_prop, f_aero                         : (3,) body-frame forces
    tau_prop, tau_mot, tau_aero, tau_iner  : (3,) body-frame torques
    q_WB, v_WB, omega_B                    : current attitude, velocity, body rate
    Omega_ss, Omega                        : steady-state and current motor speeds
    m                                      : float mass (kg)
    J                                      : (3, 3) inertia matrix (kg·m²)
    g_W                                    : (3,) gravity in world frame (m/s²)
    k_mot                                  : float motor time constant (s)

    Returns
    -------
    p_WB_dot, q_WB_dot, v_WB_dot, omega_B_dot, Omega_dot : state derivatives
    """
    p_WB_dot = v_WB

    omega_quat = np.array([0.0, omega_B[0], omega_B[1], omega_B[2]])
    q_WB_dot   = 0.5 * _quat_mult(q_WB, omega_quat)

    R_WB     = _quat2rotm(q_WB)
    v_WB_dot = (R_WB @ (f_prop + f_aero)) / m + g_W

    tau_total  = tau_prop + tau_mot + tau_aero + tau_iner
    omega_B_dot = np.linalg.solve(J, tau_total)   # avoids explicit inversion

    Omega_dot = (Omega_ss - Omega) / k_mot

    return p_WB_dot, q_WB_dot, v_WB_dot, omega_B_dot, Omega_dot


def euler_integration(
    p_WB: np.ndarray,
    q_WB: np.ndarray,
    v_WB: np.ndarray,
    omega_B: np.ndarray,
    Omega: np.ndarray,
    p_WB_dot: np.ndarray,
    q_WB_dot: np.ndarray,
    v_WB_dot: np.ndarray,
    omega_B_dot: np.ndarray,
    Omega_dot: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stage 9 — Forward Euler integration over one time step.

    The quaternion is normalised after integration to prevent drift.

    Parameters
    ----------
    *_prev  : state at iteration i
    *_dot   : state derivatives (output of rigid_body_dynamics)
    dt      : float step time (s)

    Returns
    -------
    p_WB, q_WB (normalised), v_WB, omega_B, Omega : updated state
    """
    p_new  = p_WB  + dt * p_WB_dot
    q_new  = q_WB  + dt * q_WB_dot
    q_new /= np.linalg.norm(q_new)   # re-normalise
    v_new  = v_WB  + dt * v_WB_dot
    w_new  = omega_B + dt * omega_B_dot
    Om_new = Omega + dt * Omega_dot
    return p_new, q_new, v_new, w_new, Om_new


# ============================================================================
# Physical constants — Crazyflie 2.0/2.1 calibration
# ============================================================================

# --- Mass and inertia (published, Forster thesis) ---
m    = 0.027                                       # kg
J    = np.diag([1.4e-5, 1.4e-5, 2.17e-5])         # kg·m²
J_mp = 2.0e-9                                      # kg·m²  (motor+prop rotor)

# --- Gravity (world frame, Z-up) ---
g_W = np.array([0.0, 0.0, -9.81])                 # m/s²

# --- Quad-X frame geometry (Crazyflie arm length 39.7 mm) ---
_arm = 0.0397                                      # m
r_P  = np.array([
    [ _arm,  _arm, 0.0],   # M1 (front-left)
    [-_arm,  _arm, 0.0],   # M2 (front-right)
    [-_arm, -_arm, 0.0],   # M3 (rear-right)
    [ _arm, -_arm, 0.0],   # M4 (rear-left)
])
zeta = np.array([
    [0.0, 0.0,  1.0],      # M1 CCW
    [0.0, 0.0, -1.0],      # M2 CW
    [0.0, 0.0,  1.0],      # M3 CCW
    [0.0, 0.0, -1.0],      # M4 CW
])
spin_sign = zeta[:, 2]

# Mixer matrix derived from geometry (guarantees physical consistency)
motor_geometry          = np.zeros((4, 3))
motor_geometry[:, 0]    = np.sign(r_P[:, 1])      # roll  ~ tau_x = r_y * F_z
motor_geometry[:, 1]    = -np.sign(r_P[:, 0])     # pitch ~ tau_y = -r_x * F_z
motor_geometry[:, 2]    = spin_sign               # yaw   ~ reaction drag sign

# --- Propeller / motor coefficients ---
c_l       = 5.0e-8    # N·(rad/s)⁻²   tuned: hover at ~50% stick
c_d       = 1.25e-9   # N·m·(rad/s)⁻² ratio from Forster
k_mot     = 0.02      # s             first-order lag time constant
Omega_max = 2800.0    # rad/s

# --- Aerodynamic coefficients (fitted from PX4 flight logs) ---
# Raw values in acceleration units; multiplied by m or J_ii for SI forces/torques.
_fx_raw   = np.array([ 6.582314e-02, -3.507132e-02,  0.0,            3.952726e-07])
_fy_raw   = np.array([ 5.738530e-02,  2.022478e-02,  0.0,           -6.291393e-07])
_fz_raw   = np.array([ 3.610184e-02, -1.153378e-01,  2.511414e-02, -5.881881e-07,
                        1.188474e-06, -1.705725e-08])
_taux_raw = np.array([-2.780339e-01,  5.605025e-03,  0.0,            5.870306e-07, -2.282147e-08])
_tauy_raw = np.array([ 3.090840e-01, -3.197782e-02,  0.0,           -4.958482e-07,  5.148323e-08])
_tauz_raw = np.array([-1.432000e-03,  7.798700e-02])

fx_coeffs   = _fx_raw   * m
fy_coeffs   = _fy_raw   * m
fz_coeffs   = _fz_raw   * m
taux_coeffs = _taux_raw * J[0, 0]
tauy_coeffs = _tauy_raw * J[1, 1]
tauz_coeffs = _tauz_raw * J[2, 2]

# --- Battery / ESC model (fitted polynomial) ---
eta            = 0.7
battery_coeffs = np.array([4249.813929, -194.616569, -3453.524622, 50.067265, 208.184375])

# --- PID gains (median across 15 real PX4 flight logs) ---
Kp = np.array([0.150, 0.150, 0.200])
Ki = np.array([0.200, 0.200, 0.100])
Kd = np.array([0.003, 0.003, 0.000])

# --- Command limits ---
cmd_min = 0.0
cmd_max = 1.0

# --- Simulation step ---
dt = 0.01    # 100 Hz

# --- Gamepad → physics scaling ---
C_CMD_SCALE          = 1.0    # full stick → cmd 1.0
RATE_CMD_SCALE       = 2.0    # stick [-1, 1] → body rate [-2, 2] rad/s
THROTTLE_CUT_THRESH  = 0.02   # stick < 2% → motors off

# --- Room bounds ---
ROOM_XY  = 20.0   # ± metres
ROOM_Z   = 40.0   # ceiling height (metres)


# ============================================================================
# Initial conditions
# ============================================================================

p_WB        = np.zeros(3)
q_WB        = np.array([1.0, 0.0, 0.0, 0.0])  # level, no rotation
v_WB        = np.zeros(3)
omega_B     = np.zeros(3)
omega_B_p2  = np.zeros(3)   # i-2 body rate (bootstrapped for PID D-term)
Omega       = np.zeros(4)
I_pid       = np.zeros(3)
U_bat       = 4.2            # volts — full 1S LiPo

Omega_ss    = np.zeros(4)
Omega_dot   = np.zeros(4)

# ============================================================================
# Live simulation loop
# ============================================================================

def _window_open() -> bool:
    """Return True while both visualiser windows are open."""
    bars_open = (hasattr(plot_input_bars, "_fig") and
                 plt.fignum_exists(plot_input_bars._fig.number))
    traj_open = (hasattr(plot_trajectory, "_fig") and
                 plt.fignum_exists(plot_trajectory._fig.number))
    # First iteration: neither window exists yet → keep running
    if not hasattr(plot_input_bars, "_fig"):
        return True
    return bars_open and traj_open


while _window_open():
    # --- Read controller ---
    thrust, roll, pitch, yaw = read_gamepad()
    plot_input_bars({"thrust": thrust, "roll": roll, "pitch": pitch, "yaw": yaw})

    # --- Map stick to physics commands ---
    c_cmd              = max(0.0, thrust) * C_CMD_SCALE
    omega_cmd          = np.array([roll, pitch, yaw]) * RATE_CMD_SCALE
    throttle_cut_flag  = thrust < THROTTLE_CUT_THRESH

    # --- Stage A: PID ---
    u_torque, I_pid = pid_controller(
        omega_cmd, omega_B, omega_B_p2, I_pid,
        throttle_cut_flag, Kp, Ki, Kd, dt,
    )

    # --- Stage B: Mixer ---
    cmd = mixer(u_torque, c_cmd, cmd_min, cmd_max, motor_geometry)

    # When throttle is cut, bypass ESC polynomial entirely —
    # the fitted offset at cmd=0 is non-zero and would spin motors
    if throttle_cut_flag:
        cmd      = np.zeros(4)
        Omega_ss = np.zeros(4)
    else:
        # --- Stage 2: ESC / Battery ---
        Omega_ss, U_bat = esc_battery_model(cmd, U_bat, Omega, eta, battery_coeffs, c_d)

    # --- Stage 3: Motor dynamics ---
    Omega_new, Omega_dot = motor_dynamics(Omega_ss, Omega, k_mot, dt)

    # --- Stage 4: Propeller force/torque ---
    f_props, tau_props = propeller_force_torque(Omega_new, c_l, c_d, spin_sign)

    # --- Stage 5: Aggregate ---
    f_prop, tau_prop = aggregate_forces(f_props, tau_props, r_P)

    # --- Stage 6: Gyroscopic torques ---
    tau_mot, tau_iner = motor_reaction_inertial_torque(Omega_dot, omega_B, J_mp, zeta, J)

    # --- Stage 7: Aerodynamic drag (body-frame velocity needed) ---
    R_WB  = _quat2rotm(q_WB)
    v_B   = R_WB.T @ v_WB
    f_aero, tau_aero = aerodynamic_force_torque(
        v_B, Omega_new,
        fx_coeffs, fy_coeffs, fz_coeffs,
        taux_coeffs, tauy_coeffs, tauz_coeffs,
    )

    # --- Stage 8: Rigid-body dynamics ---
    # Pass OLD Omega (not Omega_new) so motor speed is Euler-integrated
    # exactly once in Stage 9, not double-stepped.
    p_dot, q_dot, v_dot, w_dot, Od = rigid_body_dynamics(
        f_prop, f_aero, tau_prop, tau_mot, tau_aero, tau_iner,
        q_WB, v_WB, omega_B, Omega_ss, Omega,
        m, J, g_W, k_mot,
    )

    # --- Stage 9: Euler integration ---
    p_new, q_new, v_new, w_new, Om_new = euler_integration(
        p_WB, q_WB, v_WB, omega_B, Omega,
        p_dot, q_dot, v_dot, w_dot, Od, dt,
    )

    # --- Carry loop variables forward ---
    omega_B_p2 = omega_B
    p_WB       = p_new
    q_WB       = q_new
    v_WB       = v_new
    omega_B    = w_new
    Omega      = np.maximum(Om_new, 0.0)   # propellers can't spin backwards

    # --- Ground collision (hard floor at z = 0) ---
    if p_WB[2] <= 0.0:
        p_WB[2] = 0.0
        if v_WB[2] < 0.0:
            v_WB[2] = 0.0

    # --- Room boundary check ---
    if (abs(p_WB[0]) > ROOM_XY or abs(p_WB[1]) > ROOM_XY or p_WB[2] > ROOM_Z):
        print(f"[SIM] Drone left the room at {p_WB} — simulation ended.")
        break

    plot_trajectory(p_WB, q_WB)
