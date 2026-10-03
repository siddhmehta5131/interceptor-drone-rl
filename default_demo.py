"""
swift_live_demo.py

VARIABLE TAXONOMY (per project convention)
============================================================
 TRUE INPUT   : thrust, roll, pitch, yaw
                Read directly from the Amkette Evo Gamepad Pro 4.
                Cannot be derived from anything else in the system.

 TRUE OUTPUT  : p_WB (3,) position + q_WB (4,) attitude quaternion
                The only values the human user actually cares about --
                a fully defined pose of the object in 3-space.
============================================================

Three functions:
  1. plot_input_bars(u_input)     -- input:  thrust, roll, pitch, yaw
                                      output: none (just plots)
  2. plot_trajectory(p_WB, q_WB)  -- input:  3D position, attitude quaternion
                                      output: none (just plots)
  3. read_gamepad()               -- input:  none (connects to controller)
                                      output: thrust, roll, pitch, yaw
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

# No additional libraries beyond numpy (already imported above) are required by the physics-simulator functions below.


def plot_trajectory(p_WB, q_WB):
    """Live 3D visualiser for the TRUE OUTPUT (pose).

    plot_trajectory(p_WB, q_WB)

    p_WB : (3,) position in world frame        [x, y, z]
    q_WB : (4,) attitude quaternion, [w, x, y, z], body -> world

    Opens its own figure on first call (state cached on the function
    object), then only updates the trail and the body-frame arrows on
    every subsequent call. Knows nothing about how p_WB/q_WB were
    generated -- fully decoupled from the source. Drone is represented
    as a coordinate frame (3 arrows), not a mesh.
    """
    if not hasattr(plot_trajectory, "_fig") or not plt.fignum_exists(plot_trajectory._fig.number):
        fig = plt.figure(num="Drone trajectory (TRUE OUTPUT)")
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("X_W")
        ax.set_ylabel("Y_W")
        ax.set_zlabel("Z_W")
        ax.view_init(elev=20, azim=35)

        # Fixed room bounds: X/Y in [-20, 20], Z (up) in [0, 40].
        # XY plane (Z=0) is the ground.
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        ax.set_zlim(0, 40)
        ax.set_box_aspect((40, 40, 40))  # keep room proportions undistorted

        # Draw the 6 room walls as a wireframe box (ground, ceiling, 4 side walls)
        _room_x = [-20, 20]
        _room_y = [-20, 20]
        _room_z = [0, 40]
        for zz in _room_z:  # ground (z=0) and ceiling (z=40)
            ax.plot([_room_x[0], _room_x[1]], [_room_y[0], _room_y[0]], [zz, zz], color="gray", linewidth=0.7)
            ax.plot([_room_x[0], _room_x[1]], [_room_y[1], _room_y[1]], [zz, zz], color="gray", linewidth=0.7)
            ax.plot([_room_x[0], _room_x[0]], [_room_y[0], _room_y[1]], [zz, zz], color="gray", linewidth=0.7)
            ax.plot([_room_x[1], _room_x[1]], [_room_y[0], _room_y[1]], [zz, zz], color="gray", linewidth=0.7)
        for xx in _room_x:  # vertical edges
            for yy in _room_y:
                ax.plot([xx, xx], [yy, yy], [_room_z[0], _room_z[1]], color="gray", linewidth=0.7)

        trail_line, = ax.plot([], [], [], color=(0.1, 0.1, 0.1), linewidth=1.5, label="trail")

        arrow_len = 1.0  # cosmetic only
        h_body_x = ax.quiver(0, 0, 0, 1, 0, 0, length=arrow_len, color="r", label="x_B (front)")
        h_body_y = ax.quiver(0, 0, 0, 0, 1, 0, length=arrow_len, color="g", label="y_B")
        h_body_z = ax.quiver(0, 0, 0, 0, 0, 1, length=arrow_len, color="b", label="z_B")

        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

        plot_trajectory._fig = fig
        plot_trajectory._ax = ax
        plot_trajectory._trail_line = trail_line
        plot_trajectory._trail_xyz = [[], [], []]
        plot_trajectory._h_body_x = h_body_x
        plot_trajectory._h_body_y = h_body_y
        plot_trajectory._h_body_z = h_body_z
        plot_trajectory._arrow_len = arrow_len

    ax = plot_trajectory._ax
    R_WB = quat2rotm_manual(q_WB)
    x_B, y_B, z_B = R_WB[:, 0], R_WB[:, 1], R_WB[:, 2]

    trail_xyz = plot_trajectory._trail_xyz
    trail_xyz[0].append(p_WB[0])
    trail_xyz[1].append(p_WB[1])
    trail_xyz[2].append(p_WB[2])
    plot_trajectory._trail_line.set_data(np.asarray(trail_xyz[0]), np.asarray(trail_xyz[1]))
    plot_trajectory._trail_line.set_3d_properties(np.asarray(trail_xyz[2]))

    arrow_len = plot_trajectory._arrow_len
    plot_trajectory._h_body_x.remove()
    plot_trajectory._h_body_y.remove()
    plot_trajectory._h_body_z.remove()
    plot_trajectory._h_body_x = ax.quiver(p_WB[0], p_WB[1], p_WB[2], x_B[0], x_B[1], x_B[2], length=arrow_len, color="r")
    plot_trajectory._h_body_y = ax.quiver(p_WB[0], p_WB[1], p_WB[2], y_B[0], y_B[1], y_B[2], length=arrow_len, color="g")
    plot_trajectory._h_body_z = ax.quiver(p_WB[0], p_WB[1], p_WB[2], z_B[0], z_B[1], z_B[2], length=arrow_len, color="b")

    plt.draw()
    plt.pause(0.001)


def plot_input_bars(u_input):
    """Live 4-bar visualiser for the TRUE INPUT vector.

    plot_input_bars(u_input)

    u_input : (4,) array-like OR dict with keys
              thrust, roll, pitch, yaw
              Order used for the bars: [thrust, roll, pitch, yaw]

    Opens its own figure on first call (state cached on the function
    object), then only updates bar heights on every subsequent call.
    No other function writes to this figure.
    """
    if isinstance(u_input, dict):
        vals = [u_input["thrust"], u_input["roll"], u_input["pitch"], u_input["yaw"]]
    else:
        vals = list(u_input)

    if not hasattr(plot_input_bars, "_fig") or not plt.fignum_exists(plot_input_bars._fig.number):
        fig, ax = plt.subplots(num="Transmitter input (TRUE INPUT)")
        labels = ["thrust", "roll", "pitch", "yaw"]
        bars = ax.bar(labels, vals)
        ax.set_ylim(-1, 1)
        ax.grid(True)
        ax.set_ylabel("normalised command")
        ax.set_title("Live transmitter commands (TRUE INPUT)")

        plot_input_bars._fig = fig
        plot_input_bars._ax = ax
        plot_input_bars._bars = bars

    for bar, v in zip(plot_input_bars._bars, vals):
        bar.set_height(v)

    plt.draw()
    plt.pause(0.001)


def read_gamepad(deadzone_val=0.08):
    """Read one sample of the TRUE INPUT from a connected gamepad.

    read_gamepad() -> (thrust, roll, pitch, yaw)

    No input arguments -- connects to the gamepad (e.g. Amkette Evo
    Gamepad Pro 4) on first call, state cached on the function object,
    then just polls current stick position on every subsequent call.
    Pure input: no plotting, no other side effects.

    Axis mapping / deadzone follow the same convention as the
    gamepad-drone-hud.html reference visualiser:
        axis0 = left  stick X -> yaw
        axis1 = left  stick Y -> thrust  (flipped so up = positive)
        axis2 = right stick X -> roll
        axis3 = right stick Y -> pitch   (flipped so up = positive)

    Axis indices are controller/driver dependent -- if a channel comes
    back swapped, remap the axis0..3 assignments below.
    """
    import pygame

    if not hasattr(read_gamepad, "_joy"):
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError(
                "No gamepad detected. Connect the Amkette Evo Gamepad Pro 4 and try again."
            )
        joy = pygame.joystick.Joystick(0)
        joy.init()
        print(f"Connected to gamepad: {joy.get_name()}")
        read_gamepad._joy = joy

    joy = read_gamepad._joy
    pygame.event.pump()

    def dz(v):
        return 0.0 if abs(v) < deadzone_val else v

    n_axes = joy.get_numaxes()
    n_axes = joy.get_numaxes()
    lx = dz(joy.get_axis(0)) if n_axes > 0 else 0.0
    ly = dz(joy.get_axis(1)) if n_axes > 1 else 0.0
    rx = dz(joy.get_axis(3)) if n_axes > 3 else 0.0   # roll now on axis3, not axis2
    ry = dz(joy.get_axis(4)) if n_axes > 4 else 0.0   # pitch now on axis4, not axis3

    thrust = -ly
    yaw    =  lx
    pitch  = -ry
    roll   =  rx

    return thrust, roll, pitch, yaw


def quat2rotm_manual(q):
    """Dependency-free quaternion -> rotation matrix. q = [w, x, y, z].
    Body -> world: columns of R are the body x/y/z axes in world coords.
    """
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),   1 - 2 * (x**2 + z**2),   2 * (y * z - w * x)],
        [2 * (x * z - w * y),   2 * (y * z + w * x),     1 - 2 * (x**2 + y**2)],
    ])


def quat_mult(q1, q2):
    """Hamilton product of two quaternions, [w, x, y, z] convention."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


# ==================== STAGE FUNCTIONS ====================
# Ten functions, one per stage of the paper-derived physics pipeline.
# Each function's inputs/outputs match exactly what was derived on paper
# for that stage.

def pid_controller(omega_cmd, omega_B_prev, omega_B_prev2, I_prev, throttle_cut_flag,
                    Kp, Ki, Kd, dt):
    """Stage A -- Rate PID controller (per axis: roll, pitch, yaw).

    Inputs
    ------
    omega_cmd        : (3,) desired body rates [wx_cmd, wy_cmd, wz_cmd]  (TRUE INPUT, this iteration i)
    omega_B_prev      : (3,) body rates omega_B,i-1  (loop variable)
    omega_B_prev2     : (3,) body rates omega_B,i-2  (loop variable, needed only for D-term)
    I_prev            : (3,) integrator state I_x,i-1 / I_y,i-1 / I_z,i-1  (loop variable)
    throttle_cut_flag : bool, whether throttle is cut this iteration
    Kp, Ki, Kd         : (3,) proportional / integral / derivative gains, per axis  (constants)
    dt                 : step time  (constant)

    Outputs
    -------
    u        : (3,) desired torque-correction command [u_x, u_y, u_z]
    I_new    : (3,) updated integrator state, to be carried to iteration i+1
    """
    e = omega_cmd - omega_B_prev
    P = Kp * e
    if throttle_cut_flag:
        I_new = np.zeros(3)
    else:
        I_new = I_prev + Ki * e * dt
    D = -Kd * (omega_B_prev - omega_B_prev2) / dt
    u = P + I_new + D
    return u, I_new


def mixer(u, c_cmd, cmd_min, cmd_max, motor_geometry):
    """Stage B -- Mixer.

    Inputs
    ------
    u               : (3,) torque-correction command [u_x, u_y, u_z]  (output of pid_controller)
    c_cmd           : scalar collective thrust command  (TRUE INPUT, this iteration i)
    cmd_min, cmd_max : scalar saturation limits  (constants)
    motor_geometry   : motor layout signs / positions r_P,j and spin directions zeta_j  (constants)

    Outputs
    -------
    cmd : (4,) individual motor commands [cmd_1, cmd_2, cmd_3, cmd_4]
    """
    cmd_raw = c_cmd + motor_geometry @ u   # motor_geometry: (4,3) sign matrix for [u_x,u_y,u_z] per motor

    cmd_max_val = np.max(cmd_raw)

    denom = cmd_max_val - c_cmd
    if cmd_max_val > cmd_max and abs(denom) > 1e-9:
        scale = (cmd_max - c_cmd) / denom
        u = scale * u
        cmd_raw = c_cmd + motor_geometry @ u

    cmd = np.clip(cmd_raw, cmd_min, cmd_max)
    return cmd


def esc_battery_model(cmd, U_bat_prev, Omega_prev, eta, battery_coeffs, dt):
    """Stage 2 -- ESC / Battery model.

    Inputs
    ------
    cmd            : (4,) motor commands  (output of mixer)
    U_bat_prev     : scalar battery voltage U_bat,i-1  (loop variable)
    Omega_prev     : (4,) motor speeds Omega_i-1, used for instantaneous power draw  (loop variable)
    eta            : scalar motor efficiency  (constant)
    battery_coeffs : coefficients for Eq. 6 / battery voltage model  (constants, unpublished)
    dt             : step time  (constant)

    Outputs
    -------
    Omega_ss : (4,) steady-state target motor speed per motor
    U_bat_new : scalar updated battery voltage, to be carried to iteration i+1
    """
    P_mot = (c_d * Omega_prev**3) / eta   # c_d: global constant

    # Battery voltage dynamics (Eq. 6's own source model, ref. 46, is unspecified --
    # placeholder: assume constant voltage until a real battery model is supplied)
    U_bat_new = U_bat_prev

    Omega_ss = (battery_coeffs[0]
                + battery_coeffs[1] * U_bat_prev
                + battery_coeffs[2] * np.sqrt(np.clip(cmd, 0, None))
                + battery_coeffs[3] * cmd
                + battery_coeffs[4] * U_bat_prev * np.sqrt(np.clip(cmd, 0, None)))
    return Omega_ss, U_bat_new


def motor_dynamics(Omega_ss, Omega_prev, k_mot, dt):
    """Stage 3 -- Motor dynamics (first-order lag).

    Inputs
    ------
    Omega_ss   : (4,) steady-state target motor speed  (output of esc_battery_model)
    Omega_prev : (4,) motor speeds Omega_i-1  (loop variable)
    k_mot      : scalar motor time constant  (constant)
    dt         : step time  (constant)

    Outputs
    -------
    Omega     : (4,) actual motor speeds this iteration
    Omega_dot : (4,) motor speed derivative this iteration (reused in Stage 6)
    """
    Omega_dot = (1.0 / k_mot) * (Omega_ss - Omega_prev)
    Omega = Omega_prev + dt * Omega_dot
    return Omega, Omega_dot


def propeller_force_torque(Omega, c_l, c_d, spin_sign):
    """Stage 4 -- Propeller force/torque.

    Inputs
    ------
    Omega      : (4,) actual motor speeds  (output of motor_dynamics)
    c_l        : scalar propeller lift coefficient  (constant)
    c_d        : scalar propeller drag coefficient  (constant)
    spin_sign  : (4,) +1/-1 per motor, its spin direction (CW/CCW) --
                 the reaction drag torque points opposite to spin, so
                 motors spinning opposite ways must have opposite-signed
                 drag torque, or net yaw torque never cancels at hover.

    Outputs
    -------
    f_props : (4,3) per-propeller force vectors
    tau_props : (4,3) per-propeller torque vectors
    """
    f_props = np.zeros((4, 3))
    tau_props = np.zeros((4, 3))
    for j in range(4):
        f_props[j] = np.array([0.0, 0.0, c_l * Omega[j]**2])
        tau_props[j] = np.array([0.0, 0.0, spin_sign[j] * c_d * Omega[j]**2])
    return f_props, tau_props


def aggregate_forces(f_props, tau_props, r_P):
    """Stage 5 -- Aggregate propeller force/torque.

    Inputs
    ------
    f_props   : (4,3) per-propeller force vectors    (output of propeller_force_torque)
    tau_props : (4,3) per-propeller torque vectors    (output of propeller_force_torque)
    r_P       : (4,3) propeller position vectors relative to center of mass  (constant)

    Outputs
    -------
    f_prop   : (3,) total propeller force
    tau_prop : (3,) total propeller torque
    """
    f_prop = np.sum(f_props, axis=0)
    tau_prop = np.zeros(3)
    for j in range(4):
        tau_prop += tau_props[j] + np.cross(r_P[j], f_props[j])
    return f_prop, tau_prop


def motor_reaction_inertial_torque(Omega_dot, omega_B_prev, J_mp, zeta, J):
    """Stage 6 -- Motor reaction torque + inertial (gyroscopic) torque.

    Inputs
    ------
    Omega_dot    : (4,) motor speed derivatives  (output of motor_dynamics)
    omega_B_prev : (3,) body rates omega_B,i-1  (loop variable)
    J_mp         : scalar combined motor+propeller inertia  (constant)
    zeta         : (4,3) motor spin-axis unit vectors  (constant)
    J            : (3,3) vehicle inertia matrix  (constant)

    Outputs
    -------
    tau_mot  : (3,) motor reaction torque
    tau_iner : (3,) inertial/gyroscopic torque
    """
    tau_mot = J_mp * np.sum(zeta * Omega_dot[:, None], axis=0)
    tau_iner = -np.cross(omega_B_prev, J @ omega_B_prev)
    return tau_mot, tau_iner


def aerodynamic_force_torque(v_B_prev, Omega, aero_coeffs):
    """Stage 7 -- Aerodynamic force/torque.

    Inputs
    ------
    v_B_prev    : (3,) body-frame velocity v_B,i-1  (loop variable)
    Omega       : (4,) actual motor speeds  (output of motor_dynamics)
    aero_coeffs : coefficients for the aero polynomial model  (constants, unpublished)

    Outputs
    -------
    f_aero   : (3,) aerodynamic force vector
    tau_aero : (3,) aerodynamic torque vector
    """
    vx, vy, vz = v_B_prev
    v_xy = np.sqrt(vx**2 + vy**2)
    Omega_bar_sq = np.mean(Omega**2)

    f_x = fx_coeffs @ np.array([vx, vx*abs(vx), Omega_bar_sq, vx*Omega_bar_sq])
    f_y = fy_coeffs @ np.array([vy, vy*abs(vy), Omega_bar_sq, vy*Omega_bar_sq])
    f_z = fz_coeffs @ np.array([vz, vz*abs(vz), v_xy**2, v_xy*Omega_bar_sq,
                                 vz*Omega_bar_sq, v_xy*vz*Omega_bar_sq])

    tau_x = taux_coeffs @ np.array([vy, vy*abs(vy), Omega_bar_sq,
                                     vy*Omega_bar_sq, vy*abs(vy)*Omega_bar_sq])
    tau_y = tauy_coeffs @ np.array([vx, vx*abs(vx), Omega_bar_sq,
                                     vx*Omega_bar_sq, vx*abs(vx)*Omega_bar_sq])
    tau_z = tauz_coeffs @ np.array([vx, vy])

    f_aero = np.array([f_x, f_y, f_z])
    tau_aero = np.array([tau_x, tau_y, tau_z])
    return f_aero, tau_aero


def rigid_body_dynamics(f_prop, f_aero, tau_prop, tau_mot, tau_aero, tau_iner,
                         q_WB_prev, v_WB_prev, omega_B_prev, Omega_ss, Omega_prev,
                         m, J, g_W, k_mot):
    """Stage 8 -- Rigid body dynamics (Eq. 1, assembled state derivative).

    Inputs
    ------
    f_prop, f_aero               : (3,) force contributions   (outputs of aggregate_forces, aerodynamic_force_torque)
    tau_prop, tau_mot, tau_aero, tau_iner : (3,) torque contributions  (outputs of aggregate_forces,
                                             motor_reaction_inertial_torque, aerodynamic_force_torque)
    q_WB_prev   : (4,) attitude quaternion q_WB,i-1  (loop variable)
    v_WB_prev   : (3,) velocity v_WB,i-1  (loop variable)
    omega_B_prev : (3,) body rates omega_B,i-1  (loop variable)
    Omega_ss    : (4,) steady-state motor speed  (output of esc_battery_model)
    Omega_prev  : (4,) motor speeds Omega_i-1  (loop variable)
    m, J, g_W, k_mot : mass, inertia matrix, gravity vector, motor time constant  (constants)

    Outputs
    -------
    p_WB_dot     : (3,) position derivative
    q_WB_dot     : (4,) attitude quaternion derivative
    v_WB_dot     : (3,) velocity derivative
    omega_B_dot  : (3,) body rate derivative
    Omega_dot    : (4,) motor speed derivative
    """
    p_WB_dot = v_WB_prev

    omega_quat = np.array([0.0, omega_B_prev[0], omega_B_prev[1], omega_B_prev[2]])
    q_WB_dot = 0.5 * quat_mult(q_WB_prev, omega_quat)

    R_WB = quat2rotm_manual(q_WB_prev)
    v_WB_dot = (1.0 / m) * (R_WB @ (f_prop + f_aero)) + g_W

    J_inv = np.linalg.inv(J)
    omega_B_dot = J_inv @ (tau_prop + tau_mot + tau_aero + tau_iner)

    Omega_dot = (1.0 / k_mot) * (Omega_ss - Omega_prev)

    return p_WB_dot, q_WB_dot, v_WB_dot, omega_B_dot, Omega_dot


def euler_integration(p_WB_prev, q_WB_prev, v_WB_prev, omega_B_prev, Omega_prev,
                       p_WB_dot, q_WB_dot, v_WB_dot, omega_B_dot, Omega_dot, dt):
    """Stage 9 -- Euler integration.

    Inputs
    ------
    p_WB_prev, q_WB_prev, v_WB_prev, omega_B_prev, Omega_prev : loop variables at iteration i-1
    p_WB_dot, q_WB_dot, v_WB_dot, omega_B_dot, Omega_dot        : state derivatives (output of rigid_body_dynamics)
    dt : step time  (constant)

    Outputs
    -------
    p_WB      : (3,) new position                -- TRUE OUTPUT
    q_WB      : (4,) new attitude quaternion      -- TRUE OUTPUT
    v_WB      : (3,) new velocity                 -- loop variable, carried to iteration i+1
    omega_B   : (3,) new body rate                -- loop variable, carried to iteration i+1
    Omega     : (4,) new motor speeds              -- loop variable, carried to iteration i+1
    """
    p_WB = p_WB_prev + dt * p_WB_dot

    q_WB = q_WB_prev + dt * q_WB_dot
    q_WB = q_WB / np.linalg.norm(q_WB)

    v_WB = v_WB_prev + dt * v_WB_dot
    omega_B = omega_B_prev + dt * omega_B_dot
    Omega = Omega_prev + dt * Omega_dot

    return p_WB, q_WB, v_WB, omega_B, Omega


# ==================== TRUE INPUT: declare ====================
thrust, roll, pitch, yaw = read_gamepad()

# ==================== TRUE OUTPUT: declare ====================
p_WB = np.array([0.0, 0.0, 0.0])       # origin (still static for now)
q_WB = np.array([1.0, 0.0, 0.0, 0.0])  # identity orientation [w x y z]

# ==================== VARIABLES ====================
# Loop variables -- carried from iteration i-1 to iteration i by the
# physics pipeline (Stage A through Stage 9).

v_WB = np.zeros(3)          # (3,)   velocity, world frame           v_WB,i-1
omega_B = np.zeros(3)        # (3,)   body angular rate                omega_B,i-1
omega_B_prev2 = np.zeros(3)   # (3,)   body angular rate, i-2            (needed only for Stage A D-term)
Omega = np.zeros(4)          # (4,)   motor speeds                     Omega_i-1
Omega_dot = np.zeros(4)       # (4,)   motor speed derivative            Omega_dot_i-1
U_bat = 4.2                  # scalar battery voltage                  U_bat,i-1 (full charge, 1S LiPo)
I_pid = np.zeros(3)          # (3,)   PID integrator state              I_x,i-1 / I_y,i-1 / I_z,i-1
cmd = np.zeros(4)            # (4,)   motor commands                    cmd_i
u_torque = np.zeros(3)        # (3,)   torque-correction command         u_i
Omega_ss = np.zeros(4)        # (4,)   steady-state target motor speed   Omega_i,ss
f_props = np.zeros((4, 3))    # (4,3)  per-propeller forces
tau_props = np.zeros((4, 3))  # (4,3)  per-propeller torques
f_prop = np.zeros(3)          # (3,)   aggregate propeller force
tau_prop = np.zeros(3)        # (3,)   aggregate propeller torque
tau_mot = np.zeros(3)         # (3,)   motor reaction torque
tau_iner = np.zeros(3)        # (3,)   inertial/gyroscopic torque
f_aero = np.zeros(3)          # (3,)   aerodynamic force
tau_aero = np.zeros(3)        # (3,)   aerodynamic torque
p_WB_dot = np.zeros(3)        # (3,)   position derivative
q_WB_dot = np.zeros(4)        # (4,)   attitude quaternion derivative
v_WB_dot = np.zeros(3)        # (3,)   velocity derivative
omega_B_dot = np.zeros(3)     # (3,)   body rate derivative
throttle_cut_flag = False     # bool  derived from c_cmd,i each iteration

# ==================== CONSTANTS ====================
# Fixed at simulation start. Values below are the best available:
# real published Crazyflie 2.0/2.1 system-ID values where a public
# source exists; otherwise a physically-reasonable best guess,
# clearly commented as such.

# -- Mass & inertial properties --
m = 0.027                                        # kg -- Crazyflie 2.0/2.1 (published, Forster thesis)
J = np.diag([1.4e-5, 1.4e-5, 2.17e-5])           # kg*m^2 -- Crazyflie 2.0/2.1 [Ixx, Iyy, Izz] (published)
J_mp = 2.0e-9                                    # kg*m^2 -- best guess, small nano-quad motor+prop rotor inertia

# -- Gravity --
g_W = np.array([0.0, 0.0, -9.81])                # m/s^2

# -- Frame geometry (per motor, 4 motors), quad-X, Crazyflie arm length --
_arm = 0.0397                                    # m -- Crazyflie arm length (published)
r_P = np.array([
    [ _arm,  _arm, 0.0],   # M1
    [-_arm,  _arm, 0.0],   # M2
    [-_arm, -_arm, 0.0],   # M3
    [ _arm, -_arm, 0.0],   # M4
])
zeta = np.array([                                # spin directions -- sourced from Crazyflie mixer sign pattern
    [0.0, 0.0,  1.0],   # M1
    [0.0, 0.0, -1.0],   # M2
    [0.0, 0.0,  1.0],   # M3
    [0.0, 0.0, -1.0],   # M4
])
spin_sign = zeta[:, 2]                           # (4,), +1/-1 per motor, drives drag-torque sign in Stage 4

# Mixer sign matrix (4,3), derived directly from r_P and spin_sign so it
# is guaranteed physically consistent with the torque these positions
# actually produce (rather than an externally-sourced firmware pattern
# assuming a different, unverified motor-position numbering):
#   roll  column  ~ sign(r_P y-component)   (tau_x = r_y * f_z)
#   pitch column  ~ -sign(r_P x-component)  (tau_y = -r_x * f_z)
#   yaw   column  = spin_sign               (reaction torque direction)
motor_geometry = np.zeros((4, 3))
motor_geometry[:, 0] = np.sign(r_P[:, 1])
motor_geometry[:, 1] = -np.sign(r_P[:, 0])
motor_geometry[:, 2] = spin_sign

# -- Propeller / motor aerodynamic coefficients --
# Tuned so hover occurs at ~50% stick with the ESC model below.
# Hover Omega ~1150 rad/s: c_l = m*g / (4 * 1150^2) ≈ 5e-8.
c_l = 5.0e-8                    # N/(rad/s)^2 -- tuned for simulation
c_d = 1.25e-9                   # N*m/(rad/s)^2 -- same c_d/c_l ratio as Forster
k_mot = 0.02                    # s -- best guess, small-motor first-order lag time constant
Omega_max = 2800.0              # rad/s -- best guess, typical small-quad max motor speed

# -- Body aerodynamic coefficients (Stage 7, unpublished -- best-guess placeholders) --
# NOTE: Standalone Omega_bar_sq terms (index 2 in fx/fy/taux/tauy) must be
# zero -- a drone hovering in still air has no net lateral aero force.
# Linear drag scaled for a 27 g nano-quad (~0.01 N per m/s is reasonable).
fx_coeffs   = np.array([-0.01, -0.002, 0.0, -1e-10])
fy_coeffs   = np.array([-0.01, -0.002, 0.0, -1e-10])
fz_coeffs   = np.array([-0.01, -0.002, -0.002, 0.0, -1e-10, 0.0])
taux_coeffs = np.array([-0.0001, -0.00002, 0.0, -1e-12, 0.0])
tauy_coeffs = np.array([-0.0001, -0.00002, 0.0, -1e-12, 0.0])
tauz_coeffs = np.array([-0.0005, -0.0005])

# -- Battery / ESC model (Stage 2, unpublished -- best-guess placeholders) --
eta = 0.7                       # motor efficiency -- best guess
# ESC model coefficients tuned so cmd=1.0 at full battery gives
# Omega_ss ~ 2000 rad/s (above hover speed ~1500 rad/s).
battery_coeffs = np.array([0.0, 0.0, 800.0, 1200.0, 0.0])   # Eq. 6 coefficients -- tuned for new c_l

# -- Low-level controller / PID gains (Stage A -- best-guess, informed by
#    an old historical Crazyflie firmware default Kp ~ 3.5, since current
#    numeric defaults are not published outside the firmware source) --
Kp = np.array([0.04, 0.04, 0.04])
Ki = np.array([0.01, 0.01, 0.01])
Kd = np.array([0.0, 0.0, 0.002])

# -- Mixer / saturation limits (Stage B) --
cmd_min = 0.0                    # scalar, minimum motor command
cmd_max = 1.0                    # scalar, maximum motor command (normalised)

# -- Simulation / integration settings --
dt = 0.01                      # scalar, step time (100 Hz)

# ==================== INITIAL ====================
# Initial conditions at iteration i=0: drone at rest, at the origin,
# level attitude, full battery.

p_WB_0 = np.zeros(3)                       # at origin
q_WB_0 = np.array([1.0, 0.0, 0.0, 0.0])    # identity quaternion -- level, no rotation
v_WB_0 = np.zeros(3)                       # at rest
omega_B_0 = np.zeros(3)                    # not rotating
omega_B_neg1 = np.zeros(3)                 # bootstrap value for Stage A D-term at i=1
Omega_0 = np.zeros(4)                      # motors off / spun down at rest
I_pid_0 = np.zeros(3)                      # no accumulated PID error yet
U_bat_0 = 4.2                              # volts -- full charge, 1S LiPo

# Wire initial conditions into the loop variables
p_WB = p_WB_0.copy()
q_WB = q_WB_0.copy()
v_WB = v_WB_0.copy()
omega_B = omega_B_0.copy()
omega_B_prev2 = omega_B_neg1.copy()
Omega = Omega_0.copy()
I_pid = I_pid_0.copy()
U_bat = U_bat_0

# TRUE INPUT -> physics-pipeline command scaling.
# The gamepad stick springs back to center (thrust=0).  Only the
# positive half of the stick travel commands motor thrust; negative
# is ignored (propellers can't push downward).
C_CMD_SCALE = 1.0           # full stick -> cmd 1.0
RATE_CMD_SCALE = 2.0        # roll/pitch/yaw stick [-1,1] -> desired body rate [-2, 2] rad/s
THROTTLE_CUT_THRESHOLD = 0.02  # stick near zero is treated as throttle-cut (motors off)

# ==================== live loop ====================
while plt.fignum_exists(plot_input_bars._fig.number) if hasattr(plot_input_bars, "_fig") else True:
    thrust, roll, pitch, yaw = read_gamepad()
    plot_input_bars({"thrust": thrust, "roll": roll, "pitch": pitch, "yaw": yaw})

    # TRUE INPUT -> physics command
    # Only positive stick travel gives thrust (stick springs to center = 0)
    c_cmd = max(0.0, thrust) * C_CMD_SCALE
    omega_cmd = np.array([roll, pitch, yaw]) * RATE_CMD_SCALE
    throttle_cut_flag = thrust < THROTTLE_CUT_THRESHOLD

    # Stage A -- PID controller
    u_torque, I_pid = pid_controller(omega_cmd, omega_B, omega_B_prev2, I_pid,
                                      throttle_cut_flag, Kp, Ki, Kd, dt)

    # Stage B -- Mixer
    cmd = mixer(u_torque, c_cmd, cmd_min, cmd_max, motor_geometry)

    # Stage 2 -- ESC / Battery model
    Omega_ss, U_bat = esc_battery_model(cmd, U_bat, Omega, eta, battery_coeffs, dt)

    # Stage 3 -- Motor dynamics
    Omega_new, Omega_dot = motor_dynamics(Omega_ss, Omega, k_mot, dt)

    # Stage 4 -- Propeller force/torque
    f_props, tau_props = propeller_force_torque(Omega_new, c_l, c_d, spin_sign)

    # Stage 5 -- Aggregate propeller force/torque
    f_prop, tau_prop = aggregate_forces(f_props, tau_props, r_P)

    # Stage 6 -- Motor reaction torque + inertial torque
    tau_mot, tau_iner = motor_reaction_inertial_torque(Omega_dot, omega_B, J_mp, zeta, J)

    # Stage 7 -- Aerodynamic force/torque
    # aero model needs body-frame velocity, not world-frame
    R_WB = quat2rotm_manual(q_WB)
    v_B = R_WB.T @ v_WB
    f_aero, tau_aero = aerodynamic_force_torque(v_B, Omega_new, None)

    # Stage 8 -- Rigid body dynamics (assemble state derivative)
    p_WB_dot, q_WB_dot, v_WB_dot, omega_B_dot, Omega_dot = rigid_body_dynamics(
        f_prop, f_aero, tau_prop, tau_mot, tau_aero, tau_iner,
        q_WB, v_WB, omega_B, Omega_ss, Omega_new,
        m, J, g_W, k_mot)

    # Stage 9 -- Euler integration
    p_WB_new, q_WB_new, v_WB_new, omega_B_new, Omega_final = euler_integration(
        p_WB, q_WB, v_WB, omega_B, Omega_new,
        p_WB_dot, q_WB_dot, v_WB_dot, omega_B_dot, Omega_dot, dt)

    # Carry loop variables forward to iteration i+1
    omega_B_prev2 = omega_B
    p_WB = p_WB_new
    q_WB = q_WB_new
    v_WB = v_WB_new
    omega_B = omega_B_new
    Omega = Omega_final

    # Ground-plane collision: drone rests on the floor (z=0 clamp).
    if p_WB[2] < 0.0:
        p_WB[2] = 0.0
        if v_WB[2] < 0.0:
            v_WB[2] = 0.0

    # Room boundary enforcement: walls (±20 XY) and ceiling (z=40).
    # If the drone hits these, end the simulation.
    out_of_bounds = (
        p_WB[0] < -20.0 or p_WB[0] > 20.0 or
        p_WB[1] < -20.0 or p_WB[1] > 20.0 or
        p_WB[2] > 40.0
    )
    if out_of_bounds:
        print(f"Drone left the room at position {p_WB} -- ending simulation.")
        break

    plot_trajectory(p_WB, q_WB)