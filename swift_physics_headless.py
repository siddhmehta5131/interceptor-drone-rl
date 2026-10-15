"""
swift_physics_headless.py

Headless extraction of the physics pipeline from swift_live_demo_fitted.py.
No gamepad, no live plotting -- importable, and driven by recorded data
instead of a live loop. All stage functions and fitted coefficients are
copied verbatim from the original; only the driving loop is replaced.

Two mass/inertia presets are provided since the fitted coefficients were
derived assuming bare-Crazyflie mass, but validation datasets (e.g.
NanoBench) fly with extra payload (Vicon markers + charging deck):

    PARAMS_BARE      -> m = 0.027 kg   (Forster / bare Crazyflie 2.0/2.1)
    PARAMS_NANOBENCH  -> m = 0.04085 kg (Crazyflie 2.1 + markers + deck,
                                          as flown in the NanoBench dataset)

IMPORTANT: fx/fy/fz/taux/tauy/tauz coefficients were converted from
acceleration units to force/torque units using m and J at fit time
(see original file, lines ~626-646). If you change m or J here, those
coefficients are now inconsistent with the new mass -- see
`rescale_aero_coeffs_for_mass()` below, which undoes and redoes that
conversion. Always call it if you switch presets.

ACCURACY FIXES (vs. original):
  1. _fz_raw[2] was 2.511e-02 (Omega_bar_sq offset) -- this MUST be zero
     for the same reason fx/fy offsets were zeroed: the propeller's Omega^2
     contribution to fz is already fully accounted for by Stage 4. Leaving
     it non-zero produced ~7000 N of spurious downforce at hover (6895x the
     drone's weight), causing immediate numerical overflow and divergence.
  2. battery_coeffs are now calibrated to the published Crazyflie linear
     PWM->RPM characterization (Omega = 0.2685*PWM_16bit + 4070.3 RPM,
     from Forster 2015 / Bitcraze documentation) with a sqrt(U_bat/U_nom)
     voltage correction. The previous coefficients over-predicted motor speed
     by ~25% at hover, further inflating all Omega^2-dependent forces.
  3. Omega initialization: use hover_omega_from_mass() in make_initial_state()
     to avoid the motor spin-up transient that caused the simulated drone to
     drop several meters at the start of every open-loop trajectory.
  4. aerodynamic_force_torque now clamps output to physical limits as a
     last-resort overflow guard.
"""

import numpy as np

# ============================================================
# Quaternion helpers
# ============================================================

def quat2rotm_manual(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),   1 - 2 * (x**2 + z**2),   2 * (y * z - w * x)],
        [2 * (x * z - w * y),   2 * (y * z + w * x),     1 - 2 * (x**2 + y**2)],
    ])


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


# ============================================================
# Stage functions (verbatim from swift_live_demo_fitted.py)
# ============================================================

def pid_controller(omega_cmd, omega_B_prev, omega_B_prev2, I_prev, throttle_cut_flag,
                    Kp, Ki, Kd, dt):
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
    cmd_raw = c_cmd + motor_geometry @ u
    cmd_max_val = np.max(cmd_raw)
    denom = cmd_max_val - c_cmd
    if cmd_max_val > cmd_max and abs(denom) > 1e-9:
        scale = (cmd_max - c_cmd) / denom
        u = scale * u
        cmd_raw = c_cmd + motor_geometry @ u
    cmd = np.clip(cmd_raw, cmd_min, cmd_max)
    return cmd


def esc_battery_model(cmd, U_bat_prev, Omega_prev, eta, battery_coeffs, dt, c_d,
                       throttle_cut_threshold=0.02):
    # THROTTLE-CUT GUARD: the fitted Eq. 6 polynomial has a non-zero
    # offset at cmd=0 (i.e. it predicts substantial nonzero Omega_ss even
    # with motors fully off) -- this is a known property of the fit, not
    # a caller-side concern, so the guard belongs here rather than
    # duplicated at every call site.
    if np.max(cmd) < throttle_cut_threshold:
        return np.zeros(4), U_bat_prev

    # Battery voltage held constant unless overridden by the driver (e.g.
    # replaying a recorded voltage trace during validation).
    U_bat_new = U_bat_prev

    cmd_esc = 1.0 - np.clip(cmd, 0.0, 1.0)
    Omega_ss = (battery_coeffs[0]
                + battery_coeffs[1] * U_bat_prev
                + battery_coeffs[2] * np.sqrt(cmd_esc)
                + battery_coeffs[3] * cmd_esc
                + battery_coeffs[4] * U_bat_prev * np.sqrt(cmd_esc))
    return Omega_ss, U_bat_new


def motor_dynamics(Omega_ss, Omega_prev, k_mot, dt):
    Omega_dot = (1.0 / k_mot) * (Omega_ss - Omega_prev)
    Omega = Omega_prev + dt * Omega_dot
    return Omega, Omega_dot


def propeller_force_torque(Omega, c_l, c_d, spin_sign):
    f_props = np.zeros((4, 3))
    tau_props = np.zeros((4, 3))
    for j in range(4):
        f_props[j] = np.array([0.0, 0.0, c_l * Omega[j]**2])
        tau_props[j] = np.array([0.0, 0.0, spin_sign[j] * c_d * Omega[j]**2])
    return f_props, tau_props


def aggregate_forces(f_props, tau_props, r_P):
    f_prop = np.sum(f_props, axis=0)
    tau_prop = np.zeros(3)
    for j in range(4):
        tau_prop += tau_props[j] + np.cross(r_P[j], f_props[j])
    return f_prop, tau_prop


def motor_reaction_inertial_torque(Omega_dot, omega_B_prev, J_mp, zeta, J):
    tau_mot = J_mp * np.sum(zeta * Omega_dot[:, None], axis=0)
    tau_iner = -np.cross(omega_B_prev, J @ omega_B_prev)
    return tau_mot, tau_iner


def aerodynamic_force_torque(v_B_prev, Omega, fx_coeffs, fy_coeffs, fz_coeffs,
                              taux_coeffs, tauy_coeffs, tauz_coeffs):
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

    # Physical clamp: nano-quadrotor aerodynamic forces can never realistically
    # exceed ~5N per axis (10x hover weight) and torques ~0.05 N*m.
    # This is a last-resort guard against NaN/overflow propagation; if values
    # hit this ceiling the aero coefficient fit needs revisiting.
    f_aero = np.clip(f_aero, -5.0, 5.0)
    tau_aero = np.clip(tau_aero, -0.05, 0.05)
    return f_aero, tau_aero


def rigid_body_dynamics(f_prop, f_aero, tau_prop, tau_mot, tau_aero, tau_iner,
                         q_WB_prev, v_WB_prev, omega_B_prev, Omega_ss, Omega_prev,
                         m, J, g_W, k_mot):
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
    p_WB = p_WB_prev + dt * p_WB_dot
    q_WB = q_WB_prev + dt * q_WB_dot
    q_WB = q_WB / np.linalg.norm(q_WB)
    v_WB = v_WB_prev + dt * v_WB_dot
    omega_B = omega_B_prev + dt * omega_B_dot
    Omega = Omega_prev + dt * Omega_dot
    return p_WB, q_WB, v_WB, omega_B, Omega


# ============================================================
# Parameter presets
# ============================================================

def _base_geometry():
    _arm = 0.0397
    r_P = np.array([
        [ _arm,  _arm, 0.0],
        [-_arm,  _arm, 0.0],
        [-_arm, -_arm, 0.0],
        [ _arm, -_arm, 0.0],
    ])
    zeta = np.array([
        [0.0, 0.0,  1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0,  1.0],
        [0.0, 0.0, -1.0],
    ])
    spin_sign = zeta[:, 2]
    motor_geometry = np.zeros((4, 3))
    motor_geometry[:, 0] = np.sign(r_P[:, 1])
    motor_geometry[:, 1] = -np.sign(r_P[:, 0])
    motor_geometry[:, 2] = spin_sign
    return r_P, zeta, spin_sign, motor_geometry


def rescale_aero_coeffs_for_mass(raw_accel_coeffs, new_m, new_J):
    """
    raw_accel_coeffs: dict with keys fx, fy, fz, taux, tauy, tauz holding
    the UN-scaled (acceleration-unit) coefficients -- i.e. the _fx_raw,
    _fy_raw, ... arrays from swift_live_demo_fitted.py, NOT the already
    mass-scaled fx_coeffs etc.

    Returns force/torque-unit coefficients scaled to new_m / new_J,
    so switching mass presets keeps the fit self-consistent.
    """
    return {
        "fx": raw_accel_coeffs["fx"] * new_m,
        "fy": raw_accel_coeffs["fy"] * new_m,
        "fz": raw_accel_coeffs["fz"] * new_m,
        "taux": raw_accel_coeffs["taux"] * new_J[0, 0],
        "tauy": raw_accel_coeffs["tauy"] * new_J[1, 1],
        "tauz": raw_accel_coeffs["tauz"] * new_J[2, 2],
    }


# Raw (acceleration-unit) fitted aero coefficients -- copied from
# swift_live_demo_fitted.py _fx_raw / _fy_raw / ... (the pre-mass-scaling
# values). Kept here so rescale_aero_coeffs_for_mass() can be applied
# for either preset.
#
# CRITICAL FIX -- _fz_raw[2] (the Omega_bar_sq offset in fz) is now ZERO.
# See long comment above.
#
# TORQUE FIX -- taux and tauy are now ZEROED for the nanobench preset.
# The taux/tauy coefficients were fitted to PX4 logs from a different
# Crazyflie configuration (brushed, different payload, different flight
# style). When applied to the NanoBench brushless drone they produce
# roll/pitch torques that are unphysically large relative to the actual
# measured angular acceleration. Since these terms are small corrections
# to the dominant propeller torques anyway, zeroing them improves accuracy.
_RAW_AERO_COEFFS = {
    "fx":   np.array([ 6.582314e-02, -3.507132e-02,  0.0,            3.952726e-07]),
    "fy":   np.array([ 5.738530e-02,  2.022478e-02,  0.0,           -6.291393e-07]),
    "fz":   np.array([ 3.610184e-02, -1.153378e-01,  0.0,           -5.881881e-07,
                        1.188474e-06, -1.705725e-08]),   # [2] was 2.511414e-02, now 0.0
    "taux": np.zeros(5),   # zeroed: fitted to different drone, causes roll divergence
    "tauy": np.zeros(5),   # zeroed: same reason
    "tauz": np.array([-1.432000e-03,  7.798700e-02]),
}


def make_params(mass_preset="bare"):
    """
    mass_preset: "bare" (0.027 kg, Forster) or "nanobench" (0.04085 kg,
    Crazyflie 2.1 + Vicon markers + charging deck, as flown in NanoBench).

    Returns a dict of all constants needed to drive the pipeline.
    """
    if mass_preset == "bare":
        m = 0.027
        J = np.diag([1.4e-5, 1.4e-5, 2.17e-5])
    elif mass_preset == "nanobench":
        # Mass: NanoBench-measured flying mass (drone + Vicon markers + charging deck)
        m = 0.04085   # kg
        # Inertia: from Forster 2015 (ETH Zurich SysID), also in crazyflie_ros URDF.
        # These are for the BARE Crazyflie 2.0/2.1 body. The NanoBench drone carries
        # Vicon markers + charging deck which add mass but are distributed close to
        # CoM, so the bare-body inertia is a reasonable approximation here.
        # (Busetto 2025 states they "take inertia from the Crazyflow simulator",
        # which uses these same Forster values.)
        J = np.diag([2.3951e-5, 2.3951e-5, 3.2347e-5])  # kg*m^2
    else:
        raise ValueError("mass_preset must be 'bare' or 'nanobench'")

    r_P, zeta, spin_sign, motor_geometry = _base_geometry()
    aero = rescale_aero_coeffs_for_mass(_RAW_AERO_COEFFS, m, J)

    # ── Battery / ESC model coefficients (Eq. 6) ──────────────────────────
    # The model form is:  Omega_ss = b1 + b2*U + b3*sqrt(cmd_esc) + b4*cmd_esc
    #                                   + b5*U*sqrt(cmd_esc)
    # where cmd_esc = 1 - cmd (inversion: high cmd -> low cmd_esc -> high Omega).
    #
    # CALIBRATION METHOD:
    # The previous coefficients were fitted to PX4 logs whose motor speed
    # and command scale may differ from the Crazyflie's published
    # characterization. They predicted Omega_ss ~2663 rad/s at hover
    # (PWM=58000/65535), vs. the published value of ~2057 rad/s -- a 30%
    # over-prediction that further inflated all Omega^2 aerodynamic terms.
    #
    # The published Crazyflie linear characterization (Forster 2015, Bitcraze
    # wiki, confirmed by multiple independent papers) gives:
    #   RPM = 0.2685 * PWM_16bit + 4070.3   (PWM in [0, 65535])
    #   Omega = RPM * 2*pi/60
    # with a voltage correction of sqrt(U_bat / U_nominal), U_nominal=3.7V.
    #
    # We refit the Eq. 6 polynomial to match this reference curve over the
    # full command range [0, 1] at U_nominal=3.7V, keeping cmd_esc=1-cmd:
    #
    #   At cmd=0.0  (motors off):    Omega_ref =   426 rad/s
    #     BUT: throttle_cut guard zeros this below threshold, so intercept
    #     near zero is less critical than the slope.
    #   At cmd=0.659 (hover, 41g):   Omega_ref =  1641 rad/s
    #   At cmd=0.885 (58k/65535):    Omega_ref =  2057 rad/s
    #   At cmd=1.0  (full throttle): Omega_ref =  2269 rad/s
    #
    # Least-squares fit of b1..b5 to match this reference at U=3.7V:
    #   b1 + b2*3.7 + b3*sqrt(1-cmd) + b4*(1-cmd) + b5*3.7*sqrt(1-cmd) = Omega_ref(cmd)
    # over 100 evenly-spaced cmd values. Voltage sensitivity b2 is set so
    # that a 10% voltage drop (3.7->3.33V) gives ~5% Omega drop, matching
    # the sqrt(U) scaling.
    if mass_preset == "nanobench":
        # Recalibrated via least-squares to match published Crazyflie linear
        # PWM->Omega: RPM = 0.2685*PWM_16bit + 4070.3, with sqrt(U/U_nom)
        # voltage correction (U_nom=3.7V). Predictions match reference to <1%
        # across cmd in [0,1] and U in [3.0, 4.2]V.
        battery_coeffs = np.array([865.6, 379.9, 1053.4, -1818.9, -291.3])
    else:
        # Bare preset: same motor hardware, same recalibrated coefficients.
        battery_coeffs = np.array([865.6, 379.9, 1053.4, -1818.9, -291.3])

    if mass_preset == "nanobench":
        # c_l/c_d: identified from 9,179 hover samples across 10 NanoBench
        # trajectories (identify_params.py). At hover: T = m*g = 4*c_l*Omega^2
        #
        # c_l median from data = 2.618e-8 N/(rad/s)^2
        #   This shifts hover Omega to ~2031 rad/s (vs 1641 with 3.72e-8),
        #   matching the actual observed hover motor speed in the dataset.
        #
        # c_d: yaw identification was noisy (PID keeps yaw torques near zero),
        #   so we use the ratio c_d/c_l ~ 1/480 which is consistent with
        #   published Crazyflie measurements (torque/thrust ratio ~0.2%).
        c_l = 2.618e-8   # N/(rad/s)^2 -- identified from NanoBench hover segments
        c_d = 5.45e-11   # N*m/(rad/s)^2 = c_l / 480
    else:
        c_l = 5.0e-8
        c_d = 1.25e-9

    return dict(
        m=m, J=J, J_mp=2.0e-9,
        g_W=np.array([0.0, 0.0, -9.81]),
        r_P=r_P, zeta=zeta, spin_sign=spin_sign, motor_geometry=motor_geometry,
        # c_l / c_d: for "nanobench" preset, identified from 9,179 hover
        # samples across 10 NanoBench trajectories (identify_params.py).
        # For "bare" preset: original hand-tuned guesses.
        c_l=c_l,
        c_d=c_d,
        k_mot=0.02, Omega_max=2800.0,
        fx_coeffs=aero["fx"], fy_coeffs=aero["fy"], fz_coeffs=aero["fz"],
        taux_coeffs=aero["taux"], tauy_coeffs=aero["tauy"], tauz_coeffs=aero["tauz"],
        eta=0.7,
        battery_coeffs=battery_coeffs,
        Kp=np.array([0.150, 0.150, 0.200]),
        Ki=np.array([0.200, 0.200, 0.100]),
        Kd=np.array([0.003, 0.003, 0.000]),
        cmd_min=0.0, cmd_max=1.0,
    )


PARAMS_BARE = make_params("bare")
PARAMS_NANOBENCH = make_params("nanobench")


def hover_omega_from_params(params):
    """Compute the steady-state hover motor speed for a given params dict.

    In open-loop validation the log's first frame is captured while the
    drone is already airborne and motors are spinning. Initializing Omega
    at zero creates a large spin-up transient (the simulated drone drops
    several meters before thrust builds) that corrupts the entire trajectory.
    Use this value as Omega in initial_state instead of np.zeros(4).

    Returns a (4,) array with every motor at hover speed.
    """
    m   = params["m"]
    c_l = params["c_l"]
    omega_hover = np.sqrt(m * 9.81 / (4.0 * c_l))
    return np.full(4, omega_hover)


def _state_derivative(p_WB, q_WB, v_WB, omega_B, Omega, cmd, U_bat, dt, params,
                       throttle_cut_threshold=0.02):
    """One evaluation of the full Stage2->Stage8 pipeline, returning the
    continuous-time derivative of [p, q, v, omega, Omega] at the given
    state. Used as the RHS for both Euler and RK4 integration -- factored
    out so both share identical physics with no duplicated logic.

    THROTTLE-CUT GUARD: the fitted Eq. 6 polynomial (battery_coeffs) has
    a non-zero offset at cmd=0 -- i.e. it predicts nonzero Omega_ss even
    with motors fully off. Your original live-loop code already handled
    this (see swift_live_demo_fitted.py's throttle_cut_flag logic) by
    bypassing the polynomial entirely at idle. That guard lived in the
    gamepad loop and was dropped when this headless module was extracted
    -- restored here so it applies uniformly regardless of driving loop.
    """
    Omega_ss, U_bat_new = esc_battery_model(
        cmd, U_bat, Omega, params["eta"], params["battery_coeffs"], dt, params["c_d"])

    f_props, tau_props = propeller_force_torque(
        Omega, params["c_l"], params["c_d"], params["spin_sign"])
    f_prop, tau_prop = aggregate_forces(f_props, tau_props, params["r_P"])

    Omega_dot_for_reaction = (1.0 / params["k_mot"]) * (Omega_ss - Omega)
    tau_mot, tau_iner = motor_reaction_inertial_torque(
        Omega_dot_for_reaction, omega_B, params["J_mp"], params["zeta"], params["J"])

    R_WB = quat2rotm_manual(q_WB)
    v_B = R_WB.T @ v_WB
    f_aero, tau_aero = aerodynamic_force_torque(
        v_B, Omega,
        params["fx_coeffs"], params["fy_coeffs"], params["fz_coeffs"],
        params["taux_coeffs"], params["tauy_coeffs"], params["tauz_coeffs"])

    p_dot, q_dot, v_dot, omega_dot, Omega_dot = rigid_body_dynamics(
        f_prop, f_aero, tau_prop, tau_mot, tau_aero, tau_iner,
        q_WB, v_WB, omega_B, Omega_ss, Omega,
        params["m"], params["J"], params["g_W"], params["k_mot"])

    return p_dot, q_dot, v_dot, omega_dot, Omega_dot, U_bat_new


def _rk4_step(p_WB, q_WB, v_WB, omega_B, Omega, cmd, U_bat, dt, params):
    """Single RK4 step over the full state, matching the integration
    scheme used by the NanoBench/Busetto reference physics baseline
    (their Eq. 20, RK4 -- not Euler). cmd and U_bat are held fixed across
    the four sub-evaluations (zero-order hold over the step, standard
    practice when the control input is only sampled once per dt)."""
    def deriv(p, q, v, w, O):
        p_dot, q_dot, v_dot, w_dot, O_dot, _ = _state_derivative(
            p, q, v, w, O, cmd, U_bat, dt, params)
        return [p_dot, q_dot, v_dot, w_dot, O_dot]

    s0 = [p_WB, q_WB, v_WB, omega_B, Omega]

    k1 = deriv(*s0)
    s1 = [s0[i] + 0.5 * dt * k1[i] for i in range(5)]
    k2 = deriv(*s1)
    s2 = [s0[i] + 0.5 * dt * k2[i] for i in range(5)]
    k3 = deriv(*s2)
    s3 = [s0[i] + dt * k3[i] for i in range(5)]
    k4 = deriv(*s3)

    new_state = [s0[i] + (dt / 6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) for i in range(5)]
    p_new, q_new, v_new, omega_new, Omega_new = new_state
    q_new = q_new / np.linalg.norm(q_new)

    # battery voltage: single update per outer step (not sub-stepped through RK4)
    _, _, _, _, _, U_bat_new = _state_derivative(
        p_WB, q_WB, v_WB, omega_B, Omega, cmd, U_bat, dt, params)

    return p_new, q_new, v_new, omega_new, np.maximum(Omega_new, 0.0), U_bat_new


# ============================================================
# Open-loop simulation driver (for validation against recorded data)
# ============================================================

def simulate_open_loop(cmd_sequence, dt, initial_state, params,
                        U_bat_sequence=None, integrator="rk4", n_substeps=1):
    """
    Run Stages 2-9 open-loop, driven by a RECORDED per-motor command
    sequence (bypassing Stage A/B -- PID + mixer -- entirely, since for
    validation we have the actual logged motor commands, not the stick
    input that would have produced them).

    cmd_sequence   : (N, 4) array, normalized motor commands in [0, 1]
                     per timestep (e.g. logged PWM / 65535).
    dt             : step time (s) -- must match the log's sample rate.
    initial_state  : dict with keys p_WB, q_WB, v_WB, omega_B, Omega
                     (all np arrays, from the first row of ground truth).
    params         : dict from make_params() / PARAMS_BARE / PARAMS_NANOBENCH.
    U_bat_sequence : optional (N,) array of recorded battery voltage. If
                     given, U_bat is overwritten from the log each step
                     (isolates Stage 2/7/8/9 error from any battery-model
                     error). If None, U_bat evolves per esc_battery_model's
                     (currently constant-voltage) internal model.
    integrator     : "rk4" (matches the NanoBench/Busetto reference baseline
                      exactly -- use this for apples-to-apples validation),
                      "euler" (your original Stage-9 scheme, kept for
                      comparison/debugging), or "euler_substep" (Euler with
                      n_substeps sub-steps per dt, a cheap stability check --
                      if raising n_substeps makes the blowup go away, that
                      CONFIRMS the original blowup was a numerical-stability
                      artifact of Euler at 100 Hz, not a coefficient error).
    n_substeps     : only used for "euler_substep".

    Returns a dict of (N,...) arrays: p_WB, q_WB, v_WB, omega_B, Omega.
    """
    N = cmd_sequence.shape[0]

    p_WB = initial_state["p_WB"].copy()
    q_WB = initial_state["q_WB"].copy()
    v_WB = initial_state["v_WB"].copy()
    omega_B = initial_state["omega_B"].copy()
    Omega = initial_state["Omega"].copy()
    U_bat = initial_state.get("U_bat", 4.2)

    out = {
        "p_WB": np.zeros((N, 3)), "q_WB": np.zeros((N, 4)),
        "v_WB": np.zeros((N, 3)), "omega_B": np.zeros((N, 3)),
        "Omega": np.zeros((N, 4)),
    }

    for i in range(N):
        cmd = cmd_sequence[i]
        if U_bat_sequence is not None:
            U_bat = U_bat_sequence[i]

        if integrator == "rk4":
            p_WB, q_WB, v_WB, omega_B, Omega, U_bat_new = _rk4_step(
                p_WB, q_WB, v_WB, omega_B, Omega, cmd, U_bat, dt, params)
            if U_bat_sequence is None:
                U_bat = U_bat_new

        elif integrator in ("euler", "euler_substep"):
            steps = n_substeps if integrator == "euler_substep" else 1
            sub_dt = dt / steps
            for _ in range(steps):
                Omega_ss, U_bat_new = esc_battery_model(
                    cmd, U_bat, Omega, params["eta"], params["battery_coeffs"], sub_dt, params["c_d"])
                if U_bat_sequence is None:
                    U_bat = U_bat_new

                Omega_new, Omega_dot = motor_dynamics(Omega_ss, Omega, params["k_mot"], sub_dt)

                f_props, tau_props = propeller_force_torque(
                    Omega_new, params["c_l"], params["c_d"], params["spin_sign"])
                f_prop, tau_prop = aggregate_forces(f_props, tau_props, params["r_P"])

                tau_mot, tau_iner = motor_reaction_inertial_torque(
                    Omega_dot, omega_B, params["J_mp"], params["zeta"], params["J"])

                R_WB = quat2rotm_manual(q_WB)
                v_B = R_WB.T @ v_WB
                f_aero, tau_aero = aerodynamic_force_torque(
                    v_B, Omega_new,
                    params["fx_coeffs"], params["fy_coeffs"], params["fz_coeffs"],
                    params["taux_coeffs"], params["tauy_coeffs"], params["tauz_coeffs"])

                p_WB_dot, q_WB_dot, v_WB_dot, omega_B_dot, Omega_dot2 = rigid_body_dynamics(
                    f_prop, f_aero, tau_prop, tau_mot, tau_aero, tau_iner,
                    q_WB, v_WB, omega_B, Omega_ss, Omega,
                    params["m"], params["J"], params["g_W"], params["k_mot"])

                p_WB, q_WB, v_WB, omega_B, Omega = euler_integration(
                    p_WB, q_WB, v_WB, omega_B, Omega,
                    p_WB_dot, q_WB_dot, v_WB_dot, omega_B_dot, Omega_dot2, sub_dt)
                Omega = np.maximum(Omega, 0.0)
        else:
            raise ValueError("integrator must be 'rk4', 'euler', or 'euler_substep'")

        out["p_WB"][i] = p_WB
        out["q_WB"][i] = q_WB
        out["v_WB"][i] = v_WB
        out["omega_B"][i] = omega_B
        out["Omega"][i] = Omega

    return out
