"""
swift_rl_env.py

Vectorized, RL-training-ready version of the drone physics pipeline from
swift_live_demo.py. Structural changes made for RL, in order of impact:

  1. No I/O in the loop. Gamepad polling and plotting are gone entirely --
     this file is a pure function step(state, action) -> next_state, obs,
     reward, done. Use swift_live_demo.py (or a thin wrapper around this
     module) if you want to watch a human fly it.
  2. Vectorized across N parallel environments. Every array carries a
     leading (N, ...) dimension and there are no Python-level loops over
     drones or motors -- numpy broadcasting handles all of it. This is
     usually the single biggest speedup available, bigger than any of the
     physics additions below.
  3. J_inv is precomputed once (J never changes) instead of being
     re-inverted every RK4 sub-step.

Physics additions, ranked by RL value per unit of added compute (see
report). All are "do this" / "worth it" / "do it if sim-to-real matters"
items; the ones flagged "skip" in the report (full BEM, motor
electromagnetics, CFD, soft-body collision) are intentionally absent --
they cost orders of magnitude more per step and mostly correct effects
that domain randomization already makes the policy robust to.

  - Wind (OU process), applied as air-relative velocity into aerodynamics.
  - Omega_max enforcement (was an unenforced physical limit -- real bug).
  - Domain randomization: m, J, c_l, aero coeffs, and wind parameters are
    resampled per-episode per-env. This is flagged in the report as
    arguably higher RL value than any single physics term.
  - Ground effect: multiplicative thrust boost near the floor.
  - Sensor/observation noise: added only to what the policy *observes*,
    never to the true dynamics state. Toggle off if you're training
    sim-only and don't care about sim-to-real transfer.
  - Battery sag: linear voltage drain vs. cumulative energy draw, replacing
    the constant-voltage placeholder.
"""

import numpy as np


# ==================== FIXED (non-randomized) CONSTANTS ====================

_arm = 0.0397  # m -- Crazyflie arm length
r_P = np.array([
    [ _arm,  _arm, 0.0],
    [-_arm,  _arm, 0.0],
    [-_arm, -_arm, 0.0],
    [ _arm, -_arm, 0.0],
])  # (4,3)
zeta = np.array([
    [0.0, 0.0,  1.0],
    [0.0, 0.0, -1.0],
    [0.0, 0.0,  1.0],
    [0.0, 0.0, -1.0],
])  # (4,3)
spin_sign = zeta[:, 2]  # (4,)

motor_geometry = np.zeros((4, 3))
motor_geometry[:, 0] = np.sign(r_P[:, 1])
motor_geometry[:, 1] = -np.sign(r_P[:, 0])
motor_geometry[:, 2] = spin_sign

k_mot = 0.02          # s
Omega_max = 2800.0    # rad/s -- now actually enforced (see step())
J_mp = 2.0e-9         # kg*m^2 -- motor+prop rotor inertia
g_W = np.array([0.0, 0.0, -9.81])
eta = 0.7
battery_coeffs = np.array([865.6, 379.9, 1053.4, -1818.9, -291.3])
Kp = np.array([0.150, 0.150, 0.200])
Ki = np.array([0.200, 0.200, 0.100])
Kd = np.array([0.003, 0.003, 0.000])
cmd_min, cmd_max = 0.0, 1.0
dt = 0.01
GROUND_EFFECT_HEIGHT = 0.15   # m -- rotor-diameter-ish scale for the correction to kick in
GROUND_EFFECT_MAX_GAIN = 0.20  # max fractional thrust boost right at the floor
ENERGY_FULL_CHARGE_J = 1800.0  # J -- nominal usable energy before voltage sags to cutoff
U_BAT_FULL, U_BAT_CUTOFF = 4.2, 3.3

# Nominal (pre-randomization) values -- domain randomization perturbs around these
_m_nom = 0.04085
_J_nom = np.array([2.3951e-5, 2.3951e-5, 3.2347e-5])
_c_l_nom = 2.618e-8
_c_d_nom = 5.45e-11
_fx_raw_nom   = np.array([ 6.582314e-02, -3.507132e-02,  0.0,            3.952726e-07])
_fy_raw_nom   = np.array([ 5.738530e-02,  2.022478e-02,  0.0,           -6.291393e-07])
_fz_raw_nom   = np.array([ 3.610184e-02, -1.153378e-01,  0.0,          -5.881881e-07,
                            1.188474e-06, -1.705725e-08])
_tauz_raw_nom = np.array([-1.432000e-03,  7.798700e-02])


# ==================== ENV STATE ====================

class DroneEnvState:
    """All per-env arrays carry a leading (N,) or (N, k) dimension.
    Everything here is either a loop variable (evolves every step) or a
    per-episode randomized parameter (fixed within an episode, resampled
    on reset)."""

    def __init__(self, N, rng):
        self.N = N
        self.rng = rng

        # -- loop variables --
        self.p_WB = np.zeros((N, 3))
        self.q_WB = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (N, 1))
        self.v_WB = np.zeros((N, 3))
        self.omega_B = np.zeros((N, 3))
        self.omega_B_prev = np.zeros((N, 3))
        self.Omega = np.zeros((N, 4))
        self.I_pid = np.zeros((N, 3))
        self.U_bat = np.full(N, U_BAT_FULL)
        self.energy_used_J = np.zeros(N)
        self.wind_W = np.zeros((N, 3))

        self.reset(np.arange(N))

    def reset(self, idx):
        """Resample domain-randomized parameters and loop variables for
        the given env indices (idx). Call with np.arange(N) on full reset,
        or with just the done envs for auto-reset."""
        n = len(idx)
        rng = self.rng

        self.p_WB[idx] = 0.0
        self.q_WB[idx] = np.array([1.0, 0.0, 0.0, 0.0])
        self.v_WB[idx] = 0.0
        self.omega_B[idx] = 0.0
        self.omega_B_prev[idx] = 0.0
        self.I_pid[idx] = 0.0
        self.U_bat[idx] = U_BAT_FULL
        self.energy_used_J[idx] = 0.0
        self.wind_W[idx] = 0.0

        # -- domain randomization: physical params, +/-10-20% around nominal --
        if not hasattr(self, "m"):
            self.m = np.full(self.N, _m_nom)
            self.J = np.tile(_J_nom, (self.N, 1))
            self.c_l = np.full(self.N, _c_l_nom)
            self.c_d = np.full(self.N, _c_d_nom)
            self.fx_raw = np.tile(_fx_raw_nom, (self.N, 1))
            self.fy_raw = np.tile(_fy_raw_nom, (self.N, 1))
            self.fz_raw = np.tile(_fz_raw_nom, (self.N, 1))
            self.tauz_raw = np.tile(_tauz_raw_nom, (self.N, 1))
            self.wind_mean_W = np.zeros((self.N, 3))
            self.wind_theta = np.full(self.N, 0.5)
            self.wind_sigma = np.full(self.N, 1.0)

        self.m[idx] = _m_nom * rng.uniform(0.9, 1.1, size=n)
        self.J[idx] = _J_nom * rng.uniform(0.9, 1.1, size=(n, 3))
        self.c_l[idx] = _c_l_nom * rng.uniform(0.85, 1.15, size=n)
        self.c_d[idx] = _c_d_nom * rng.uniform(0.85, 1.15, size=n)
        self.fx_raw[idx] = _fx_raw_nom * rng.uniform(0.8, 1.2, size=(n, 4))
        self.fy_raw[idx] = _fy_raw_nom * rng.uniform(0.8, 1.2, size=(n, 4))
        self.fz_raw[idx] = _fz_raw_nom * rng.uniform(0.8, 1.2, size=(n, 6))
        self.tauz_raw[idx] = _tauz_raw_nom * rng.uniform(0.8, 1.2, size=(n, 2))

        # wind randomized per episode -- turns one wind model into a whole
        # family of disturbance conditions for ~free
        self.wind_mean_W[idx] = rng.normal(0.0, 1.5, size=(n, 3))
        self.wind_theta[idx] = rng.uniform(0.2, 1.0, size=n)
        self.wind_sigma[idx] = rng.uniform(0.3, 2.0, size=n)

        # hover-speed motor init (per-env, since m and c_l are now randomized)
        omega_hover = np.sqrt(self.m[idx] * 9.81 / (4.0 * self.c_l[idx]))
        self.Omega[idx] = omega_hover[:, None]


# ==================== VECTORIZED PHYSICS ====================
# Every function below takes (N, ...) arrays and has no Python loop over
# N, envs, or motors.

def quat2rotm_batch(q):
    """q: (N,4) [w,x,y,z] -> R: (N,3,3), body->world."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3))
    R[:, 0, 0] = 1 - 2 * (y**2 + z**2); R[:, 0, 1] = 2 * (x*y - w*z);     R[:, 0, 2] = 2 * (x*z + w*y)
    R[:, 1, 0] = 2 * (x*y + w*z);       R[:, 1, 1] = 1 - 2 * (x**2 + z**2); R[:, 1, 2] = 2 * (y*z - w*x)
    R[:, 2, 0] = 2 * (x*z - w*y);       R[:, 2, 1] = 2 * (y*z + w*x);     R[:, 2, 2] = 1 - 2 * (x**2 + y**2)
    return R


def quat_mult_batch(q1, q2):
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=1)


def pid_controller_batch(omega_cmd, omega_B_prev, omega_B_prev2, I_prev,
                          throttle_cut_flag, dt):
    e = omega_cmd - omega_B_prev
    P = Kp * e
    I_new = np.where(throttle_cut_flag[:, None], 0.0, I_prev + Ki * e * dt)
    D = -Kd * (omega_B_prev - omega_B_prev2) / dt
    return P + I_new + D, I_new


def mixer_batch(u, c_cmd):
    """u: (N,3), c_cmd: (N,) -> cmd: (N,4)."""
    cmd_raw = c_cmd[:, None] + u @ motor_geometry.T          # (N,4)
    cmd_max_val = np.max(cmd_raw, axis=1)                    # (N,)
    denom = cmd_max_val - c_cmd
    needs_scale = (cmd_max_val > cmd_max) & (np.abs(denom) > 1e-9)
    scale = np.where(needs_scale, (cmd_max - c_cmd) / np.where(denom == 0, 1.0, denom), 1.0)
    u_scaled = u * scale[:, None]
    cmd_raw = c_cmd[:, None] + u_scaled @ motor_geometry.T
    return np.clip(cmd_raw, cmd_min, cmd_max)


def esc_battery_batch(cmd, U_bat):
    """Battery sag: voltage now a function of cumulative energy drawn,
    not constant. Returns Omega_ss (N,4)."""
    cmd_esc = np.clip(cmd, 0.0, 1.0)
    throttle_cut = cmd_esc < 0.02
    Omega_ss = (battery_coeffs[0]
                + battery_coeffs[1] * U_bat[:, None]
                + battery_coeffs[2] * np.sqrt(cmd_esc)
                + battery_coeffs[3] * cmd_esc
                + battery_coeffs[4] * U_bat[:, None] * np.sqrt(cmd_esc))
    return np.where(throttle_cut, 0.0, Omega_ss)


def battery_sag_update(U_bat, energy_used_J, cmd, Omega, eta, dt):
    """Stage 2 addition: linear voltage drain vs. cumulative energy drawn,
    replacing the constant-voltage placeholder. Cheap: one extra multiply
    and a linear interpolation."""
    # Instantaneous electrical power draw, crude proxy: ESC command * motor
    # speed, scaled by inefficiency -- good enough for a training-time
    # disturbance signal, not a certified power model.
    power_W = np.sum(cmd * Omega, axis=1) / eta * 1e-4  # scaling constant tuned so a hover ~ a few W
    energy_used_J = energy_used_J + power_W * dt
    frac_used = np.clip(energy_used_J / ENERGY_FULL_CHARGE_J, 0.0, 1.0)
    U_bat_new = U_BAT_FULL + frac_used * (U_BAT_CUTOFF - U_BAT_FULL)
    return U_bat_new, energy_used_J


def propeller_force_torque_batch(Omega, c_l, c_d):
    """Omega: (N,4) -> f_props: (N,4,3), tau_props: (N,4,3)."""
    N = Omega.shape[0]
    f_props = np.zeros((N, 4, 3))
    tau_props = np.zeros((N, 4, 3))
    f_props[:, :, 2] = c_l[:, None] * Omega**2
    tau_props[:, :, 2] = spin_sign[None, :] * c_d[:, None] * Omega**2
    return f_props, tau_props


def aggregate_forces_batch(f_props, tau_props):
    """f_props, tau_props: (N,4,3) -> f_prop: (N,3), tau_prop: (N,3)."""
    f_prop = np.sum(f_props, axis=1)
    tau_prop = np.sum(tau_props + np.cross(r_P[None, :, :], f_props), axis=1)
    return f_prop, tau_prop


def motor_reaction_inertial_torque_batch(Omega_dot, omega_B_prev, J):
    """Omega_dot: (N,4), omega_B_prev: (N,3), J: (N,3) diag -> tau_mot, tau_iner: (N,3)."""
    tau_mot = J_mp * np.sum(zeta[None, :, :] * Omega_dot[:, :, None], axis=1)
    Jw = J * omega_B_prev  # diag(J) @ w, broadcast
    tau_iner = -np.cross(omega_B_prev, Jw)
    return tau_mot, tau_iner


def ground_effect_gain(height_m):
    """Multiplicative thrust correction near the floor: 1.0 far from the
    ground, up to (1 + GROUND_EFFECT_MAX_GAIN) right at it. One cheap
    correction, you already track z."""
    h = np.clip(height_m, 0.0, GROUND_EFFECT_HEIGHT)
    return 1.0 + GROUND_EFFECT_MAX_GAIN * (1.0 - h / GROUND_EFFECT_HEIGHT)


def aerodynamic_force_torque_batch(v_rel_B, Omega, m, J, fx_raw, fy_raw, fz_raw, tauz_raw):
    vx, vy, vz = v_rel_B[:, 0], v_rel_B[:, 1], v_rel_B[:, 2]
    v_xy = np.sqrt(vx**2 + vy**2)
    Omega_bar_sq = np.mean(Omega**2, axis=1)

    fx_coeffs = fx_raw * m[:, None]
    fy_coeffs = fy_raw * m[:, None]
    fz_coeffs = fz_raw * m[:, None]
    tauz_coeffs = tauz_raw * J[:, 2:3]

    f_x = np.sum(fx_coeffs * np.stack([vx, vx*np.abs(vx), Omega_bar_sq, vx*Omega_bar_sq], axis=1), axis=1)
    f_y = np.sum(fy_coeffs * np.stack([vy, vy*np.abs(vy), Omega_bar_sq, vy*Omega_bar_sq], axis=1), axis=1)
    f_z = np.sum(fz_coeffs * np.stack([
        vz, vz*np.abs(vz), v_xy**2, v_xy*Omega_bar_sq, vz*Omega_bar_sq, v_xy*vz*Omega_bar_sq
    ], axis=1), axis=1)
    tau_z = np.sum(tauz_coeffs * np.stack([vx, vy], axis=1), axis=1)

    f_aero = np.clip(np.stack([f_x, f_y, f_z], axis=1), -50.0, 50.0)
    tau_aero = np.clip(np.stack([np.zeros_like(tau_z), np.zeros_like(tau_z), tau_z], axis=1), -5.0, 5.0)
    return f_aero, tau_aero


def update_wind_batch(wind_W, dt, rng, wind_mean_W, wind_theta, wind_sigma):
    """OU step, vectorized over N envs. Call once per outer dt."""
    noise = rng.standard_normal(wind_W.shape) * wind_sigma[:, None] * np.sqrt(dt)
    return wind_W + wind_theta[:, None] * (wind_mean_W - wind_W) * dt + noise


def rigid_body_dynamics_batch(f_prop, f_aero, tau_prop, tau_mot, tau_aero, tau_iner,
                               q_WB, v_WB, omega_B, Omega_ss, Omega, m, J_inv, height_m):
    p_dot = v_WB
    omega_quat = np.concatenate([np.zeros((omega_B.shape[0], 1)), omega_B], axis=1)
    q_dot = 0.5 * quat_mult_batch(q_WB, omega_quat)

    R = quat2rotm_batch(q_WB)
    ge_gain = ground_effect_gain(height_m)  # (N,)
    f_prop_ge = f_prop * np.stack([np.ones_like(ge_gain), np.ones_like(ge_gain), ge_gain], axis=1)
    v_dot = np.einsum('nij,nj->ni', R, f_prop_ge + f_aero) / m[:, None] + g_W

    tau_total = tau_prop + tau_mot + tau_aero + tau_iner  # (N,3)
    omega_dot = np.einsum('nij,nj->ni', J_inv, tau_total)

    Omega_dot = (1.0 / k_mot) * (Omega_ss - Omega)
    return p_dot, q_dot, v_dot, omega_dot, Omega_dot


def _pipeline_deriv(state, p, q, v, w, Om, cmd, U_bat, wind_W, J_inv):
    Omega_ss = esc_battery_batch(cmd, U_bat)
    Omega_dot_mot = (1.0 / k_mot) * (Omega_ss - Om)
    Omega_eff = np.maximum(Om + 0.5 * dt * Omega_dot_mot, 0.0)
    Omega_eff = np.minimum(Omega_eff, Omega_max)  # enforce Omega_max -- was unenforced

    f_props, tau_props = propeller_force_torque_batch(Omega_eff, state.c_l, state.c_d)
    f_prop, tau_prop = aggregate_forces_batch(f_props, tau_props)
    tau_mot, tau_iner = motor_reaction_inertial_torque_batch(Omega_dot_mot, w, state.J)

    R = quat2rotm_batch(q)
    v_rel_B = np.einsum('nij,nj->ni', np.transpose(R, (0, 2, 1)), v - wind_W)
    f_aero, tau_aero = aerodynamic_force_torque_batch(
        v_rel_B, Omega_eff, state.m, state.J, state.fx_raw, state.fy_raw, state.fz_raw, state.tauz_raw)

    height_m = p[:, 2]
    return rigid_body_dynamics_batch(
        f_prop, f_aero, tau_prop, tau_mot, tau_aero, tau_iner,
        q, v, w, Omega_ss, Om, state.m, J_inv, height_m)


def rk4_step_batch(state, cmd, U_bat, wind_W):
    """cmd, U_bat, wind_W held fixed (zero-order hold) over dt.
    J_inv is precomputed once per call from state.J (diagonal -> trivial
    inverse), not re-inverted every sub-step."""
    J_inv = np.zeros((state.N, 3, 3))
    idx = np.arange(3)
    J_inv[:, idx, idx] = 1.0 / state.J

    p, q, v, w, Om = state.p_WB, state.q_WB, state.v_WB, state.omega_B, state.Omega

    def deriv(p_, q_, v_, w_, O_):
        return _pipeline_deriv(state, p_, q_, v_, w_, O_, cmd, U_bat, wind_W, J_inv)

    s0 = [p, q, v, w, Om]
    k1 = list(deriv(*s0))
    s1 = [s0[i] + 0.5 * dt * k1[i] for i in range(5)]
    k2 = list(deriv(*s1))
    s2 = [s0[i] + 0.5 * dt * k2[i] for i in range(5)]
    k3 = list(deriv(*s2))
    s3 = [s0[i] + dt * k3[i] for i in range(5)]
    k4 = list(deriv(*s3))
    new_state = [s0[i] + (dt / 6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) for i in range(5)]
    p_n, q_n, v_n, w_n, Om_n = new_state
    q_n = q_n / np.linalg.norm(q_n, axis=1, keepdims=True)
    Om_n = np.clip(Om_n, 0.0, Omega_max)  # enforce Omega_max after integration too
    return p_n, q_n, v_n, w_n, Om_n


# ==================== ENV STEP ====================

def step(state: DroneEnvState, action, obs_noise=True):
    """Pure function: state, action -> (obs, reward, done, info).
    Mutates `state` in place (its loop variables) and auto-resets any env
    that hit `done` this call.

    action : (N,4) array in [-1,1], columns [thrust, roll_cmd, pitch_cmd, yaw_cmd]
             (same convention as the human TRUE INPUT, just batched).
    """
    N = state.N
    rng = state.rng

    thrust = action[:, 0]
    omega_cmd = action[:, 1:4] * 2.0  # RATE_CMD_SCALE
    throttle_cut = thrust < 0.02

    c_cmd = np.clip(thrust, 0.0, None)
    u_torque, state.I_pid = pid_controller_batch(
        omega_cmd, state.omega_B, state.omega_B_prev, state.I_pid, throttle_cut, dt)
    cmd = mixer_batch(u_torque, c_cmd)
    cmd = np.where(throttle_cut[:, None], 0.0, cmd)

    state.wind_W = update_wind_batch(
        state.wind_W, dt, rng, state.wind_mean_W, state.wind_theta, state.wind_sigma)

    p_n, q_n, v_n, w_n, Om_n = rk4_step_batch(state, cmd, state.U_bat, state.wind_W)

    if not np.all(throttle_cut):
        state.U_bat, state.energy_used_J = battery_sag_update(
            state.U_bat, state.energy_used_J, cmd, state.Omega, eta, dt)

    state.omega_B_prev = state.omega_B.copy()     # FIX: .copy() prevents aliasing (D-term was dead)
    state.p_WB, state.q_WB, state.v_WB, state.omega_B, state.Omega = p_n, q_n, v_n, w_n, Om_n

    # ground clamp
    below = state.p_WB[:, 2] <= 0.0
    state.p_WB[below, 2] = 0.0
    falling = below & (state.v_WB[:, 2] < 0.0)
    state.v_WB[falling, 2] = 0.0

    out_of_bounds = (
        (state.p_WB[:, 0] < -20.0) | (state.p_WB[:, 0] > 20.0) |
        (state.p_WB[:, 1] < -20.0) | (state.p_WB[:, 1] > 20.0) |
        (state.p_WB[:, 2] > 40.0)
    )
    done = out_of_bounds

    # placeholder reward: penalize distance from a 5m hover + control effort.
    # Replace with your task's actual reward.
    dist = np.linalg.norm(state.p_WB - np.array([0.0, 0.0, 5.0]), axis=1)
    reward = -dist - 0.01 * np.sum(u_torque**2, axis=1)
    reward = np.where(done, reward - 10.0, reward)

    obs = np.concatenate([state.p_WB, state.v_WB, state.q_WB, state.omega_B, state.Omega / Omega_max], axis=1)
    if obs_noise:
        # sensor noise added only to what the policy observes, never to
        # the true dynamics state above
        obs = obs + rng.normal(0.0, 0.01, size=obs.shape)

    if np.any(done):
        state.reset(np.where(done)[0])

    return obs, reward, done, {}


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    N = 4096
    rng = np.random.default_rng(0)
    state = DroneEnvState(N, rng)

    action = np.zeros((N, 4))
    action[:, 0] = 0.6  # roughly hover-ish thrust command

    import time
    t0 = time.time()
    n_steps = 200
    for _ in range(n_steps):
        obs, reward, done, info = step(state, action)
    dt_wall = time.time() - t0
    print(f"{N} envs x {n_steps} steps in {dt_wall:.3f}s "
          f"({N * n_steps / dt_wall:,.0f} env-steps/sec)")
