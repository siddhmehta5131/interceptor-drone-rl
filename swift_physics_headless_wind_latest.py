"""
swift_physics_headless_wind.py  v3.0

Headless, importable scalar physics core — no pygame, no matplotlib.
Identical physics to rlvec v3.0 but in scalar (non-vectorized) form
for readability and comparison testing.

v3.0 changelog (from v2.0):
──────────────────────────────────────────────────────────────────────
  BUG FIXES
    1. ground_effect_factor: ratio clamped to 0.99 → no /0
    2. blade flapping: uses per-rotor LOCAL velocity (ω×r_P)
    3. J_inv precomputed once per step (was recomputed 4x inside deriv)

  MISSING PHYSICS ADDED
    4. Body aerodynamic forces/torques (polynomial model)
    5. Motor reaction torque (τ_mot = J_mp · Σ(ζ·Ω̇))
    6. Motor dynamics (first-order lag) via rk4_full_step()
    7. Omega_max enforcement after integration

  API
    8. rk4_rigid_body_step(): backward-compat, now includes aero drag
    9. rk4_full_step(): NEW — full pipeline with motor commands
   10. Both accept optional wind_W parameter
──────────────────────────────────────────────────────────────────────
"""

import numpy as np

__version__ = "wind-only-headless-3.0"

# ── Frame geometry ─────────────────────────────────────────────────────────────
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
spin_sign = zeta[:, 2]   # +1 CCW, -1 CW per motor

# ── Aero coefficients (nominal, acceleration units) ────────────────────────────
_fx_raw   = np.array([ 6.582314e-02, -3.507132e-02,  0.0,            3.952726e-07])
_fy_raw   = np.array([ 5.738530e-02,  2.022478e-02,  0.0,           -6.291393e-07])
_fz_raw   = np.array([ 3.610184e-02, -1.153378e-01,  0.0,           -5.881881e-07,
                        1.188474e-06, -1.705725e-08])
_tauz_raw = np.array([-1.432000e-03,  7.798700e-02])

# ── Parameters ─────────────────────────────────────────────────────────────────
PARAMS_NANOBENCH = dict(
    # Rigid body
    m          = 0.04085,
    J          = np.diag([2.3951e-5, 2.3951e-5, 3.2347e-5]),
    J_mp       = 2.0e-9,
    g_W        = np.array([0.0, 0.0, -9.81]),
    # Frame
    r_P        = r_P,
    zeta       = zeta,
    spin_sign  = spin_sign,
    # Propeller coefficients (NanoBench sysid)
    c_l        = 2.618e-8,
    c_d        = 5.45e-11,
    k_mot      = 0.02,
    Omega_max  = 2800.0,
    # v2.0 — rotor aero
    C_FLAP     = 5.0e-9,
    C_HUB      = 2.5e-10,
    R_ROTOR    = 0.023,
    # v3.0 — body aero (acceleration units)
    fx_raw     = _fx_raw.copy(),
    fy_raw     = _fy_raw.copy(),
    fz_raw     = _fz_raw.copy(),
    tauz_raw   = _tauz_raw.copy(),
    # ESC / battery
    battery_coeffs = np.array([865.6, 379.9, 1053.4, -1818.9, -291.3]),
    eta        = 0.7,
)


# ── Quaternion helpers ─────────────────────────────────────────────────────────

def quat2rotm_manual(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2+z**2),  2*(x*y - w*z),      2*(x*z + w*y)    ],
        [2*(x*y + w*z),      1 - 2*(x**2+z**2),   2*(y*z - w*x)    ],
        [2*(x*z - w*y),      2*(y*z + w*x),       1 - 2*(x**2+y**2)],
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


# ── Propeller helpers ──────────────────────────────────────────────────────────

def hover_omega_from_params(params):
    return np.full(4, float(np.sqrt(params["m"] * 9.81 / (4.0 * params["c_l"]))))


def propeller_force_torque(Omega, c_l, c_d, spin_sign_):
    f_props   = np.zeros((4, 3))
    tau_props = np.zeros((4, 3))
    for j in range(4):
        f_props[j]   = [0., 0., c_l * Omega[j]**2]
        tau_props[j] = [0., 0., spin_sign_[j] * c_d * Omega[j]**2]
    return f_props, tau_props


def aggregate_forces(f_props, tau_props, r_P_):
    f_body   = np.sum(f_props, axis=0)
    tau_body = np.zeros(3)
    for j in range(4):
        tau_body += tau_props[j] + np.cross(r_P_[j], f_props[j])
    return f_body, tau_body


# ── v3.0 physics (scalar, edge-case safe) ──────────────────────────────────────

def ground_effect_factor(height, R_rotor):
    """Cheeseman-Bennett ground effect.  SAFE: ratio clamped < 1.0."""
    if R_rotor <= 0.0 or height < 0.001 or height >= 4.0 * R_rotor:
        return 1.0
    ratio = min(R_rotor / (4.0 * max(height, 1e-6)), 0.99)      # ← FIX
    return min(1.0 / (1.0 - ratio**2), 2.0)


def hub_drag_force(Omega, v_B, C_HUB):
    """Body-frame hub drag:  F = −C_HUB · Σ(Ω²) · [vx, vy, 0]."""
    if C_HUB <= 0.0:
        return np.zeros(3)
    Osq = float(np.sum(Omega**2))
    return np.array([-C_HUB * Osq * v_B[0],
                     -C_HUB * Osq * v_B[1],
                     0.0])


def blade_flapping_torque(Omega, v_B, omega_B, spin_sign_, r_P_, C_FLAP):
    """Per-rotor blade flapping with LOCAL velocity at each hub.

    FIX vs v2.0: uses v_local_j = v_B + ω × r_j per rotor, so flapping
    activates with angular velocity even at symmetric hover.
    """
    if C_FLAP <= 0.0:
        return np.zeros(3)
    tau = np.zeros(3)
    for j in range(4):
        v_local = v_B + np.cross(omega_B, r_P_[j])
        tau += C_FLAP * Omega[j] * spin_sign_[j] * np.array(
            [v_local[1], -v_local[0], 0.0])
    return tau


def aerodynamic_force_torque(v_B, Omega, m, params):
    """Body aerodynamic forces/torques (polynomial model, scalar).

    NEW in v3.0 — was completely absent in wind v1.0-v2.0.
    Uses nominal acceleration-unit coefficients scaled by mass.
    """
    fx_raw  = params.get('fx_raw',  np.zeros(4))
    fy_raw  = params.get('fy_raw',  np.zeros(4))
    fz_raw  = params.get('fz_raw',  np.zeros(6))
    tauz_raw = params.get('tauz_raw', np.zeros(2))

    if np.all(fx_raw == 0) and np.all(fy_raw == 0):
        return np.zeros(3), np.zeros(3)

    vx, vy, vz = float(v_B[0]), float(v_B[1]), float(v_B[2])
    v_xy = np.sqrt(vx**2 + vy**2)
    Ob2  = float(np.mean(Omega**2))

    fx = (fx_raw * m) @ [vx, vx*abs(vx), Ob2, vx*Ob2]
    fy = (fy_raw * m) @ [vy, vy*abs(vy), Ob2, vy*Ob2]
    fz = (fz_raw * m) @ [vz, vz*abs(vz), v_xy**2, v_xy*Ob2,
                          vz*Ob2, v_xy*vz*Ob2]

    J_diag = np.diag(params['J']) if params['J'].ndim == 2 else params['J']
    tz = (tauz_raw * float(J_diag[2])) @ [vx, vy]

    f_aero   = np.clip([fx, fy, fz],     -50.0, 50.0)
    tau_aero = np.clip([0.0, 0.0, tz],    -5.0,  5.0)
    return np.asarray(f_aero), np.asarray(tau_aero)


def quadratic_body_drag(v_B):
    """Physics-based quadratic body drag (scalar): F = -½ρCdA × v × |v|.

    Always dissipative.  Dominates at high speed where the fitted polynomial
    becomes unreliable.
    """
    C_XY = 0.5 * 1.225 * 1.0 * 0.003     # 0.00184 N/(m/s)²
    C_Z  = 0.5 * 1.225 * 1.3 * 0.008     # 0.00637 N/(m/s)²
    return np.array([
        -C_XY * v_B[0] * abs(v_B[0]),
        -C_XY * v_B[1] * abs(v_B[1]),
        -C_Z  * v_B[2] * abs(v_B[2]),
    ])


def ensure_dissipative(f, v, eps=1e-12):
    """Remove energy-injecting component from an aero force (scalar).

    If F·v > 0, project out the component along v.
    """
    dot  = float(np.dot(f, v))
    v_sq = float(np.dot(v, v)) + eps
    if dot > 0:
        return f - (dot / v_sq) * v
    return f


def _esc_model(cmd, U_bat, params):
    """Scalar ESC polynomial → steady-state motor speeds."""
    bc = params.get('battery_coeffs',
                     np.array([865.6, 379.9, 1053.4, -1818.9, -291.3]))
    c  = np.clip(cmd, 0.0, 1.0)
    Oss = (bc[0] + bc[1]*U_bat + bc[2]*np.sqrt(c) + bc[3]*c
           + bc[4]*U_bat*np.sqrt(c))
    Oss = np.where(c < 0.02, 0.0, Oss)
    return Oss


# ── Core derivative (fixed-force mode) ────────────────────────────────────────

def _rigid_body_deriv_fixed(p, q, v, w, f_prop, tau_prop, J_inv, m, g_W,
                             Omega, params, wind_W):
    """Derivative with fixed propeller wrench + velocity-dependent extras."""
    R   = quat2rotm_manual(q)
    v_air = v - wind_W
    v_B = R.T @ v_air

    # ── Extra forces / torques ────────────────────────────────────────────
    f_extra   = np.zeros(3)
    tau_extra = np.zeros(3)

    # Hub drag
    C_HUB = params.get('C_HUB', 0.0)
    if C_HUB > 0.0:
        f_extra += hub_drag_force(Omega, v_B, C_HUB)

    # Blade flapping (per-rotor local velocity)
    C_FLAP = params.get('C_FLAP', 0.0)
    if C_FLAP > 0.0:
        sp = params.get('spin_sign', spin_sign)
        rp = params.get('r_P', r_P)
        tau_extra += blade_flapping_torque(Omega, v_B, w, sp, rp, C_FLAP)

    # Body aero drag (polynomial, made safe)
    f_a, t_a = aerodynamic_force_torque(v_B, Omega, m, params)
    f_a = ensure_dissipative(f_a, v_B)     # remove energy injection
    f_extra += f_a
    tau_extra += t_a

    # Quadratic body drag (always active, always dissipative)
    f_extra += quadratic_body_drag(v_B)

    # Safety clamp
    f_extra = np.clip(f_extra, -0.5, 0.5)

    # Gyroscopic: τ_iner = −ω × Jω
    J_diag = np.diag(params['J']) if params['J'].ndim == 2 else params['J']
    Jw = J_diag * w
    tau_extra -= np.cross(w, Jw)

    # Ground effect
    R_rotor = params.get('R_ROTOR', 0.0)
    k_ge = ground_effect_factor(float(p[2]), R_rotor)
    f_prop_ge = f_prop.copy()
    f_prop_ge[2] *= k_ge

    # ── Dynamics ──────────────────────────────────────────────────────────
    p_dot     = v
    omega_quat = np.array([0.0, w[0], w[1], w[2]])
    q_dot     = 0.5 * quat_mult(q, omega_quat)
    v_dot     = (1.0 / m) * (R @ (f_prop_ge + f_extra)) + g_W
    omega_dot = J_inv @ (tau_prop + tau_extra)

    return p_dot, q_dot, v_dot, omega_dot


# ── Core derivative (full pipeline mode) ──────────────────────────────────────

def _full_pipeline_deriv(p, q, v, w, Omega, cmd, U_bat, J_inv, m, g_W,
                          params, wind_W, dt):
    """Full Stages 2-8 derivative including motor dynamics."""
    # Stage 2: ESC
    Omega_ss = _esc_model(cmd, U_bat, params)

    # Stage 3: Motor dynamics
    k_m = params.get('k_mot', 0.02)
    Omega_max_p = params.get('Omega_max', 2800.0)
    Omega_dot = (Omega_ss - Omega) / k_m
    Omega_eff = np.clip(Omega + 0.5 * dt * Omega_dot, 0.0, Omega_max_p)

    # Stage 4-5: Propeller forces
    c_l = params['c_l']
    c_d = params['c_d']
    sp  = params.get('spin_sign', spin_sign)
    rp  = params.get('r_P', r_P)
    f_props, tau_props = propeller_force_torque(Omega_eff, c_l, c_d, sp)
    f_prop, tau_prop = aggregate_forces(f_props, tau_props, rp)

    # Stage 6: Motor reaction torque
    zt = params.get('zeta', zeta)
    J_mp_p = params.get('J_mp', 2.0e-9)
    tau_mot = J_mp_p * np.sum(zt * Omega_dot[:, None], axis=0)

    # Rotation + body-frame velocity
    R   = quat2rotm_manual(q)
    v_air = v - wind_W
    v_B = R.T @ v_air

    # ── Extra forces / torques ────────────────────────────────────────────
    f_extra   = np.zeros(3)
    tau_extra = np.zeros(3)

    C_HUB = params.get('C_HUB', 0.0)
    if C_HUB > 0.0:
        f_extra += hub_drag_force(Omega_eff, v_B, C_HUB)

    C_FLAP = params.get('C_FLAP', 0.0)
    if C_FLAP > 0.0:
        tau_extra += blade_flapping_torque(Omega_eff, v_B, w, sp, rp, C_FLAP)

    f_a, t_a = aerodynamic_force_torque(v_B, Omega_eff, m, params)
    f_a = ensure_dissipative(f_a, v_B)     # remove energy injection
    f_extra += f_a
    tau_extra += t_a

    # Quadratic body drag (always active, always dissipative)
    f_extra += quadratic_body_drag(v_B)

    # Safety clamp
    f_extra = np.clip(f_extra, -0.5, 0.5)

    J_diag = np.diag(params['J']) if params['J'].ndim == 2 else params['J']
    Jw = J_diag * w
    tau_extra -= np.cross(w, Jw)

    R_rotor = params.get('R_ROTOR', 0.0)
    k_ge = ground_effect_factor(float(p[2]), R_rotor)
    f_prop_ge = f_prop.copy()
    f_prop_ge[2] *= k_ge

    # ── Dynamics ──────────────────────────────────────────────────────────
    p_dot     = v
    omega_quat = np.array([0.0, w[0], w[1], w[2]])
    q_dot     = 0.5 * quat_mult(q, omega_quat)
    v_dot     = (1.0 / m) * (R @ (f_prop_ge + f_extra)) + g_W
    omega_dot = J_inv @ (tau_prop + tau_mot + tau_extra)

    return p_dot, q_dot, v_dot, omega_dot, Omega_dot


# ── RK4: backward-compatible step ─────────────────────────────────────────────

def rk4_rigid_body_step(state0, f_body, tau_body, params, dt, wind_W=None):
    """RK4 step with FIXED propeller wrench + velocity-dependent extras.

    Backward-compatible with v2.0 callers.  Omega is static.

    v3.0: adds body aero drag, per-rotor flapping, safe GE, precomputed J_inv.
    """
    m     = params["m"]
    J     = params["J"]
    g_W_p = params["g_W"]
    J_inv = np.linalg.inv(J)                                   # precomputed once

    p, q, v, w = (state0["p_WB"], state0["q_WB"],
                  state0["v_WB"], state0["omega_B"])
    Omega = state0.get("Omega")
    if Omega is None:
        Omega = hover_omega_from_params(params)

    w_W = np.zeros(3) if wind_W is None else np.asarray(wind_W)

    def D(p_, q_, v_, w_):
        return _rigid_body_deriv_fixed(
            p_, q_, v_, w_, f_body, tau_body, J_inv, m, g_W_p,
            Omega, params, w_W)

    s = [p, q, v, w]
    k1 = list(D(*s))
    k2 = list(D(*[s[i] + 0.5*dt*k1[i] for i in range(4)]))
    k3 = list(D(*[s[i] + 0.5*dt*k2[i] for i in range(4)]))
    k4 = list(D(*[s[i] +     dt*k3[i] for i in range(4)]))

    out = [s[i] + (dt/6.0)*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
           for i in range(4)]
    out[1] = out[1] / np.linalg.norm(out[1])

    return dict(p_WB=out[0], q_WB=out[1], v_WB=out[2], omega_B=out[3],
                Omega=Omega, U_bat=state0.get("U_bat"))


# ── RK4: full pipeline step ───────────────────────────────────────────────────

def rk4_full_step(state0, cmd, params, dt, wind_W=None):
    """Full-pipeline RK4 step with motor dynamics.

    Includes ALL physics: ESC, motor lag, propeller, motor reaction,
    body aero, ground effect, hub drag, blade flapping, gyroscopic.

    Omega is evolved as a state variable.
    """
    m     = params["m"]
    J     = params["J"]
    g_W_p = params["g_W"]
    J_inv = np.linalg.inv(J)

    p, q, v, w = (state0["p_WB"], state0["q_WB"],
                  state0["v_WB"], state0["omega_B"])
    Omega = state0["Omega"]
    U_bat = state0.get("U_bat", 4.2)

    w_W = np.zeros(3) if wind_W is None else np.asarray(wind_W)

    def D(p_, q_, v_, w_, O_):
        return _full_pipeline_deriv(
            p_, q_, v_, w_, O_, cmd, U_bat, J_inv, m, g_W_p,
            params, w_W, dt)

    s = [p, q, v, w, Omega]
    k1 = list(D(*s))
    k2 = list(D(*[s[i] + 0.5*dt*k1[i] for i in range(5)]))
    k3 = list(D(*[s[i] + 0.5*dt*k2[i] for i in range(5)]))
    k4 = list(D(*[s[i] +     dt*k3[i] for i in range(5)]))

    out = [s[i] + (dt/6.0)*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
           for i in range(5)]
    out[1] = out[1] / np.linalg.norm(out[1])
    Omega_max_p = params.get('Omega_max', 2800.0)
    out[4] = np.clip(out[4], 0.0, Omega_max_p)

    return dict(p_WB=out[0], q_WB=out[1], v_WB=out[2], omega_B=out[3],
                Omega=out[4], U_bat=U_bat)
