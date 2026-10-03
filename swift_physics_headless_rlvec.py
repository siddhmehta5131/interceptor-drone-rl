"""
swift_physics_headless_rlvec.py  v3.0

Complete vectorized (N=1 batch) physics engine for comparison testing
and RL training integration.

v3.0 changelog (from v2.0):
──────────────────────────────────────────────────────────────────────
  BUG FIXES
    1. ground_effect_factor_batch: ratio clamped to 0.99 → no /0
    2. blade flapping: uses per-rotor LOCAL velocity (ω×r_P) → activates
       even at symmetric hover when drone has angular velocity
    3. omega_quat: uses w.shape[0] instead of hardcoded 1
    4. Omega_max enforcement: clamped after RK4 integration

  MISSING PHYSICS ADDED
    5. Body aerodynamic forces/torques (polynomial model from swift_rl_env)
    6. Motor reaction torque (τ_mot = J_mp · Σ(ζ·Ω̇))
    7. Motor dynamics (first-order lag, Ω as integrated state variable)
       → rk4_full_step() uses ESC model + motor lag
    8. Gyroscopic precession (was present, now shared cleanly)

  PERFORMANCE
    9. J_inv precomputed once per step (was inside derivative → 4x waste)
   10. Aero coefficients tiled once per step (not per sub-step)

  API
   11. rk4_rigid_body_step(): backward-compatible, now includes aero drag
   12. rk4_full_step(): NEW — full pipeline with motor commands
   13. Both accept optional wind_W parameter
──────────────────────────────────────────────────────────────────────
"""

import numpy as np
import swift_rl_env as rlenv

__version__ = "rlvec-headless-3.0"

# ─── Constants (from rlenv) ───────────────────────────────────────────────────
r_P       = rlenv.r_P            # (4,3)
zeta      = rlenv.zeta           # (4,3)
spin_sign = rlenv.spin_sign      # (4,)
J_mp      = rlenv.J_mp           # scalar
k_mot     = rlenv.k_mot          # scalar
Omega_max = rlenv.Omega_max      # scalar
g_W       = rlenv.g_W            # (3,)

# ─── Parameters (complete, all physics) ───────────────────────────────────────
PARAMS_NANOBENCH = dict(
    # Rigid body
    m          = rlenv._m_nom,
    J          = np.diag(rlenv._J_nom),          # (3,3) inertia matrix
    J_mp       = J_mp,
    g_W        = g_W,
    # Frame
    r_P        = r_P,
    zeta       = zeta,
    spin_sign  = spin_sign,
    # Propeller (NanoBench sysid)
    c_l        = rlenv._c_l_nom,
    c_d        = rlenv._c_d_nom,
    k_mot      = k_mot,
    Omega_max  = Omega_max,
    # v2.0 — rotor aero
    C_FLAP     = 5.0e-9,         # N·m·s/rad   blade flapping
    C_HUB      = 2.5e-10,        # N·s/(m·rad²) hub drag
    R_ROTOR    = 0.023,           # m            Crazyflie blade radius
    # v3.0 — body aero (acceleration-unit polynomial coefficients)
    fx_raw     = rlenv._fx_raw_nom.copy(),       # (4,)
    fy_raw     = rlenv._fy_raw_nom.copy(),       # (4,)
    fz_raw     = rlenv._fz_raw_nom.copy(),       # (6,)
    tauz_raw   = rlenv._tauz_raw_nom.copy(),     # (2,)
    # ESC / battery
    battery_coeffs = rlenv.battery_coeffs.copy(),
    eta        = rlenv.eta,
)


# ─── Propeller helpers (thin wrappers for N=1) ───────────────────────────────

def hover_omega_from_params(params):
    return np.full(4, float(np.sqrt(params["m"] * 9.81 / (4.0 * params["c_l"]))))


def propeller_force_torque(Omega, c_l, c_d, spin_sign_):
    f, t = rlenv.propeller_force_torque_batch(
        Omega[None, :], np.array([c_l]), np.array([c_d]))
    return f[0], t[0]


def aggregate_forces(f_props, tau_props, r_P_):
    f, t = rlenv.aggregate_forces_batch(
        f_props[None, :, :], tau_props[None, :, :])
    return f[0], t[0]


# ─── v3.0 physics (vectorized, edge-case safe) ───────────────────────────────

def ground_effect_factor_batch(p, R_rotor):
    """Cheeseman-Bennett 1957 ground effect. SAFE: ratio clamped < 1.0.

    k_ge = 1 / (1 − (R/4z)²),   capped at 2.0.
    Active when 1 mm ≤ z < 4·R_rotor.  Outside that range returns 1.0.
    """
    if R_rotor <= 0.0:
        return np.ones(p.shape[0])
    height = p[:, 2]
    safe_h = np.maximum(height, 1e-6)
    ratio  = np.minimum(R_rotor / (4.0 * safe_h), 0.99)     # ← FIX: prevents /0
    k      = 1.0 / (1.0 - ratio**2)
    k      = np.minimum(k, 2.0)
    active = (height >= 0.001) & (height < 4.0 * R_rotor)
    return np.where(active, k, 1.0)


def ground_effect_factor(height, R_rotor):
    """Scalar version (backward compat for test notebooks)."""
    if R_rotor <= 0.0 or height < 0.001 or height >= 4.0 * R_rotor:
        return 1.0
    ratio = min(R_rotor / (4.0 * max(height, 1e-6)), 0.99)
    return min(1.0 / (1.0 - ratio**2), 2.0)


def hub_drag_force_batch(Omega, v_B, C_HUB):
    """Body-frame hub drag opposing horizontal translation.
    F_hub = −C_HUB · Σ(Ω²) · [vx, vy, 0]
    """
    if C_HUB <= 0.0:
        return np.zeros_like(v_B)
    Osq = np.sum(Omega**2, axis=1, keepdims=True)    # (N,1)
    f = np.zeros_like(v_B)
    f[:, 0] = -C_HUB * Osq[:, 0] * v_B[:, 0]
    f[:, 1] = -C_HUB * Osq[:, 0] * v_B[:, 1]
    return f


def blade_flapping_torque_batch(Omega, v_B, omega_B, C_FLAP):
    """Per-rotor blade flapping with LOCAL velocity at each hub.

    v_local_j = v_B + ω_B × r_P_j     (body frame, per-rotor)
    τ_j = C_FLAP · Ω_j · σ_j · [v_local_y, −v_local_x, 0]

    FIX vs v2.0:  v2.0 used global v_B → net torque was zero at symmetric
    hover.  v3.0 uses per-rotor v_local → nonzero when drone has angular
    velocity (different rotors see different local airspeed).
    """
    if C_FLAP <= 0.0:
        return np.zeros_like(v_B)
    N = Omega.shape[0]
    # Local velocity at each rotor: v_B + ω × r_j
    w_cross_r = np.cross(omega_B[:, None, :], r_P[None, :, :])  # (N,4,3)
    v_local = v_B[:, None, :] + w_cross_r                        # (N,4,3)

    # Per-rotor flapping coefficient: C_FLAP · Ω_j · σ_j
    coeff = C_FLAP * Omega * spin_sign[None, :]                   # (N,4)

    return np.stack([
        np.sum(coeff *  v_local[:, :, 1], axis=1),    # τ_x
        np.sum(coeff * -v_local[:, :, 0], axis=1),    # τ_y
        np.zeros(N),                                    # τ_z = 0
    ], axis=1)                                          # (N,3)


def quadratic_body_drag_batch(v_B):
    """Physics-based quadratic body drag: F = -½ρCdA × v × |v|.

    ALWAYS dissipative (opposes motion).  Dominates at high speed where the
    fitted polynomial becomes unreliable.

    Crazyflie approximate parameters:
      ρ = 1.225 kg/m³, Cd_xy ≈ 1.0, Cd_z ≈ 1.3
      A_xy ≈ 0.003 m² (30mm height × 100mm span)
      A_z  ≈ 0.008 m² (80mm × 100mm top area)
    """
    C_XY = 0.5 * 1.225 * 1.0 * 0.003     # 0.00184 N/(m/s)²
    C_Z  = 0.5 * 1.225 * 1.3 * 0.008     # 0.00637 N/(m/s)²
    f = np.zeros_like(v_B)
    f[:, 0] = -C_XY * v_B[:, 0] * np.abs(v_B[:, 0])
    f[:, 1] = -C_XY * v_B[:, 1] * np.abs(v_B[:, 1])
    f[:, 2] = -C_Z  * v_B[:, 2] * np.abs(v_B[:, 2])
    return f


def ensure_dissipative_batch(f, v, eps=1e-12):
    """Remove any energy-injecting component from an aerodynamic force.

    If F·v > 0 (force has a component accelerating the body along its
    velocity), project that component out.  Cross-axis forces (e.g. side
    force from forward flight) are preserved.

    This prevents the fitted polynomial's v×Ω² cross-terms from creating
    positive feedback at high speeds while keeping near-hover behavior
    completely unchanged (both F and v are ~0 at hover).
    """
    dot   = np.sum(f * v, axis=1, keepdims=True)      # F·v  (N,1)
    v_sq  = np.sum(v**2, axis=1, keepdims=True) + eps  # |v|²  (N,1)
    # Only remove the positive (energy-injecting) projection
    inject = np.maximum(dot, 0.0) / v_sq               # scalar projection
    return f - inject * v                               # safe force


# ─── Precomputed cache (one allocation per rk4 call) ─────────────────────────

def _make_cache(params, N):
    """Precompute batched arrays.  Called ONCE per rk4_*_step, not per sub-step."""
    J_mat = params['J']
    J_diag = np.diag(J_mat) if J_mat.ndim == 2 else J_mat

    J_inv = np.zeros((N, 3, 3))
    ix = np.arange(3)
    J_inv[:, ix, ix] = 1.0 / J_diag

    m_arr = np.full(N, params['m'])
    J_d   = np.tile(J_diag, (N, 1))                              # (N,3)
    c_l   = np.full(N, params['c_l'])
    c_d   = np.full(N, params['c_d'])

    # Aero coefficients tiled for rlenv.aerodynamic_force_torque_batch
    fx  = np.tile(params.get('fx_raw',  np.zeros(4)), (N, 1))
    fy  = np.tile(params.get('fy_raw',  np.zeros(4)), (N, 1))
    fz  = np.tile(params.get('fz_raw',  np.zeros(6)), (N, 1))
    tz  = np.tile(params.get('tauz_raw', np.zeros(2)), (N, 1))
    has_aero = not (np.all(fx == 0) and np.all(fy == 0) and
                    np.all(fz == 0) and np.all(tz == 0))

    return dict(
        m_arr=m_arr, J_diag=J_d, J_inv=J_inv,
        c_l=c_l, c_d=c_d,
        fx_raw=fx, fy_raw=fy, fz_raw=fz, tauz_raw=tz,
        has_aero=has_aero,
        R_ROTOR=params.get('R_ROTOR', 0.0),
        C_HUB=params.get('C_HUB', 0.0),
        C_FLAP=params.get('C_FLAP', 0.0),
    )


# ─── Shared extras (aero drag, hub drag, flapping, gyroscopic) ───────────────

def _compute_extras(p, q, v, w, Omega, cache, wind_W):
    """All velocity-dependent body-frame forces/torques beyond base propeller.

    Returns
    -------
    f_extra   : (N,3)  hub drag + body aero force
    tau_extra : (N,3)  blade flapping + body aero torque − gyroscopic
    k_ge      : (N,)   ground effect thrust multiplier
    R         : (N,3,3) rotation matrix (reused by caller for v_dot)
    """
    N = p.shape[0]
    R = rlenv.quat2rotm_batch(q)

    # Body-frame air-relative velocity
    v_air = v - wind_W
    v_B = np.einsum('nij,nj->ni', R.transpose(0, 2, 1), v_air)

    f_extra   = np.zeros((N, 3))
    tau_extra = np.zeros((N, 3))

    # Hub drag
    f_extra += hub_drag_force_batch(Omega, v_B, cache['C_HUB'])

    # Blade flapping (per-rotor local velocity)
    tau_extra += blade_flapping_torque_batch(Omega, v_B, w, cache['C_FLAP'])

    # Body aerodynamic drag (polynomial model, made safe)
    if cache['has_aero']:
        f_a, t_a = rlenv.aerodynamic_force_torque_batch(
            v_B, Omega, cache['m_arr'], cache['J_diag'],
            cache['fx_raw'], cache['fy_raw'],
            cache['fz_raw'], cache['tauz_raw'])
        # FIX: the polynomial has v×Ω² terms that inject energy at high speed.
        # Remove any component that accelerates along the velocity vector.
        f_a = ensure_dissipative_batch(f_a, v_B)
        f_extra += f_a
        tau_extra += t_a

    # Quadratic body drag (always active, always dissipative, correct at high speed)
    f_extra += quadratic_body_drag_batch(v_B)

    # Safety clamp: aero forces can never exceed ~12x body weight per axis
    F_MAX = 0.5    # N  (body weight ≈ 0.04085 * 9.81 ≈ 0.40 N)
    f_extra = np.clip(f_extra, -F_MAX, F_MAX)

    # Gyroscopic precession:  τ_iner = −ω × Jω
    Jw = cache['J_diag'] * w                    # (N,3) = diag(J) ⊙ ω
    tau_extra -= np.cross(w, Jw)                 # (N,3)

    # Ground effect
    k_ge = ground_effect_factor_batch(p, cache['R_ROTOR'])

    return f_extra, tau_extra, k_ge, R


# ─── Derivative: fixed-force mode (backward compat) ──────────────────────────

def _deriv_fixed(p, q, v, w, f_prop, tau_prop, cache, Omega, wind_W):
    """Rigid-body derivative with propeller wrench held FIXED.
    Aero drag, hub drag, flapping, gyroscopic are velocity-dependent.
    Omega is static (not evolved).  Integrates 4 state variables.
    """
    N = p.shape[0]
    f_extra, tau_extra, k_ge, R = _compute_extras(
        p, q, v, w, Omega, cache, wind_W)

    # Boost propeller z-thrust by ground effect
    f_prop_ge = f_prop.copy()
    f_prop_ge[:, 2] *= k_ge

    # p, q, v, ω dynamics
    p_dot = v
    omega_quat = np.concatenate([np.zeros((N, 1)), w], axis=1)
    q_dot = 0.5 * rlenv.quat_mult_batch(q, omega_quat)
    v_dot = (np.einsum('nij,nj->ni', R, f_prop_ge + f_extra)
             / cache['m_arr'][:, None] + g_W)
    omega_dot = np.einsum('nij,nj->ni', cache['J_inv'],
                          tau_prop + tau_extra)
    return p_dot, q_dot, v_dot, omega_dot


# ─── Derivative: full pipeline mode ──────────────────────────────────────────

def _deriv_full(p, q, v, w, Omega, cache, cmd, U_bat, wind_W, dt):
    """Full Stages 2-8 derivative with motor dynamics.
    Integrates 5 state variables (p, q, v, ω, Ω).
    """
    N = p.shape[0]

    # Stage 2: ESC → steady-state motor speed
    Omega_ss = rlenv.esc_battery_batch(cmd, U_bat)               # (N,4)

    # Stage 3: Motor dynamics (first-order lag)
    Omega_dot = (Omega_ss - Omega) / k_mot                       # (N,4)
    Omega_eff = np.clip(Omega + 0.5 * dt * Omega_dot,
                        0.0, Omega_max)                          # (N,4)

    # Stage 4-5: Propeller forces + moment-arm aggregation
    f_props, tau_props = rlenv.propeller_force_torque_batch(
        Omega_eff, cache['c_l'], cache['c_d'])
    f_prop, tau_prop = rlenv.aggregate_forces_batch(f_props, tau_props)

    # Stage 6: Motor reaction torque (spinning propeller inertia)
    tau_mot = J_mp * np.sum(
        zeta[None, :, :] * Omega_dot[:, :, None], axis=1)       # (N,3)

    # All extras (aero, hub drag, flapping, gyroscopic, ground effect)
    f_extra, tau_extra, k_ge, R = _compute_extras(
        p, q, v, w, Omega_eff, cache, wind_W)

    # Boost propeller z-thrust by ground effect
    f_prop_ge = f_prop.copy()
    f_prop_ge[:, 2] *= k_ge

    # Rigid-body dynamics
    p_dot = v
    omega_quat = np.concatenate([np.zeros((N, 1)), w], axis=1)
    q_dot = 0.5 * rlenv.quat_mult_batch(q, omega_quat)
    v_dot = (np.einsum('nij,nj->ni', R, f_prop_ge + f_extra)
             / cache['m_arr'][:, None] + g_W)
    omega_dot = np.einsum('nij,nj->ni', cache['J_inv'],
                          tau_prop + tau_mot + tau_extra)

    return p_dot, q_dot, v_dot, omega_dot, Omega_dot


# ─── RK4: backward-compatible step ───────────────────────────────────────────

def rk4_rigid_body_step(state0, f_body, tau_body, params, dt, wind_W=None):
    """RK4 step with FIXED propeller wrench + velocity-dependent extras.

    Backward-compatible with v2.0 callers.  Omega is static.

    v3.0 improvements over v2.0:
      - Body aerodynamic drag included (params.fx_raw etc.)
      - Blade flapping uses per-rotor local velocity
      - Ground effect safely clamped
      - J_inv precomputed (4x faster)

    Parameters
    ----------
    state0     : dict with p_WB, q_WB, v_WB, omega_B, [Omega], [U_bat]
    f_body     : (3,) body-frame propeller force (held fixed)
    tau_body   : (3,) body-frame propeller torque (held fixed)
    params     : PARAMS_NANOBENCH or compatible dict
    dt         : timestep (s)
    wind_W     : (3,) world-frame wind velocity or None
    """
    cache = _make_cache(params, 1)
    Omega = state0.get("Omega")
    if Omega is None:
        Omega = hover_omega_from_params(params)
    Omega_b = Omega[None, :]

    w_W = (np.zeros((1, 3)) if wind_W is None
           else wind_W[None, :] if wind_W.ndim == 1
           else wind_W)

    p = state0["p_WB"][None, :]
    q = state0["q_WB"][None, :]
    v = state0["v_WB"][None, :]
    w = state0["omega_B"][None, :]
    f = f_body[None, :]
    tau = tau_body[None, :]

    def D(p_, q_, v_, w_):
        return _deriv_fixed(p_, q_, v_, w_, f, tau, cache, Omega_b, w_W)

    s = [p, q, v, w]
    k1 = list(D(*s))
    k2 = list(D(*[s[i] + 0.5*dt*k1[i] for i in range(4)]))
    k3 = list(D(*[s[i] + 0.5*dt*k2[i] for i in range(4)]))
    k4 = list(D(*[s[i] +     dt*k3[i] for i in range(4)]))

    out = [s[i] + (dt/6.0)*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
           for i in range(4)]
    out[1] = out[1] / np.linalg.norm(out[1], axis=1, keepdims=True)

    return dict(p_WB=out[0][0], q_WB=out[1][0], v_WB=out[2][0],
                omega_B=out[3][0], Omega=Omega,
                U_bat=state0.get("U_bat"))


# ─── RK4: full pipeline step ─────────────────────────────────────────────────

def rk4_full_step(state0, cmd, params, dt, wind_W=None):
    """Full-pipeline RK4 step with motor dynamics.

    Includes ALL physics: ESC model, motor lag (first-order), propeller
    thrust/drag, motor reaction torque, body aero drag, ground effect,
    hub drag, blade flapping, gyroscopic precession.

    Omega is evolved as a state variable and clamped to [0, Omega_max].

    Parameters
    ----------
    state0   : dict with p_WB, q_WB, v_WB, omega_B, Omega, U_bat
    cmd      : (4,) motor commands in [0, 1]
    params   : PARAMS_NANOBENCH or compatible dict
    dt       : timestep (s)
    wind_W   : (3,) world-frame wind velocity or None
    """
    cache = _make_cache(params, 1)

    w_W = (np.zeros((1, 3)) if wind_W is None
           else wind_W[None, :] if wind_W.ndim == 1
           else wind_W)

    p  = state0["p_WB"][None, :]
    q  = state0["q_WB"][None, :]
    v  = state0["v_WB"][None, :]
    w  = state0["omega_B"][None, :]
    Om = state0["Omega"][None, :]
    U  = np.array([state0.get("U_bat", 4.2)])
    cm = cmd[None, :]

    def D(p_, q_, v_, w_, O_):
        return _deriv_full(p_, q_, v_, w_, O_, cache, cm, U, w_W, dt)

    s = [p, q, v, w, Om]
    k1 = list(D(*s))
    k2 = list(D(*[s[i] + 0.5*dt*k1[i] for i in range(5)]))
    k3 = list(D(*[s[i] + 0.5*dt*k2[i] for i in range(5)]))
    k4 = list(D(*[s[i] +     dt*k3[i] for i in range(5)]))

    out = [s[i] + (dt/6.0)*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
           for i in range(5)]
    out[1] = out[1] / np.linalg.norm(out[1], axis=1, keepdims=True)
    out[4] = np.clip(out[4], 0.0, Omega_max)                    # enforce limits

    return dict(p_WB=out[0][0], q_WB=out[1][0], v_WB=out[2][0],
                omega_B=out[3][0], Omega=out[4][0],
                U_bat=float(U[0]))
