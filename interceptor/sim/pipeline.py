"""pipeline.py  --  full Crazyflie physics pipeline (rate PID -> RK4).

Scalar (single-env) pipeline, verbatim port of ``hover_env.py``'s
``_ph_*`` functions with the small parameterization needed for domain
randomization:

* Physical parameters (mass, inertia, lift/drag coeffs, aero polynomial
  coefficients) live in a :class:`PhysicsParams` container instead of
  module-level constants.
* Wind (optional) enters through air-relative body velocity.
* Ground effect (optional) boosts prop thrust near the floor.

With default parameters, ``wind=None`` and ``ge_gain=1.0`` the pipeline is
numerically identical to ``hover_env.py``.  See ``scripts/smoke_test.py``
for the byte-level parity check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import aero as _aero
from .constants import (
    DR_RANGES,
    NOM_C_D,
    NOM_C_L,
    NOM_FX_RAW,
    NOM_FY_RAW,
    NOM_FZ_RAW,
    NOM_J,
    NOM_M,
    NOM_TX_RAW,
    NOM_TY_RAW,
    NOM_TZ_RAW,
    ROTOR_DRAG_XYZ,
    PH_BAT,
    PH_C_D,
    PH_C_L,
    PH_CUT_THR,
    PH_DT,
    PH_G_W,
    PH_J,
    PH_J_MP,
    PH_KD,
    PH_KI,
    PH_K_MOT,
    PH_KP,
    PH_M,
    PH_MIX,
    PH_OMEGA_MAX,
    PH_R_P,
    PH_SPIN_SIGN,
    PH_ZETA,
)
from .quaternion import quat2rot, quat_mult, normalize_quat

__all__ = [
    "PhysicsParams",
    "sample_physics_params",
    "pid",
    "mixer", "mixer_capped",
    "esc",
    "prop_ft",
    "aggregate",
    "react",
    "rigid_body",
    "deriv",
    "rk4",
    "substeps",
    "control_step",
    "command_from_action",
]


# ============================================================
# Domain-randomizable parameter container
# ============================================================

@dataclass
class PhysicsParams:
    """All physical parameters that are fixed within an episode.

    ``sample_physics_params()`` perturbs these around their nominals; the
    pipeline is otherwise a pure function of this container.
    """

    mass: float = NOM_M
    J: np.ndarray = field(default_factory=lambda: np.array(PH_J, dtype=np.float64))
    c_l: float = NOM_C_L
    c_d: float = NOM_C_D
    fx_raw: np.ndarray = field(default_factory=lambda: np.array(NOM_FX_RAW, dtype=np.float64))
    fy_raw: np.ndarray = field(default_factory=lambda: np.array(NOM_FY_RAW, dtype=np.float64))
    fz_raw: np.ndarray = field(default_factory=lambda: np.array(NOM_FZ_RAW, dtype=np.float64))
    tx_raw: np.ndarray = field(default_factory=lambda: np.array(NOM_TX_RAW, dtype=np.float64))
    ty_raw: np.ndarray = field(default_factory=lambda: np.array(NOM_TY_RAW, dtype=np.float64))
    tz_raw: np.ndarray = field(default_factory=lambda: np.array(NOM_TZ_RAW, dtype=np.float64))
    # "polynomial": NanoBench-style polynomial (fixture parity; drag-limited to ~3 m/s, F10).
    # "rotor_drag": linear rotor drag a = -D v (EQ-SIX-11), the P3.5 fallback model.
    aero_model: str = "polynomial"
    drag_xyz: np.ndarray = field(default_factory=lambda: np.array(ROTOR_DRAG_XYZ, dtype=np.float64))

    @classmethod
    def rotor_drag(cls, **kw) -> "PhysicsParams":
        """Nominal parameters with the linear rotor-drag model (P3.5 fallback)."""
        return cls(aero_model="rotor_drag", **kw)

    @property
    def J_inv(self) -> np.ndarray:
        """Inverse inertia matrix (diagonal in the nominal case)."""
        return np.linalg.inv(self.J)


def sample_physics_params(
    rng: np.random.Generator,
    ranges: Optional[dict] = None,
) -> PhysicsParams:
    """Sample one episode's physics params with domain randomization.

    Ranges mirror ``swift_rl_env.py`` (m/J +/-10%, c_l/c_d +/-15%,
    aero +/-20%).
    """
    r = ranges or DR_RANGES
    lo, hi = r["m_frac"]
    m = NOM_M * rng.uniform(lo, hi)
    lo, hi = r["J_frac"]
    J = np.diag(NOM_J * rng.uniform(lo, hi, size=(3,)))
    lo, hi = r["c_l_frac"]
    c_l = NOM_C_L * rng.uniform(lo, hi)
    lo, hi = r["c_d_frac"]
    c_d = NOM_C_D * rng.uniform(lo, hi)

    lo, hi = r["aero_frac"]
    scale = rng.uniform(lo, hi)
    fx_raw = np.array(NOM_FX_RAW) * scale
    fum = rng.uniform(lo, hi)
    fy_raw = np.array(NOM_FY_RAW) * fum
    fz_raw = np.array(NOM_FZ_RAW) * rng.uniform(lo, hi)

    tx_raw = np.array(NOM_TX_RAW) * rng.uniform(lo, hi)
    ty_raw = np.array(NOM_TY_RAW) * rng.uniform(lo, hi)
    tz_raw = np.array(NOM_TZ_RAW) * rng.uniform(lo, hi)
    drag_xyz = np.array(ROTOR_DRAG_XYZ) * scale  # reuses the fx draw: no extra RNG call, legacy draw order kept

    return PhysicsParams(
        mass=float(m),
        J=J,
        c_l=float(c_l),
        c_d=float(c_d),
        fx_raw=fx_raw,
        fy_raw=fy_raw,
        fz_raw=fz_raw,
        tx_raw=tx_raw,
        ty_raw=ty_raw,
        tz_raw=tz_raw,
        drag_xyz=drag_xyz,
    )


# ============================================================
# Stage A -- rate PID controller
# ============================================================

def pid(
    omega_cmd: np.ndarray,
    omega_prev: np.ndarray,
    omega_prev2: np.ndarray,
    I_prev: np.ndarray,
    throttle_cut: bool,
    dt: float = PH_DT,
) -> tuple[np.ndarray, np.ndarray]:
    """Rate PID (roll, pitch, yaw).  Port of ``hover_env._ph_pid`` (EQ-CTRL-3)."""
    e = omega_cmd - omega_prev
    p = PH_KP * e
    i = np.zeros(3) if throttle_cut else I_prev + PH_KI * e * dt
    d = -PH_KD * (omega_prev - omega_prev2) / dt
    return p + i + d, i


# ============================================================
# Stage B -- motor mixer
# ============================================================

def mixer(u: np.ndarray, c_cmd: float) -> np.ndarray:
    """Saturation-aware motor mixer.  Port of ``hover_env._ph_mixer`` (EQ-SIX-4)."""
    raw = c_cmd + PH_MIX @ u
    vmax = float(np.max(raw))
    denom = vmax - c_cmd
    if vmax > 1.0 and abs(denom) > 1e-9:
        u = u * ((1.0 - c_cmd) / denom)
        raw = c_cmd + PH_MIX @ u
    return np.clip(raw, 0.0, 1.0)


def mixer_capped(u: np.ndarray, c_cmd: float) -> np.ndarray:
    """Motor mixer with the Crazyflie firmware thrust cap (attitude priority).

    Port of ``powerDistributionCap`` in crazyflie-firmware
    ``src/modules/src/power_distribution_quadrotor.c`` (lines 160-189, master, read
    2026-11-16): if the highest motor command exceeds the maximum, the excess is
    subtracted from all four motors, so the torque demand is kept and collective
    thrust is given up; the lower bound is then applied per motor
    (``capMinThrust``, idle thrust 0 here).  Identical to :func:`mixer` whenever no
    motor saturates.  :func:`mixer` does the opposite (keeps the collective, scales
    the torque to zero at full collective), which leaves no attitude authority when
    the demanded thrust reaches the maximum (BUGS #37).
    """
    raw = c_cmd + PH_MIX @ u
    excess = float(np.max(raw)) - 1.0
    if excess > 0.0:
        raw = raw - excess
    return np.clip(raw, 0.0, 1.0)


# ============================================================
# Stage 2 -- ESC / battery model
# ============================================================

def esc(cmd: np.ndarray, U_bat: float) -> tuple[np.ndarray, float]:
    """ESC polynomial -> steady-state motor speeds (constant-voltage battery).

    Port of ``hover_env._ph_esc``; equation EQ-SIX-7.
    """
    c = np.clip(cmd, 0.0, 1.0)
    oss = (PH_BAT[0] + PH_BAT[1] * U_bat
           + PH_BAT[2] * np.sqrt(c) + PH_BAT[3] * c
           + PH_BAT[4] * U_bat * np.sqrt(c))
    oss = np.where(c < PH_CUT_THR, 0.0, oss)
    return oss, U_bat


# ============================================================
# Stage 4 -- propeller force / reaction drag
# ============================================================

def prop_ft(Omega: np.ndarray, params: PhysicsParams) -> tuple[np.ndarray, np.ndarray]:
    """Per-propeller thrust + reaction-drag torque (EQ-SIX-5).

    Port of ``hover_env._ph_prop_ft``.
    """
    f = np.zeros((4, 3))
    tau = np.zeros((4, 3))
    for j in range(4):
        o2 = Omega[j] ** 2
        f[j] = [0.0, 0.0, params.c_l * o2]
        tau[j] = [0.0, 0.0, PH_SPIN_SIGN[j] * params.c_d * o2]
    return f, tau


# ============================================================
# Stage 5 -- force/torque aggregation with moment arms
# ============================================================

def aggregate(f_props: np.ndarray, tau_props: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Torque assembly at the arm positions (EQ-SIX-4).  Port of ``hover_env._ph_aggregate``."""
    f_tot = f_props.sum(axis=0)
    tau_tot = np.zeros(3)
    for j in range(4):
        tau_tot += tau_props[j] + np.cross(PH_R_P[j], f_props[j])
    return f_tot, tau_tot


# ============================================================
# Stage 6 -- motor reaction + gyroscopic inertial torques
# ============================================================

def react(Omega_dot: np.ndarray, omega_B: np.ndarray, params: PhysicsParams) -> tuple[np.ndarray, np.ndarray]:
    """Rotor reaction torque (EQ-SIX-8) and the inertial gyroscopic term (EQ-SIX-2).

    Port of ``hover_env._ph_react``.
    """
    tau_mot = PH_J_MP * np.sum(PH_ZETA * Omega_dot[:, None], axis=0)
    tau_iner = -np.cross(omega_B, params.J @ omega_B)
    return tau_mot, tau_iner


# ============================================================
# Stage 7 -- aerodynamics (via aero module, wind-aware)
# ============================================================

def aero_force_torque(
    q_WB: np.ndarray,
    v_WB: np.ndarray,
    Omega: np.ndarray,
    params: PhysicsParams,
    wind_W: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Body-frame aero forces/torques, optionally accounting for wind."""
    if wind_W is not None:
        v_rel_B = _aero.air_relative_body_velocity(q_WB, v_WB, wind_W)
    else:
        v_rel_B = quat2rot(q_WB).T @ v_WB
    if params.aero_model == "rotor_drag":
        return _aero.rotor_drag_force_torque(v_rel_B, mass=params.mass, drag_xyz=params.drag_xyz)
    return _aero.aero_force_torque(
        v_rel_B, Omega,
        mass=params.mass, J=params.J,
        fx_raw=params.fx_raw, fy_raw=params.fy_raw, fz_raw=params.fz_raw,
        tx_raw=params.tx_raw, ty_raw=params.ty_raw, tz_raw=params.tz_raw,
    )


# ============================================================
# Stage 8 -- rigid-body state derivatives
# ============================================================

def rigid_body(
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
    params: PhysicsParams,
    ge_gain: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """State derivatives (EQ-SIX-1, EQ-SIX-2, EQ-SIX-3, EQ-SIX-6).

    Port of ``hover_env._ph_rigid_body`` plus the optional ground-effect
    thrust boost of EQ-SIX-10 on the collective axis.
    """
    r_wb = quat2rot(q_WB)
    p_dot = v_WB
    q_dot = 0.5 * quat_mult(q_WB, np.r_[0.0, omega_B])
    if ge_gain != 1.0:
        f_prop = f_prop * np.array([1.0, 1.0, ge_gain])
    v_dot = r_wb @ (f_prop + f_aero) / params.mass + PH_G_W
    omega_dot = params.J_inv @ (tau_prop + tau_mot + tau_aero + tau_iner)
    omega_dot_cast = np.asarray(omega_dot, dtype=np.float64)
    Omega_dot = (Omega_ss - Omega) / PH_K_MOT
    return p_dot, q_dot, v_dot, omega_dot_cast, Omega_dot


# ============================================================
# Full derivative (Stages 2-8) -- RK4 RHS
# ============================================================

def deriv(
    p: np.ndarray,
    q: np.ndarray,
    v: np.ndarray,
    omega: np.ndarray,
    Omega: np.ndarray,
    cmd: np.ndarray,
    U_bat: float,
    params: PhysicsParams,
    wind_W: Optional[np.ndarray] = None,
    ge_gain: float = 1.0,
    dt: float = PH_DT,
    predict_rotor: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Port of ``hover_env._ph_deriv`` (wind / ground-effect aware).

    ``dt`` is the integration step (plan P3.1, defect F13; EQ-SIX-9).  It
    defaults to the historical ``PH_DT`` so the P1 fixtures replay
    bit-for-bit.

    ``predict_rotor=True`` keeps the legacy half-step rotor-speed prediction
    ``Omega + dt/2 * Omega_dot`` used for thrust and aero.  That prediction is
    an O(dt) model correction, so it limits the scheme to first-order
    convergence.  The validated 1000 Hz path (:func:`control_step`) disables
    it and evaluates thrust at the RK4 stage state instead, which restores
    fourth-order convergence (see ``tests/unit/test_sim_timestep.py``).
    """
    Omega_ss, _ = esc(cmd, U_bat)
    Omega_dot = (Omega_ss - Omega) / PH_K_MOT
    if predict_rotor:
        Omega_eff = np.maximum(Omega + 0.5 * dt * Omega_dot, 0.0)
    else:
        Omega_eff = np.maximum(Omega, 0.0)
    f_ps, t_ps = prop_ft(Omega_eff, params)
    f_p, tau_p = aggregate(f_ps, t_ps)
    tau_m, tau_i = react(Omega_dot, omega, params)
    f_a, tau_a = aero_force_torque(q, v, Omega_eff, params, wind_W)
    return rigid_body(
        f_p, f_a, tau_p, tau_m, tau_a, tau_i,
        q, v, omega, Omega_ss, Omega, params, ge_gain=ge_gain,
    )


# ============================================================
# Stage 9 -- classical RK4 integration
# ============================================================

def rk4(
    p: np.ndarray,
    q: np.ndarray,
    v: np.ndarray,
    omega: np.ndarray,
    Omega: np.ndarray,
    cmd: np.ndarray,
    U_bat: float,
    params: PhysicsParams,
    wind_W: Optional[np.ndarray] = None,
    ge_gain: float = 1.0,
    dt: float = PH_DT,
    predict_rotor: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Port of ``hover_env._ph_rk4``.  ``cmd``/``U_bat``/``wind_W``/``ge_gain``
    are held fixed (zero-order hold) across the four sub-steps.

    ``dt`` is the physical integration step (plan P3.1, defect F13); it
    defaults to ``PH_DT`` so the recorded fixtures replay unchanged.  Use
    :func:`control_step` to advance one control interval at the S8 rate
    (500 Hz control = 0.002 s) with the physics at 1000 Hz.

    ``predict_rotor`` is forwarded to :func:`deriv`; leave it ``True`` for
    legacy/fixture parity and disable it for the fourth-order validated path.
    """
    def _d(p_, q_, v_, om_, Om_):
        return deriv(p_, q_, v_, om_, Om_, cmd, U_bat, params, wind_W, ge_gain,
                     dt, predict_rotor)

    s = [p, q, v, omega, Omega]
    k1 = list(_d(*s))
    k2 = list(_d(*[s[i] + 0.5 * dt * k1[i] for i in range(5)]))
    k3 = list(_d(*[s[i] + 0.5 * dt * k2[i] for i in range(5)]))
    k4 = list(_d(*[s[i] + dt * k3[i] for i in range(5)]))
    out = [s[i] + (dt / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(5)]
    out[1] = normalize_quat(out[1])
    out[4] = np.maximum(out[4], 0.0)
    return tuple(out)


def substeps(control_dt: float, physics_dt: float) -> int:
    """Number of physics steps in one control step (plan S8).

    The ratio must be a positive integer; anything else is a configuration
    error, not something to round silently.

    Equation: EQ-SIX-9 (discrete sub-stepping of the RK4 state update).
    """
    ratio = float(control_dt) / float(physics_dt)
    n = int(round(ratio))
    if n < 1 or abs(ratio - n) > 1e-9:
        raise ValueError(
            f"control_dt={control_dt} is not an integer multiple of "
            f"physics_dt={physics_dt} (ratio {ratio})"
        )
    return n


def control_step(
    state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    cmd: np.ndarray,
    U_bat: float,
    params: PhysicsParams,
    control_dt: float = 0.002,
    physics_dt: float = 0.001,
    wind_W: Optional[np.ndarray] = None,
    ge_gain: float = 1.0,
    predict_rotor: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Advance one control interval with N zero-order-hold physics steps.

    Defaults are the plan S8 rates (500 Hz flight controller, 1000 Hz
    physics), i.e. two RK4 steps per control update.  The command, wind and
    ground-effect gain are held constant across the interval.

    The legacy half-step rotor prediction is off by default here: at 1000 Hz
    it changes the answer far below the integration error but it reduces the
    scheme to first order, which would hide model errors in the convergence
    study.  Pass ``predict_rotor=True`` for bit-for-bit legacy behaviour.
    """
    n = substeps(control_dt, physics_dt)
    dt = float(control_dt) / n
    s = state
    for _ in range(n):
        s = rk4(*s, cmd, U_bat, params, wind_W=wind_W, ge_gain=ge_gain,
                dt=dt, predict_rotor=predict_rotor)
    return s


# ============================================================
# Action mapping (shared by all stages)
# ============================================================

def command_from_action(action: np.ndarray) -> tuple[float, np.ndarray, bool]:
    """Map a Box([-1,1]^4) action -> physics commands.

    ``action`` : [thrust, roll, pitch, yaw]  (same convention as
    ``swift_live_demo_fitted.py`` TRUE INPUT / ``hover_env``).
    Returns ``(c_cmd, omega_cmd, throttle_cut)``.
    """
    from .constants import PH_C_SCALE, PH_CUT_THR, PH_W_SCALE
    thrust, roll, pitch, yaw = action
    c_cmd = float(max(0.0, thrust)) * PH_C_SCALE
    omega_cmd = np.array([roll, pitch, yaw]) * PH_W_SCALE
    throttle_cut = bool(thrust < PH_CUT_THR)
    return c_cmd, omega_cmd, throttle_cut