"""
hover_env.py  --  Gymnasium environment for Stage 1: Altitude Hold
==================================================================

TASK
----
The drone must reach and hold a *horizontal altitude plane* at a specified
height.  No lateral position target exists -- horizontal drift is unconstrained
and never penalised.  This decouples altitude control from lateral control so
the first curriculum stage is as simple as possible.

PHYSICS
-------
Verbatim port of the validated Crazyflie 2.1 physics pipeline from
swift_live_demo_fitted.py (the primary reference).  All stage functions,
physical constants, aerodynamic coefficients, and the RK4 integrator are
reproduced exactly -- only the driving loop and plotting code are removed.

  Stage A   -- Rate PID controller (roll, pitch, yaw)
  Stage B   -- Motor mixer (saturation-aware)
  Stage 2   -- ESC / battery model -> steady-state motor speeds
  Stage 3   -- Motor dynamics (first-order lag, absorbed into RK4)
  Stage 4   -- Propeller force / reaction-drag torque
  Stage 5   -- Force/torque aggregation with moment arms
  Stage 6   -- Motor-reaction + gyroscopic inertial torques
  Stage 7   -- Body aerodynamic forces/torques (polynomial fit)
  Stage 8   -- Rigid-body state derivatives
  Stage 9   -- Classical 4th-order Runge-Kutta integration (RK4)

ACTION SPACE
------------
Box([-1, 1]^4, float32)  ->  [thrust, roll, pitch, yaw]
  thrust  : positive half-range -> collective motor command;
            values <= PH_CUT_THR treated as throttle-cut (motors off).
  roll/pitch/yaw : desired body-rate commands, scaled by PH_W_SCALE rad/s.
  Convention identical to swift_live_demo_fitted.py TRUE INPUT so a trained
  policy and a human gamepad operator produce the same numerical commands.

OBSERVATION SPACE  (14-dim, float32)
--------------------------------------
  [0:6]   6D rotation  -- first two columns of R_{WB} (body->world).
           Continuous, no gimbal lock.  Noisy.
  [6:9]   World-frame velocity  v_{WB}  (m/s).  Noisy.
  [9:12]  Body-frame angular rate  omega_B  (rad/s).  Noisy.
  [12]    Altitude error  delta_z = z - z_target  (m),  clipped +-max_z_error.
           Positive -> above target;  negative -> below target.
  [13]    Vertical velocity  v_z  (m/s).  Noisy.

REWARD  (Stage 1 -- Altitude Hold)
-----------------------------------
  Dense (every step):
    r_alive  = +k_alive                        survival bonus
    r_alt    = -k_alt  * |delta_z|             altitude accuracy
    r_tilt   = -k_tilt * theta_tilt            attitude stability (rad)
    r_angvel = -k_av   * max(0,|omega|-omega_safe)^2  rate damping
    r_thrust = -k_thr  * |c_cmd - c_hover|    efficiency / hover trim
    r_smooth = -k_sm   * ||delta_a||^2         action smoothness

  Terminal (cause immediate episode end):
    -k_crash  if theta_tilt > 60 deg (flip) or hard ground contact
    -k_oob    if z < 0 or z > z_target+margin, or |x|/|y| > xy_limit

CURRICULUM STAGES
------------------
  1a  randomize_altitude=False  (default)
      Fixed altitude, spawn at target  ->  pure hold, easiest start.

  1b  randomize_altitude=True, spawn_at_target=True
      Altitude sampled from altitude_range each episode.
      Policy must read delta_z to know where to go.

  1c  randomize_altitude=True, spawn_at_target=False
      Spawns below target  ->  climb-then-hold.
      Strictly harder; advance only after 1b converges.

  2+  Horizontal target interception (separate environment/curriculum).

ISSUES FIXED vs. ORIGINAL hover_env.py
----------------------------------------
  1. Reward redesigned from 3D-position penalty -> altitude-plane penalty.
     XY drift no longer penalised (not part of this stage task).
  2. Alive bonus added (k_alive = +0.10/step) -- provides positive signal
     so early training can distinguish surviving episodes from crashes.
  3. render() stub added -- required by Gymnasium spec; check_env passes.
  4. close() stub added.
  5. Physics functions renamed _ph_* prefix to avoid namespace collisions.
  6. All module-level constants now PH_* (uppercase) -- uniform convention.
  7. Small random tilt perturbation at reset avoids zero-gradient starts.
  8. Small random XY jitter at reset improves lateral generalisation.
  9. Observation includes explicit delta_z and v_z (directly task-relevant).
 10. Action cast to float64 before physics; observation returned as float32.
 11. terminated/truncated flags cast to bool, reward to float -> no SB3 warnings.
 12. _GYM_OK guard allows import and smoke-test without gymnasium installed.
"""

from __future__ import annotations

__version__ = "2.0.0"   # 2024-09-10  Full rewrite: altitude-plane reward, uniform PH_* naming, Gymnasium-compliant

import numpy as np
from dataclasses import dataclass

# ------------------------------------------------------------------------------
# Optional gymnasium import -- file remains importable without it (smoke-test)
# ------------------------------------------------------------------------------
try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_OK = True
except ImportError:
    _GYM_OK = False

    class _StubEnv:
        metadata = {}

    class _StubBox:
        def __init__(self, low, high, shape=None, dtype=np.float32):
            self.low, self.high, self.shape, self.dtype = low, high, shape, dtype

        def sample(self):
            lo = np.broadcast_to(np.asarray(self.low, dtype=float), self.shape)
            hi = np.broadcast_to(np.asarray(self.high, dtype=float), self.shape)
            return np.random.uniform(lo, hi).astype(self.dtype)

    class _StubSpaces:
        Box = _StubBox

    gym    = type("gym", (), {"Env": _StubEnv})()
    spaces = _StubSpaces()


# ==============================================================================
# Section 1:  PHYSICS CORE
#   All functions prefixed _ph_ (physics helper) to avoid name collisions.
#   Math is verbatim from swift_live_demo_fitted.py; identifier names differ.
# ==============================================================================

# -- Section 1.1: Quaternion helpers -------------------------------------------

def _ph_quat2rot(q: np.ndarray) -> np.ndarray:
    """[w, x, y, z] quaternion -> 3x3 rotation matrix  (body -> world)."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y)    ],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z),   2*(y*z - w*x)    ],
        [2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def _ph_quat_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions, [w, x, y, z] convention."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)


# -- Section 1.2: Physical constants -------------------------------------------
# Source: swift_live_demo_fitted.py inline citations.
# PH_* prefix = Physics constant.

PH_M          = 0.04085                                        # kg    NanoBench flying mass
PH_J          = np.diag([2.3951e-5, 2.3951e-5, 3.2347e-5])   # kg*m^2  Forster 2015
PH_J_MP       = 2.0e-9                                         # kg*m^2  rotor+propeller inertia
PH_G_W        = np.array([0.0, 0.0, -9.81])                   # m/s^2   gravity (world frame)
PH_DT         = 0.01                                           # s       100 Hz control rate

PH_ARM        = 0.0397                                         # m     Crazyflie arm length
PH_R_P        = np.array([                                     # (4,3) propeller positions
    [ PH_ARM,  PH_ARM, 0.0],   # M1
    [-PH_ARM,  PH_ARM, 0.0],   # M2
    [-PH_ARM, -PH_ARM, 0.0],   # M3
    [ PH_ARM, -PH_ARM, 0.0],   # M4
])
PH_ZETA       = np.array([                                     # (4,3) spin-axis unit vectors
    [0., 0.,  1.],   # M1  CCW
    [0., 0., -1.],   # M2  CW
    [0., 0.,  1.],   # M3  CCW
    [0., 0., -1.],   # M4  CW
])
PH_SPIN_SIGN  = PH_ZETA[:, 2]                                  # (4,) +1 / -1 per motor

# Mixer sign matrix (4x3): columns = [roll, pitch, yaw] contribution per motor.
# Derived from arm geometry -> physically consistent with actual produced torques.
PH_MIX        = np.column_stack([
    np.sign(PH_R_P[:, 1]),    # roll  ~ sign(r_y)
    -np.sign(PH_R_P[:, 0]),   # pitch ~ -sign(r_x)
    PH_SPIN_SIGN,              # yaw   = spin-reaction direction
])

PH_C_L        = 2.618e-8    # N/(rad/s)^2   lift coeff  (NanoBench sysid hover data)
PH_C_D        = 5.45e-11    # N*m/(rad/s)^2 drag coeff  (c_l/480 ratio)
PH_K_MOT      = 0.02        # s             motor first-order lag time constant
PH_OMEGA_MAX  = 2800.0      # rad/s         maximum motor speed

PH_ETA        = 0.7                                            # motor efficiency
PH_BAT        = np.array([865.6, 379.9, 1053.4, -1818.9, -291.3])  # ESC/battery coeffs

PH_KP         = np.array([0.150, 0.150, 0.200])   # rate-PID proportional gains
PH_KI         = np.array([0.200, 0.200, 0.100])   # rate-PID integral gains
PH_KD         = np.array([0.003, 0.003, 0.000])   # rate-PID derivative gains

PH_CMD_MIN    = 0.0    # motor command lower bound (normalised)
PH_CMD_MAX    = 1.0    # motor command upper bound (normalised)
PH_C_SCALE    = 1.0    # collective scale: full stick -> cmd 1.0
PH_W_SCALE    = 2.0    # rate scale: stick +/-1 -> desired rate +/-2 rad/s
PH_CUT_THR    = 0.02   # throttle-cut threshold (stick near zero -> motors off)

# -- Aerodynamic polynomial coefficients --------------------------------------
# Fitted by least-squares regression against NanoBench / PX4 flight log data.
#
# FIX (units): raw values are in acceleration units (m/s^2, rad/s^2).
#   Pipeline expects force (N) and torque (N*m) -> multiply by m or J diagonal.
# FIX (offsets): pure Omega_bar^2-only terms (index [2] of fx/fy, taux/tauy)
#   encode propeller contributions already in Stages 4-6. Zeroed.
# FIX (taux/tauy): zeroed -- over-predicted roll/pitch torques 120-230x on
#   NanoBench configuration.

_AX_RAW   = np.array([ 6.582314e-02, -3.507132e-02,  0.0,            3.952726e-07])
_AY_RAW   = np.array([ 5.738530e-02,  2.022478e-02,  0.0,           -6.291393e-07])
_AZ_RAW   = np.array([ 3.610184e-02, -1.153378e-01,  0.0,           -5.881881e-07,
                        1.188474e-06, -1.705725e-08])
_ATX_RAW  = np.zeros(5)    # zeroed (see FIX note above)
_ATY_RAW  = np.zeros(5)    # zeroed (see FIX note above)
_ATZ_RAW  = np.array([-1.432000e-03, 7.798700e-02])

# Convert to force (N) / torque (N*m) units:
PH_FAX    = _AX_RAW  * PH_M
PH_FAY    = _AY_RAW  * PH_M
PH_FAZ    = _AZ_RAW  * PH_M
PH_FTX    = _ATX_RAW * PH_J[0, 0]
PH_FTY    = _ATY_RAW * PH_J[1, 1]
PH_FTZ    = _ATZ_RAW * PH_J[2, 2]

# Hover-equilibrium motor speed Omega_hover:  T_hover = m*g = 4*c_l*Omega^2
PH_OMEGA_HOVER = float(np.sqrt(PH_M * 9.81 / (4.0 * PH_C_L)))


def _ph_hover_cmd(u_bat: float = 4.2) -> float:
    """Invert ESC polynomial -> collective cmd that produces Omega_hover at u_bat volts."""
    try:
        from scipy.optimize import brentq
        def _f(c: float) -> float:
            return (PH_BAT[0] + PH_BAT[1]*u_bat
                    + PH_BAT[2]*np.sqrt(c) + PH_BAT[3]*c
                    + PH_BAT[4]*u_bat*np.sqrt(c)) - PH_OMEGA_HOVER
        return float(brentq(_f, 0.02, 1.0))
    except Exception:
        g  = np.linspace(0.02, 1.0, 20_000)
        om = (PH_BAT[0] + PH_BAT[1]*u_bat + PH_BAT[2]*np.sqrt(g)
              + PH_BAT[3]*g + PH_BAT[4]*u_bat*np.sqrt(g))
        return float(g[np.argmin(np.abs(om - PH_OMEGA_HOVER))])


PH_C_HOVER = _ph_hover_cmd()   # ~0.37 -- collective command at hover equilibrium


# -- Section 1.3: Stage functions ---------------------------------------------

def _ph_pid(omega_cmd, omega_prev, omega_prev2, I_prev, throttle_cut, dt):
    """Stage A -- rate PID controller (roll, pitch, yaw axes)."""
    e   = omega_cmd - omega_prev
    P   = PH_KP * e
    I   = np.zeros(3) if throttle_cut else I_prev + PH_KI * e * dt
    D   = -PH_KD * (omega_prev - omega_prev2) / dt
    return P + I + D, I


def _ph_mixer(u, c_cmd):
    """Stage B -- motor mixer with saturation-aware proportional rescaling."""
    raw  = c_cmd + PH_MIX @ u
    vmax = float(np.max(raw))
    denom = vmax - c_cmd
    if vmax > PH_CMD_MAX and abs(denom) > 1e-9:
        u   = u * ((PH_CMD_MAX - c_cmd) / denom)
        raw = c_cmd + PH_MIX @ u
    return np.clip(raw, PH_CMD_MIN, PH_CMD_MAX)


def _ph_esc(cmd, U_bat):
    """Stage 2 -- ESC/battery polynomial -> steady-state motor speeds Omega_ss."""
    c    = np.clip(cmd, 0.0, 1.0)
    Oss  = (PH_BAT[0] + PH_BAT[1]*U_bat
            + PH_BAT[2]*np.sqrt(c) + PH_BAT[3]*c
            + PH_BAT[4]*U_bat*np.sqrt(c))
    Oss  = np.where(c < 0.02, 0.0, Oss)
    return Oss, U_bat   # constant-voltage battery model


def _ph_prop_ft(Omega):
    """Stage 4 -- per-propeller thrust force and reaction drag torque."""
    f   = np.zeros((4, 3))
    tau = np.zeros((4, 3))
    for j in range(4):
        O2      = Omega[j] ** 2
        f[j]    = [0., 0., PH_C_L * O2]
        tau[j]  = [0., 0., PH_SPIN_SIGN[j] * PH_C_D * O2]
    return f, tau


def _ph_aggregate(f_props, tau_props):
    """Stage 5 -- aggregate propeller forces/torques with moment arms."""
    f_tot   = f_props.sum(axis=0)
    tau_tot = np.zeros(3)
    for j in range(4):
        tau_tot += tau_props[j] + np.cross(PH_R_P[j], f_props[j])
    return f_tot, tau_tot


def _ph_react(Omega_dot, omega_B):
    """Stage 6 -- motor reaction torque + gyroscopic inertial torque."""
    tau_mot  = PH_J_MP * np.sum(PH_ZETA * Omega_dot[:, None], axis=0)
    tau_iner = -np.cross(omega_B, PH_J @ omega_B)
    return tau_mot, tau_iner


def _ph_aero(v_B, Omega):
    """Stage 7 -- body-frame aerodynamic forces and torques (polynomial model)."""
    vx, vy, vz = v_B
    v_xy       = np.sqrt(vx**2 + vy**2)
    Ob2        = float(np.mean(Omega**2))   # Omega_bar^2

    fx   = PH_FAX @ [vx,  vx*abs(vx),  Ob2,      vx*Ob2]
    fy   = PH_FAY @ [vy,  vy*abs(vy),  Ob2,      vy*Ob2]
    fz   = PH_FAZ @ [vz,  vz*abs(vz),  v_xy**2,  v_xy*Ob2,  vz*Ob2,  v_xy*vz*Ob2]
    tx   = PH_FTX @ [vy,  vy*abs(vy),  Ob2,      vy*Ob2,    vy*abs(vy)*Ob2]
    ty   = PH_FTY @ [vx,  vx*abs(vx),  Ob2,      vx*Ob2,    vx*abs(vx)*Ob2]
    tz   = PH_FTZ @ [vx, vy]

    f_aero   = np.clip([fx, fy, fz], -50., 50.)
    tau_aero = np.clip([tx, ty, tz],  -5.,  5.)
    return np.asarray(f_aero, dtype=np.float64), np.asarray(tau_aero, dtype=np.float64)


def _ph_rigid_body(f_prop, f_aero, tau_prop, tau_mot, tau_aero, tau_iner,
                    q_WB, v_WB, omega_B, Omega_ss, Omega):
    """Stage 8 -- rigid-body state derivatives."""
    R_WB      = _ph_quat2rot(q_WB)
    p_dot     = v_WB
    q_dot     = 0.5 * _ph_quat_mult(q_WB, np.r_[0., omega_B])
    v_dot     = R_WB @ (f_prop + f_aero) / PH_M + PH_G_W
    omega_dot = np.linalg.inv(PH_J) @ (tau_prop + tau_mot + tau_aero + tau_iner)
    Omega_dot = (Omega_ss - Omega) / PH_K_MOT
    return p_dot, q_dot, v_dot, omega_dot, Omega_dot


def _ph_deriv(p, q, v, omega, Omega, cmd, U_bat):
    """Full Stages 2-8 derivative -- RHS function for RK4."""
    Omega_ss, _ = _ph_esc(cmd, U_bat)
    Omega_dot   = (Omega_ss - Omega) / PH_K_MOT
    Omega_eff   = np.maximum(Omega + 0.5 * PH_DT * Omega_dot, 0.)
    f_ps, t_ps  = _ph_prop_ft(Omega_eff)
    f_p,  tau_p = _ph_aggregate(f_ps, t_ps)
    tau_m, tau_i = _ph_react(Omega_dot, omega)
    v_B          = _ph_quat2rot(q).T @ v
    f_a,  tau_a  = _ph_aero(v_B, Omega_eff)
    return _ph_rigid_body(f_p, f_a, tau_p, tau_m, tau_a, tau_i,
                           q, v, omega, Omega_ss, Omega)


def _ph_rk4(p, q, v, omega, Omega, cmd, U_bat):
    """Stage 9 -- Classical 4th-order Runge-Kutta step.
    cmd and U_bat are held fixed (zero-order hold) across the 4 sub-steps,
    matching swift_live_demo_fitted.py exactly."""
    def D(p_, q_, v_, om_, Om_):
        return _ph_deriv(p_, q_, v_, om_, Om_, cmd, U_bat)

    s  = [p, q, v, omega, Omega]
    k1 = list(D(*s))
    k2 = list(D(*[s[i] + 0.5*PH_DT*k1[i] for i in range(5)]))
    k3 = list(D(*[s[i] + 0.5*PH_DT*k2[i] for i in range(5)]))
    k4 = list(D(*[s[i] +     PH_DT*k3[i] for i in range(5)]))
    out = [s[i] + (PH_DT / 6.) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
           for i in range(5)]
    out[1] /= np.linalg.norm(out[1])   # quaternion renormalisation
    out[4]  = np.maximum(out[4], 0.)   # motor speeds must be non-negative
    return out


# ==============================================================================
# Section 2:  REWARD CONFIGURATION
# ==============================================================================

@dataclass
class AltitudeHoldRewardConfig:
    """
    Reward weights for Stage 1 -- Altitude Hold.

    Tuning guide
    ------------
    Terminal terms (k_crash, k_oob) should dominate cumulative dense reward
    over an episode so crash-avoidance is always the dominant objective.
    With max_episode_steps=1000, a good episode earns approximately:
        k_alive * 1000  ~  100     (survival bonus)
        k_alt * err * 1000         (e.g. ~75 at 0.5 m error, k_alt=0.15)
    Setting k_crash = k_oob = 200 makes crashes ~2x worse than a perfect
    episode -- strong enough to dominate, not so large it crushes early signal.
    """

    # Dense reward terms
    k_alive:    float = 0.10   # +per-step survival bonus
    k_alt:      float = 0.15   # - per-metre altitude error
    k_tilt:     float = 0.40   # - per-radian tilt angle
    k_angvel:   float = 0.02   # - quadratic beyond omega_safe threshold
    omega_safe: float = 3.0    # rad/s -- angular rate below which no penalty
    k_thrust:   float = 0.15   # - deviation from hover-equilibrium collective
    k_smooth:   float = 0.05   # - squared action-delta ||delta_a||^2

    # Terminal penalties
    k_crash:    float = 200.0  # flip (tilt > threshold) or hard landing
    k_oob:      float = 200.0  # altitude / lateral out-of-bounds

    # Safety thresholds
    tilt_crash_threshold:       float = float(np.deg2rad(60.0))  # rad
    ground_contact_v_threshold: float = 1.0   # m/s -- hard-landing detection


# Backward-compatible aliases (used by existing notebooks)
HoverRewardConfig = AltitudeHoldRewardConfig


# ==============================================================================
# Section 3:  GYMNASIUM ENVIRONMENT
# ==============================================================================

_BASE_CLS = gym.Env if _GYM_OK else object


class AltitudeHoldEnv(_BASE_CLS):
    """
    Curriculum Stage 1 -- Altitude Hold  (Gymnasium-compatible).

    The drone earns reward for:
      - reaching the target altitude plane  (alt_err -> 0)
      - remaining level                     (tilt -> 0)
      - keeping angular rates calm          (|omega| below threshold)
      - using near-hover thrust             (fuel efficiency)
      - acting smoothly                     (actuator health)
      - surviving the episode               (alive bonus)

    Horizontal XY position is NOT penalised -- drone may drift freely
    within the lateral out-of-bounds limit (+-15 m).

    Observation : Box, float32, shape (14,)
    Action      : Box([-1,1]^4, float32) = [thrust, roll, pitch, yaw]
    """

    metadata = {"render_modes": ["human"], "render_fps": 100}

    def __init__(
        self,
        reward_config=None,
        target_altitude: float = 5.0,
        max_episode_steps: int = 1000,
        obs_noise_std: float = 0.01,
        seed=None,
        # Curriculum options
        randomize_altitude: bool = False,
        altitude_range: tuple = (2.0, 8.0),
        spawn_at_target: bool = True,
        spawn_offset_z: float = 2.0,
        # Observation clip
        max_z_error: float = 20.0,
    ):
        if _GYM_OK:
            super().__init__()

        self.cfg               = reward_config or AltitudeHoldRewardConfig()
        self.base_target_alt   = float(target_altitude)
        self.target_alt        = self.base_target_alt
        self.max_episode_steps = int(max_episode_steps)
        self.obs_noise_std     = float(obs_noise_std)
        self.randomize_altitude = bool(randomize_altitude)
        self.altitude_range    = (float(altitude_range[0]), float(altitude_range[1]))
        self.spawn_at_target   = bool(spawn_at_target)
        self.spawn_offset_z    = float(spawn_offset_z)
        self.max_z_error       = float(max_z_error)

        # World bounds
        self._z_margin  = 10.0    # OOB ceiling above target altitude
        self._xy_limit  = 15.0    # lateral OOB limit (large -- not penalised by reward)

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self._reset_state()

    # -- Gymnasium API ---------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """Reset to a fresh episode.  Returns (obs, info) per Gymnasium spec."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._reset_state()
        obs  = self._get_obs()
        info = {"target_alt": float(self.target_alt)}
        return (obs, info) if _GYM_OK else obs

    def step(self, action):
        """
        Advance one 100 Hz physics step.

        Parameters
        ----------
        action : array-like, shape (4,)
            [thrust, roll, pitch, yaw] in [-1, 1].

        Returns (Gymnasium API)
        -------
        obs, reward, terminated, truncated, info
        """
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        c_cmd, crashed_ground = self._physics_step(action)
        reward, terminated, info = self._compute_reward(action, c_cmd, crashed_ground)

        self._prev_action = action.copy()
        self._step_count += 1
        truncated = bool(self._step_count >= self.max_episode_steps)
        obs = self._get_obs()

        if _GYM_OK:
            return obs, float(reward), bool(terminated), truncated, info
        return obs, float(reward), bool(terminated or truncated), info

    def render(self):
        """Headless environment -- visual rendering not implemented.
        Trajectories are recorded via rollout functions in the training notebook."""
        return None

    def close(self):
        """Clean up resources (none in this headless environment)."""
        pass

    # -- Internal: state initialisation ----------------------------------------

    def _reset_state(self):
        """Initialise or re-initialise all internal simulation state."""
        # Sample or keep fixed target altitude
        if self.randomize_altitude:
            self.target_alt = float(self._rng.uniform(*self.altitude_range))
        else:
            self.target_alt = self.base_target_alt

        # Per-episode z bounds
        self._z_lo = 0.0
        self._z_hi = self.target_alt + self._z_margin

        # Spawn z: at target (hold only) or below (climb-then-hold)
        spawn_z = self.target_alt if self.spawn_at_target else \
                  max(0.2, self.target_alt - self.spawn_offset_z)

        # Small random XY jitter (+-0.3 m) -> improves lateral generalisation
        xy = self._rng.uniform(-0.3, 0.3, size=2)

        # Small random tilt (uniform 0-5 deg) -> avoids zero-gradient starts
        tilt_rad = self._rng.uniform(0.0, np.deg2rad(5.0))
        axis     = self._rng.standard_normal(3)
        axis    /= np.linalg.norm(axis) + 1e-12
        q_init   = np.r_[np.cos(tilt_rad / 2.0), np.sin(tilt_rad / 2.0) * axis]
        q_init  /= np.linalg.norm(q_init)

        # Drone state (naming convention: swift_live_demo_fitted.py)
        self._p_WB        = np.array([xy[0], xy[1], spawn_z], dtype=np.float64)
        self._q_WB        = q_init.astype(np.float64)
        self._v_WB        = np.zeros(3, dtype=np.float64)
        self._omega_B     = np.zeros(3, dtype=np.float64)
        self._omega_B_2   = np.zeros(3, dtype=np.float64)   # omega at step i-2 (PID D-term)
        self._Omega       = np.full(4, PH_OMEGA_HOVER, dtype=np.float64)  # motors pre-spun
        self._I_pid       = np.zeros(3, dtype=np.float64)
        self._U_bat       = 4.2    # V -- full 1S LiPo charge
        self._prev_action = np.zeros(4, dtype=np.float64)
        self._step_count  = 0

    # -- Internal: observation --------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Build the 14-dimensional observation vector with sensor noise."""
        R_WB  = _ph_quat2rot(self._q_WB)
        rot6d = R_WB[:, :2].ravel()   # first two columns -> 6D rotation (6,)

        rot_n = rot6d         + self._rng.normal(0., self.obs_noise_std, 6)
        v_n   = self._v_WB    + self._rng.normal(0., self.obs_noise_std, 3)
        w_n   = self._omega_B + self._rng.normal(0., self.obs_noise_std, 3)

        alt_err = float(np.clip(self._p_WB[2] - self.target_alt,
                                -self.max_z_error, self.max_z_error))
        vz_n    = float(v_n[2])   # vertical velocity (already noisy from v_n)

        return np.array([
            *rot_n,    # [0:6]   6D rotation
            *v_n,      # [6:9]   world-frame velocity
            *w_n,      # [9:12]  body angular rate
            alt_err,   # [12]    altitude error delta_z = z - z_target
            vz_n,      # [13]    vertical velocity v_z
        ], dtype=np.float32)

    # -- Internal: physics step ------------------------------------------------

    def _physics_step(self, action: np.ndarray):
        """Run one RK4 step through the full physics pipeline."""
        thrust, roll, pitch, yaw = action

        # Map [-1,1] action -> physics commands (identical to swift_live_demo_fitted.py)
        c_cmd        = float(max(0.0, thrust)) * PH_C_SCALE
        omega_cmd    = np.array([roll, pitch, yaw]) * PH_W_SCALE
        throttle_cut = bool(thrust < PH_CUT_THR)

        # Stage A -- Rate PID
        u_torque, self._I_pid = _ph_pid(
            omega_cmd, self._omega_B, self._omega_B_2,
            self._I_pid, throttle_cut, PH_DT)

        # Stage B -- Mixer
        cmd = _ph_mixer(u_torque, c_cmd)
        if throttle_cut:
            cmd = np.zeros(4)

        # Stages 2-9 -- RK4 integration
        p_n, q_n, v_n, om_n, Om_n = _ph_rk4(
            self._p_WB, self._q_WB, self._v_WB,
            self._omega_B, self._Omega, cmd, self._U_bat)

        # Battery voltage: single update per outer step (constant-voltage model)
        if not throttle_cut:
            _, self._U_bat = _ph_esc(cmd, self._U_bat)

        # Carry loop variables forward (omega_B_2 <- omega_B before update)
        self._omega_B_2 = self._omega_B.copy()
        self._p_WB      = p_n
        self._q_WB      = q_n
        self._v_WB      = v_n
        self._omega_B   = om_n
        self._Omega     = Om_n

        # Ground-plane collision model
        crashed_ground = False
        if self._p_WB[2] <= 0.0:
            if self._v_WB[2] < -self.cfg.ground_contact_v_threshold:
                crashed_ground = True
            self._p_WB[2] = 0.0
            self._v_WB[2] = max(float(self._v_WB[2]), 0.0)

        return c_cmd, crashed_ground

    # -- Internal: reward ------------------------------------------------------

    def _compute_reward(self, action: np.ndarray, c_cmd: float, crashed_ground: bool):
        """Compute step reward, termination flag, and info dict."""
        R_WB = _ph_quat2rot(self._q_WB)
        # Tilt: angle between body z-axis R_WB[:,2] and world z-axis [0,0,1]
        # R_WB[2,2] = z_B dot z_W = cos(theta_tilt)
        tilt       = float(np.arccos(np.clip(R_WB[2, 2], -1.0, 1.0)))
        alt_err    = float(abs(self._p_WB[2] - self.target_alt))
        angvel_n   = float(np.linalg.norm(self._omega_B))
        thr_dev    = float(abs(c_cmd - PH_C_HOVER))
        act_delta  = float(np.sum((action - self._prev_action) ** 2))

        # Dense reward components
        r_alive   = self.cfg.k_alive
        r_alt     = -self.cfg.k_alt    * alt_err
        r_tilt    = -self.cfg.k_tilt   * tilt
        r_angvel  = -self.cfg.k_angvel * max(0.0, angvel_n - self.cfg.omega_safe) ** 2
        r_thrust  = -self.cfg.k_thrust * thr_dev
        r_smooth  = -self.cfg.k_smooth * act_delta

        # Termination conditions
        flipped       = tilt > self.cfg.tilt_crash_threshold
        crashed       = flipped or crashed_ground
        out_of_bounds = (
            abs(self._p_WB[0]) > self._xy_limit or
            abs(self._p_WB[1]) > self._xy_limit or
            self._p_WB[2] < self._z_lo or
            self._p_WB[2] > self._z_hi
        )

        reward     = r_alive + r_alt + r_tilt + r_angvel + r_thrust + r_smooth
        terminated = False

        if crashed:
            reward    -= self.cfg.k_crash
            terminated = True
        if out_of_bounds:
            reward    -= self.cfg.k_oob
            terminated = True

        info = dict(
            target_alt    = float(self.target_alt),
            alt_err       = alt_err,
            tilt          = tilt,
            angvel_norm   = angvel_n,
            thrust_dev    = thr_dev,
            r_alive       = r_alive,
            r_alt         = r_alt,
            r_tilt        = r_tilt,
            r_angvel      = r_angvel,
            r_thrust      = r_thrust,
            r_smooth      = r_smooth,
            crashed       = bool(crashed),
            out_of_bounds = bool(out_of_bounds),
        )
        return reward, bool(terminated), info


# Backward-compatible aliases
HoverEnv = AltitudeHoldEnv


# ==============================================================================
# Section 4:  SMOKE TEST
# ==============================================================================

if __name__ == "__main__":

    def _smoke(label, env, steps=500):
        result = env.reset()
        obs    = result[0] if _GYM_OK else result
        print(f"\n[{label}]")
        print(f"  obs shape   : {obs.shape}")
        print(f"  target_alt  : {env.target_alt:.2f} m")
        print(f"  spawn_z     : {env._p_WB[2]:.2f} m")
        print(f"  obs[12] dz  : {obs[12]:.3f} m  (altitude error at reset)")

        total_r = 0.0
        for i in range(steps):
            act = np.array([
                float(np.clip(np.random.normal(2*PH_C_HOVER - 1, 0.05), -1, 1)),
                float(np.random.normal(0, 0.02)),
                float(np.random.normal(0, 0.02)),
                float(np.random.normal(0, 0.02)),
            ])
            res = env.step(act)
            if _GYM_OK:
                obs, r, terminated, truncated, info = res
                done = terminated or truncated
            else:
                obs, r, done, info = res
            total_r += r
            if done:
                print(f"  ended at step {i}  "
                      f"crashed={info['crashed']}  oob={info['out_of_bounds']}")
                break
        print(f"  total_reward : {total_r:.2f}")
        print(f"  final z      : {env._p_WB[2]:.3f} m   "
              f"alt_err = {abs(env._p_WB[2] - env.target_alt):.3f} m")

    print("=" * 60)
    print("hover_env.py -- AltitudeHoldEnv smoke tests")
    print("=" * 60)

    _smoke("Stage 1a -- fixed altitude, spawn at target",
           AltitudeHoldEnv(seed=0))

    _smoke("Stage 1b -- random altitude, spawn at target",
           AltitudeHoldEnv(seed=1, randomize_altitude=True, altitude_range=(3., 7.)))

    _smoke("Stage 1c -- random altitude, spawn below target",
           AltitudeHoldEnv(seed=2, randomize_altitude=True,
                            spawn_at_target=False, spawn_offset_z=2.0))

    if _GYM_OK:
        try:
            from stable_baselines3.common.env_checker import check_env
            print("\nRunning SB3 check_env ...")
            check_env(AltitudeHoldEnv(seed=42), warn=True)
            print("check_env passed.")
        except ImportError:
            print("\nstable_baselines3 not installed -- skipping check_env.")
