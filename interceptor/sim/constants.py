"""
constants.py  --  Crazyflie 2.1 physics constants (single canonical source)
=============================================================================

All values are verbatim ports of the validated physics constants from
``hover_env.py`` (originally from ``swift_live_demo_fitted.py``).  The
interceptor-training pipeline is the single consumer of these constants;
nothing else re-defines them.

Two families of constants exist:

* ``PH_*``                   -- fixed physical constants (geometry, PID, ...).
* ``_*_NOM`` / ``NOM_*``     -- nominal values of *domain-randomizable*
                                parameters (mass, inertia, lift coeff, aero
                                polynomial coefficients, ...).  When domain
                                randomization is disabled the pipeline uses
                                exactly these nominals, which reproduce
                                ``hover_env.py`` behaviour bit-for-bit.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    # fixed constants
    "PH_M", "PH_J", "PH_J_MP", "PH_G_W", "PH_DT",
    "PH_ARM", "PH_R_P", "PH_ZETA", "PH_SPIN_SIGN", "PH_MIX",
    "PH_C_L", "PH_C_D", "PH_K_MOT", "PH_OMEGA_MAX",
    "PH_ETA", "PH_BAT",
    "PH_KP", "PH_KI", "PH_KD",
    "PH_CMD_MIN", "PH_CMD_MAX", "PH_C_SCALE", "PH_W_SCALE", "PH_CUT_THR",
    "PH_OMEGA_HOVER", "PH_C_HOVER",
    "PH_TILT_CRASH_THRESHOLD", "PH_GROUND_CONTACT_V_THRESHOLD",
    # domain-randomizable nominal values
    "NOM_M", "NOM_J", "NOM_C_L", "NOM_C_D",
    "NOM_FX_RAW", "NOM_FY_RAW", "NOM_FZ_RAW",
    "NOM_TX_RAW", "NOM_TY_RAW", "NOM_TZ_RAW", "ROTOR_DRAG_XYZ",
    "DR_RANGES",
    # ground effect / wind
    "GROUND_EFFECT_HEIGHT", "GROUND_EFFECT_MAX_GAIN",
    "WIND_MEAN_STD", "WIND_THETA_RANGE", "WIND_SIGMA_RANGE",
    "RATE_CMD_SCALE",
]

# ---------------------------------------------------------------------------
# Fixed physical constants (identical to hover_env.py)
# ---------------------------------------------------------------------------

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
# ESC/battery coeffs -> steady-state speed  Omega_ss(c) = PH_BAT[0] + PH_BAT[1]*U
# + PH_BAT[2]*sqrt(c) + PH_BAT[3]*c + PH_BAT[4]*U*sqrt(c).
#
# BLOCKING FIX (2026-10-06, BUGS.md #13/R-1): the previously shipped fit
# ``[865.6, 379.9, 1053.4, -1818.9, -291.3]`` was monotonically DECREASING
# (Omega_ss(0.02)~2400 -> Omega_ss(1.0)~472), i.e. raising the throttle slowed
# the rotors.  That inverted both control loops at once:
#   * collective: thrust action up -> thrust down (altitude loop = +feedback)
#   * attitude:   mixer column sign assumed thrust rises with cmd, so the
#     rate PID became positive feedback (roll/pitch/yaw diverged in <0.3 s)
# Every rendered video and every trained checkpoint crashed with
# ``crashed_flip`` within 12-29 steps as a result.
#
# Replacement is the monotone increasing two-term fit  Omega_ss(c) = A + B*sqrt(c)
# anchored on the two physically known points:
#   Omega_ss(PH_C_HOVER) = PH_OMEGA_HOVER (1956.211093185006, thrust == weight)
#   Omega_ss(1.0)        = PH_OMEGA_MAX   (2800.0, full stick)
# The hover command is re-derived below by root-finding, so it stays at the
# documented value (0.2325374363535...), and the sensitivity at hover
# (dOmega/dc ~ +1687 rad/s per unit cmd) keeps the same magnitude the rate PID
# was tuned against.  Mirrored in ``hover_env.py`` so Stage-1 parity holds.
PH_BAT        = np.array([1170.36749799968, 0.0, 1629.63250200032, 0.0, 0.0])

PH_KP         = np.array([0.150, 0.150, 0.200])   # rate-PID proportional gains
PH_KI         = np.array([0.200, 0.200, 0.100])   # rate-PID integral gains
PH_KD         = np.array([0.001, 0.001, 0.000])   # rate-PID derivative gains
# BLOCKING FIX (2026-10-06, BUGS.md #16/R-3): Kd was 0.003 on roll/pitch.
# With d = -Kd*(w_k - w_{k-1})/dt the derivative term acts over a 1-step delay,
# and the closed-loop plant gain here is K ~ 4.8 (rad/s) per unit u per step,
# giving an effective Nyquist factor K*Kd/dt ~ 1.4 > 1: the rate loop sat in a
# permanent period-2 limit cycle (w alternating 1.02 / -0.63 rad/s for a 0.2
# rad/s setpoint) and any attitude error drove the airframe into a crash.
# 0.001 puts the factor at ~0.48 (well inside unity); measured rate tracking
# error drops from mean 0.805 to 0.001 rad/s.  Mirrored in ``hover_env.py``.

PH_CMD_MIN    = 0.0    # motor command lower bound (normalised)
PH_CMD_MAX    = 1.0    # motor command upper bound (normalised)
PH_C_SCALE    = 1.0    # collective scale: full stick -> cmd 1.0
PH_W_SCALE    = 2.0    # rate scale: stick +/-1 -> desired rate +/-2 rad/s
PH_CUT_THR    = 0.02   # throttle-cut threshold (stick near zero -> motors off)

RATE_CMD_SCALE = PH_W_SCALE   # action [-1,1] -> desired body rate (rad/s)

# Safety thresholds (shared by reward / termination logic)
PH_TILT_CRASH_THRESHOLD       = float(np.deg2rad(60.0))   # rad
PH_GROUND_CONTACT_V_THRESHOLD = 1.0                       # m/s

# ---------------------------------------------------------------------------
# Aerodynamic polynomial coefficients (raw, acceleration units)
# ---------------------------------------------------------------------------
# Fitted by least-squares regression against NanoBench / PX4 flight log data.
# ``_NOM_*_RAW`` are in acceleration units; the pipeline multiplies by mass /
# inertia at evaluation time so that domain randomization of m and J scales
# the aero forces correctly (same layout as swift_rl_env.py).

# BLOCKING FIX (2026-10-06, BUGS.md #14/R-2): two velocity x Omega_bar^2
# coefficients carried the WRONG SIGN, i.e. they pushed the airframe ALONG its
# motion instead of opposing it.  At hover Omega_bar^2 = 3.83e6 the old
# fz[4] = +1.188474e-06 gave an "anti-damping" of +4.55 (m/s^2)/(m/s) in z and
# fx[3] = +3.952726e-07 gave +1.51 (m/s^2)/(m/s) in x -- both unstable poles,
# so any vertical/lateral velocity ran away exponentially (tau ~ 0.22 s in z).
# That is why every rendered video ended with a crash inside 1 s and why a
# nominal hover fell 5 m in 1.4 s.  Aerodynamic terms must be dissipative, so
# both signs are flipped (magnitudes untouched -- they were fitted from
# collinear features, the sign is what is unphysical).  Mirrored in
# ``hover_env.py`` so Stage-1 parity still holds.
NOM_FX_RAW   = np.array([ 6.582314e-02, -3.507132e-02,  0.0,           -3.952726e-07])
NOM_FY_RAW   = np.array([ 5.738530e-02,  2.022478e-02,  0.0,           -6.291393e-07])
NOM_FZ_RAW   = np.array([ 3.610184e-02, -1.153378e-01,  0.0,          -5.881881e-07,
                         -1.188474e-06, -1.705725e-08])
NOM_TX_RAW   = np.zeros(5)    # zeroed (over-predicted roll/pitch torques, see hover_env.py)
NOM_TY_RAW   = np.zeros(5)    # zeroed
NOM_TZ_RAW   = np.array([-1.432000e-03, 7.798700e-02])

# Linear rotor-drag coefficients D = diag(kx, ky, kz) [1/s], EQ-SIX-11.  ASSUMPTION:
# order of magnitude of published quadrotor rotor-drag identifications
# (Faessler et al., RA-L 2018); NOT fitted to a Crazyflie.  Replace on NanoBench refit (P3.5).
ROTOR_DRAG_XYZ = (0.5, 0.5, 0.7)

# ---------------------------------------------------------------------------
# Hover equilibrium
# ---------------------------------------------------------------------------

PH_OMEGA_HOVER = float(np.sqrt(PH_M * 9.81 / (4.0 * PH_C_L)))


def _hover_cmd(u_bat: float = 4.2) -> float:
    """Invert ESC polynomial -> collective cmd that produces Omega_hover."""
    try:
        from scipy.optimize import brentq

        def _f(c: float) -> float:
            return (PH_BAT[0] + PH_BAT[1] * u_bat
                    + PH_BAT[2] * np.sqrt(c) + PH_BAT[3] * c
                    + PH_BAT[4] * u_bat * np.sqrt(c)) - PH_OMEGA_HOVER

        return float(brentq(_f, 0.02, 1.0))
    except Exception:  # pragma: no cover - scipy is a hard requirement
        g = np.linspace(0.02, 1.0, 20_000)
        om = (PH_BAT[0] + PH_BAT[1] * u_bat + PH_BAT[2] * np.sqrt(g)
              + PH_BAT[3] * g + PH_BAT[4] * u_bat * np.sqrt(g))
        return float(g[np.argmin(np.abs(om - PH_OMEGA_HOVER))])


PH_C_HOVER = _hover_cmd()   # 0.23253743635354834 -- collective command at hover

# ---------------------------------------------------------------------------
# Domain-randomization nominal values + default ranges
# ---------------------------------------------------------------------------
# Ranges mirror swift_rl_env.py (state.reset): m/J +/-10%, c_l/c_d +/-15%,
# aero coefficients +/-20%.

NOM_M   = PH_M
NOM_J   = np.array([2.3951e-5, 2.3951e-5, 3.2347e-5])   # diagonal, (3,)
NOM_C_L = PH_C_L
NOM_C_D = PH_C_D

DR_RANGES = {
    "m_frac":         (0.90, 1.10),
    "J_frac":         (0.90, 1.10),
    "c_l_frac":       (0.85, 1.15),
    "c_d_frac":       (0.85, 1.15),
    "aero_frac":      (0.80, 1.20),
}

# ---------------------------------------------------------------------------
# Ground effect (from swift_rl_env.py) -- thrust boost near the floor
# ---------------------------------------------------------------------------

GROUND_EFFECT_HEIGHT   = 0.15    # m -- vertical scale of the correction
GROUND_EFFECT_MAX_GAIN = 0.20    # max fractional thrust boost at the floor

# ---------------------------------------------------------------------------
# Wind / Ornstein-Uhlenbeck process (from swift_rl_env.py)
# ---------------------------------------------------------------------------
# wind_mean sampled N(0, WIND_MEAN_STD) per axis, theta and sigma uniform per
# episode.  Applies as air-relative velocity into the aerodynamic model.

WIND_MEAN_STD    = 1.5
WIND_THETA_RANGE = (0.2, 1.0)
WIND_SIGMA_RANGE = (0.3, 2.0)

# U_BAT defaults
U_BAT_FULL = 4.2  # V -- full 1S LiPo charge