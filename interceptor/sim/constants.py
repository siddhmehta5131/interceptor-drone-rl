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
    "PH_C_L", "PH_C_D", "PH_K_MOT", "PH_OMEGA_MAX", "LEGACY_V1", "esc_omega_ss", "esc_cmd_for_omega",
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

# Motor / propeller constants: Forster (2015), System Identification of the Crazyflie 2.0
# Nano Quadrocopter, ETH Zurich, doi:10.3929/ethz-b-000214143 (read 2026-11-15; BUGS #34).
# Derivation and comparison: docs/crazyflie_forster_comparison.md; recomputed from the
# thesis mappings by tests/unit/test_forster_constants.py.
#   thrust per motor   f(cmd)     = 2.130295e-11 cmd^2 + 1.032633e-6 cmd + 5.484560e-4 N   (eq. 3.3)
#   rotor speed        omega(cmd) = 0.04076521 cmd + 380.8359 rad/s, cmd in [0, 65535]     (Sec. 3.3.2)
#   torque / thrust    0.005964552 m                                                       (eq. 3.5)
#   command -> thrust  H(z) = 7.2345374e-8 / (1 - 0.9695404 z^-1) at 500 Hz                (eq. 3.16)
FORSTER_THRUST_POLY = (2.130295e-11, 1.032633e-6, 5.484560e-4)
FORSTER_OMEGA_LIN   = (0.04076521, 380.8359)
FORSTER_TORQUE_PER_THRUST = 0.005964552
FORSTER_POLE, FORSTER_TS = 0.9695404, 0.002
PH_C_L        = 1.817e-8    # N/(rad/s)^2   least squares of f(cmd) = c_l omega(cmd)^2 over cmd 10000..60000 (step 1000)
PH_C_D        = 1.084e-10   # N*m/(rad/s)^2 = 0.005964552 m x PH_C_L (eq. 3.5 slope)
PH_K_MOT      = 0.0647      # s             -Ts / ln(pole) of eq. (3.16): command-to-thrust first-order lag
PH_OMEGA_MAX  = 3052.4      # rad/s         omega(65535) from the Sec. 3.3.2 map

# Values used until 2026-11-15 (own NanoBench fit, assumed lag, two-point ESC fit).  Kept only
# so the golden fixtures recorded from the predecessor simulator can be replayed
# (tests/unit/test_fixture_parity_sim.py); nothing else may use them.
LEGACY_V1 = {
    "c_l": 2.618e-8, "c_d": 5.45e-11, "k_mot": 0.02, "omega_max": 2800.0,
    "bat": np.array([1170.36749799968, 0.0, 1629.63250200032, 0.0, 0.0]),
}

PH_ETA        = 0.7                                            # motor efficiency (unused by the v2 pipeline)
# ESC map -> steady-state speed  Omega_ss(c) = PH_BAT[0] + PH_BAT[1]*U + PH_BAT[2]*sqrt(c)
# + PH_BAT[3]*c + PH_BAT[4]*U*sqrt(c), c = normalised command in [0, 1] (EQ-SIX-7).
# Since 2026-11-15 this is Forster's measured linear map omega = 380.8359 + 0.04076521 cmd
# with cmd = 65535 c, i.e. Omega_ss(c) = 380.8359 + 2671.548 c (no battery-voltage term:
# the thesis map was measured on a charged cell).  The thesis says the map is not valid
# below cmd 1000 (c < 0.015); the simulator cuts the motors below PH_CUT_THR = 0.02.
# It replaces the two-point sqrt fit of BUGS R-1 (LEGACY_V1["bat"]).
PH_BAT        = np.array([FORSTER_OMEGA_LIN[1], 0.0, 0.0, FORSTER_OMEGA_LIN[0] * 65535.0, 0.0])


def esc_omega_ss(c, u_bat: float = 4.2, bat=None):
    """Steady-state rotor speed (rad/s) for normalised command ``c`` (no throttle cut)."""
    b = PH_BAT if bat is None else bat
    s = np.sqrt(np.clip(c, 0.0, 1.0))
    return b[0] + b[1] * u_bat + (b[2] + b[4] * u_bat) * s + b[3] * s * s


def esc_cmd_for_omega(omega: float, u_bat: float = 4.2, bat=None) -> float:
    """Inverse of :func:`esc_omega_ss`: command in [0, 1] giving rotor speed ``omega``."""
    b = PH_BAT if bat is None else bat
    c0, c1, c2 = b[0] + b[1] * u_bat - float(omega), b[2] + b[4] * u_bat, b[3]
    if abs(c2) < 1e-12:
        s = -c0 / c1
    else:
        s = (-c1 + np.sqrt(max(c1 * c1 - 4.0 * c2 * c0, 0.0))) / (2.0 * c2)
    return float(np.clip(s, 0.0, 1.0) ** 2)


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
# Faessler, Franchi & Scaramuzza (2018), arXiv:1712.02402, Sec. VII-C, 610 g quadrotor:
# d_x = 0.544 / 0.491 1/s, d_y = 0.386 / 0.236 1/s (circle / lemniscate).  kx = ky = 0.5
# is inside that d_x range and above the d_y range.  The paper does not identify d_z
# (it sets d_z = 0).  kz is set equal to kx (isotropic) so all three coefficients share
# one justification; kz = 0 would leave vertical speed unbounded in this model.
# Changed 2026-11-14 from the unsourced kz = 0.7 (BUGS #27).
ROTOR_DRAG_XYZ = (0.5, 0.5, 0.5)

# ---------------------------------------------------------------------------
# Hover equilibrium
# ---------------------------------------------------------------------------

PH_OMEGA_HOVER = float(np.sqrt(PH_M * 9.81 / (4.0 * PH_C_L)))


def _hover_cmd(u_bat: float = 4.2) -> float:
    """Collective command that produces Omega_hover (inverse of the ESC map)."""
    return esc_cmd_for_omega(PH_OMEGA_HOVER, u_bat)


PH_C_HOVER = _hover_cmd()   # 0.736 with the Forster map (hover at 2348 rad/s)

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