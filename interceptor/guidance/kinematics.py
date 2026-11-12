"""Engagement geometry, 3-D APN and kinematic constraints (plan P1.7, P5.5-P5.6).

All vectors are world ``x north, y west, z up``.  The formulas are invariant
under the NED -> z-up rotation used by the original notebook, so the golden
fixture replay differs only by the frame transform ``T = diag(1, -1, -1)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .platform import InterceptorLimits

__all__ = [
    "EngagementKinematics",
    "compute_engagement_kinematics",
    "compute_optimal_intercept_oracle",
    "compute_3d_apn_command",
    "compute_axial_acceleration",
    "apply_interceptor_constraints",
    "heading_rate_from_acceleration",
    "closest_approach_distance",
]


@dataclass
class EngagementKinematics:
    r_vec: np.ndarray
    r: float
    v_rel: np.ndarray
    v_cl: float
    los_vec: np.ndarray
    los_rate_vec: np.ndarray
    t_go: float
    zem_perp: np.ndarray
    zem_norm: float


def compute_engagement_kinematics(p_T, v_T, a_T, p_M, v_M,
                                 r_min_clamp: float = 15.0) -> EngagementKinematics:
    p_T = np.asarray(p_T, dtype=float)
    v_T = np.asarray(v_T, dtype=float)
    a_T = np.asarray(a_T, dtype=float)
    p_M = np.asarray(p_M, dtype=float)
    v_M = np.asarray(v_M, dtype=float)
    r_vec = p_T - p_M
    r = float(np.linalg.norm(r_vec))
    if r < 1e-6:
        r = 1e-6
    los_vec = r_vec / r
    v_rel = v_T - v_M
    v_cl = float(-np.dot(v_rel, los_vec))
    r_eff = max(r, r_min_clamp)
    los_rate_vec = (v_rel + v_cl * los_vec) / r_eff
    t_go = float(np.clip(r / max(v_cl, 1.0), 0.05, 20.0)) if v_cl > 0 else 20.0
    zem_vec = v_rel * t_go + 0.5 * a_T * t_go ** 2
    zem_perp = zem_vec - np.dot(zem_vec, los_vec) * los_vec
    return EngagementKinematics(r_vec, r, v_rel, v_cl, los_vec, los_rate_vec,
                                t_go, zem_perp, float(np.linalg.norm(zem_perp)))


def compute_optimal_intercept_oracle(p_int, v_int, p_target, v_target,
                                     a_target, v_max, a_max,
                                     t_go) -> Tuple[np.ndarray, np.ndarray]:
    """Closed-form boundary-value intercept oracle (critic only, P5.10)."""
    p_int = np.asarray(p_int, dtype=float)
    v_int = np.asarray(v_int, dtype=float)
    p_target = np.asarray(p_target, dtype=float)
    v_target = np.asarray(v_target, dtype=float)
    a_target = np.asarray(a_target, dtype=float)
    T = float(np.clip(t_go, 0.1, 20.0))
    pf = p_target + v_target * T + 0.5 * a_target * T ** 2
    vf = v_target + a_target * T
    opt_a = (6.0 * (pf - p_int) - 4.0 * T * v_int - 2.0 * T * vf) / T ** 2
    na = float(np.linalg.norm(opt_a))
    if na > a_max and na > 1e-9:
        opt_a = opt_a * (a_max / na)
    opt_v = (pf - p_int) / T
    nv = float(np.linalg.norm(opt_v))
    if nv > v_max and nv > 1e-9:
        opt_v = opt_v * (v_max / nv)
    return opt_a, opt_v


def compute_axial_acceleration(v_M, limits: InterceptorLimits, los_vec,
                               closing_speed: Optional[float] = None) -> np.ndarray:
    """Separate axial speed-management term (P5.5; notebook v2 boost)."""
    v_M = np.asarray(v_M, dtype=float)
    los_vec = np.asarray(los_vec, dtype=float)
    speed = float(np.linalg.norm(v_M))
    v_cl = float(closing_speed) if closing_speed is not None else float(np.dot(v_M, los_vec))
    if speed < 0.96 * limits.v_max or v_cl < 0.45 * limits.v_max:
        if limits.omnidirectional:
            return 0.80 * limits.a_t_max * los_vec
        return limits.a_t_max * (v_M / max(speed, 1e-6))
    return np.zeros(3)


def compute_3d_apn_command(kin: EngagementKinematics, a_T_est, v_M,
                           limits: InterceptorLimits, N_nav: float = 4.0,
                           include_axial_boost: bool = True) -> np.ndarray:
    """Textbook perpendicular APN, optionally plus the axial speed term."""
    v_rel = kin.v_rel
    a_T_est = np.asarray(a_T_est, dtype=float)
    v_rel_perp = v_rel - np.dot(v_rel, kin.los_vec) * kin.los_vec
    a_T_perp = a_T_est - np.dot(a_T_est, kin.los_vec) * kin.los_vec
    a_pn = N_nav * v_rel_perp / max(kin.t_go, 0.06)
    a_aug = 0.5 * N_nav * a_T_perp
    if not include_axial_boost:
        return a_pn + a_aug
    return a_pn + a_aug + compute_axial_acceleration(
        v_M, limits, kin.los_vec, closing_speed=kin.v_cl)


def apply_interceptor_constraints(a_cmd_world, v_M, limits: InterceptorLimits) -> np.ndarray:
    """Enforce the per-platform tangential/normal acceleration envelope."""
    a = np.asarray(a_cmd_world, dtype=float).copy()
    v_M = np.asarray(v_M, dtype=float)
    if limits.omnidirectional:
        n = float(np.linalg.norm(a))
        if n > limits.a_n_max:
            a *= limits.a_n_max / n
        return a
    speed = float(np.linalg.norm(v_M))
    t_hat = v_M / max(speed, 1e-6)
    a_t = float(np.dot(a, t_hat))
    a_n_vec = a - a_t * t_hat
    a_t = float(np.clip(a_t, limits.a_t_min, limits.a_t_max))
    if speed <= limits.v_min and a_t < 0.0:
        a_t = 0.0
    elif speed >= limits.v_max and a_t > 0.0:
        a_t = 0.0
    n = float(np.linalg.norm(a_n_vec))
    if n > limits.a_n_max:
        a_n_vec *= limits.a_n_max / n
    return a_t * t_hat + a_n_vec


def heading_rate_from_acceleration(v_M, a_total, omega_max: float) -> np.ndarray:
    v_M = np.asarray(v_M, dtype=float)
    vn = float(np.linalg.norm(v_M))
    if vn < 1e-3:
        return np.zeros(3)
    omega = np.cross(v_M / vn, np.asarray(a_total, dtype=float)) / max(vn, 1.0)
    n = float(np.linalg.norm(omega))
    return omega * (omega_max / n) if n > omega_max else omega


def closest_approach_distance(p_T_prev, p_T_now, p_M_prev, p_M_now) -> float:
    """Continuous CPA over one step using linear relative motion (P5.6)."""
    dr0 = np.asarray(p_T_prev) - np.asarray(p_M_prev)
    dv = (np.asarray(p_T_now) - np.asarray(p_T_prev)) - (np.asarray(p_M_now) - np.asarray(p_M_prev))
    dv2 = float(np.dot(dv, dv))
    tau = float(np.clip(-np.dot(dr0, dv) / max(dv2, 1e-9), 0.0, 1.0)) if dv2 > 1e-9 else 1.0
    return float(np.linalg.norm(dr0 + tau * dv))
