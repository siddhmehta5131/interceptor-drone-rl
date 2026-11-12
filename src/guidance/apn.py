"""apn.py -- 3D Augmented Proportional Navigation (APN) guidance law.

The APN command is:

    a_M = N' * (V_c x omega_LOS) + (N'/2) * a_T_perp

In this codebase:
    - Interceptor = the agent we control
    - Target = the entity we want to intercept
    - All vectors are in the world (inertial) frame
    - LOS = Line of Sight = vector from interceptor to target

Implementation note
-------------------
The plan (docs/updation_plan.md, v1.0) states the command as a literal cross
product ``N' * cross(V_rel, omega_LOS)`` in one place and, in another,
derives it from the Zero Effort Miss as ``a_M = N' * ZEM_perp / t_go**2``.
Those two are only equivalent for the *perpendicular* component; the literal
cross product also injects an along-LOS (non-guidance) component and has the
wrong lateral magnitude.  Because the ZEM derivation is the mathematically
correct one (and is also what the plan's own Section 2.4 states), this module
implements the equivalent, correct form:

    a_pn  = N' * perp(V_rel) / t_go
    a_aug = (N' / 2) * perp(a_T)
    a_cmd = a_pn + a_aug

where ``perp(x) = x - (x . R_hat) R_hat``.  This is identical to
``N' * Vc * lambda_dot`` resolved into the plane perpendicular to the LOS.

All functions are pure NumPy; no other project dependencies.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

__all__ = [
    "compute_los_rate",
    "compute_closing_velocity",
    "compute_time_to_go",
    "compute_zem",
    "compute_apn_command",
]

_EPS = 1e-9


def compute_los_rate(R_vec: np.ndarray, V_rel: np.ndarray) -> np.ndarray:
    """Line-of-sight angular velocity vector ``omega_LOS = (R x V_rel) / |R|^2``.

    Parameters
    ----------
    R_vec : (3,) LOS vector ``p_target - p_interceptor``.
    V_rel : (3,) relative velocity ``v_target - v_interceptor``.

    Returns
    -------
    (3,) LOS angular velocity vector (rad/s).
    """
    R_vec = np.asarray(R_vec, dtype=float)
    V_rel = np.asarray(V_rel, dtype=float)
    R2 = float(np.dot(R_vec, R_vec))
    if R2 < _EPS:
        return np.zeros(3, dtype=float)
    return np.cross(R_vec, V_rel) / R2


def compute_closing_velocity(V_rel: np.ndarray, R_hat: np.ndarray) -> float:
    """Closing velocity ``Vc = -dot(V_rel, R_hat)`` (positive when closing)."""
    return float(-np.dot(np.asarray(V_rel, dtype=float),
                         np.asarray(R_hat, dtype=float)))


def compute_time_to_go(R: float, V_c: float, t_go_min: float = 0.1,
                       t_go_max: float = 20.0) -> float:
    """Time-to-go estimate ``t_go = clip(R / max(Vc, 0.1), t_go_min, t_go_max)``.

    The upper clamp is a numerical safeguard: when the interceptor is opening
    (``Vc <= 0``) the raw estimate ``R / 0.1`` grows to tens of seconds and the
    ZEM (which scales with ``t_go``) explodes.  Clamping keeps the guidance and
    reward bounded without changing the geometry of a closing engagement.
    """
    return float(np.clip(R / max(V_c, 0.1), t_go_min, t_go_max))


def compute_zem(V_rel: np.ndarray, a_target: np.ndarray, t_go: float,
                R_hat: np.ndarray) -> Tuple[float, np.ndarray]:
    """Zero Effort Miss.

    ``ZEM_vec = V_rel * t_go + 0.5 * a_T * t_go**2``; only the component
    perpendicular to the LOS is a *guidance* error, so it is removed.

    Returns
    -------
    (ZEM_scalar, ZEM_vec_perp)
    """
    V_rel = np.asarray(V_rel, dtype=float)
    a_target = np.asarray(a_target, dtype=float)
    R_hat = np.asarray(R_hat, dtype=float)
    ZEM_vec = V_rel * t_go + 0.5 * a_target * t_go ** 2
    ZEM_along = np.dot(ZEM_vec, R_hat) * R_hat
    ZEM_perp = ZEM_vec - ZEM_along
    return float(np.linalg.norm(ZEM_perp)), ZEM_perp


def _perp(vec: np.ndarray, R_hat: np.ndarray) -> np.ndarray:
    """Component of ``vec`` perpendicular to the unit LOS ``R_hat``."""
    return vec - np.dot(vec, R_hat) * R_hat


def compute_apn_command(
    p_interceptor: np.ndarray,
    v_interceptor: np.ndarray,
    p_target: np.ndarray,
    v_target: np.ndarray,
    a_target: np.ndarray,
    N_prime: float = 4.0,
    t_go_min: float = 0.1,
    t_go_max: float = 20.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Compute the 3D APN acceleration command.

    Returns
    -------
    a_cmd : (3,) acceleration command vector in the world frame.
    info  : dict with ``range``, ``closing_vel``, ``t_go``, ``LOS_rate``
            (scalar magnitude), ``LOS_rate_vec``, ``ZEM``, ``a_pn``,
            ``a_aug``.
    """
    p_interceptor = np.asarray(p_interceptor, dtype=float)
    v_interceptor = np.asarray(v_interceptor, dtype=float)
    p_target = np.asarray(p_target, dtype=float)
    v_target = np.asarray(v_target, dtype=float)
    a_target = np.asarray(a_target, dtype=float)

    # 1. Relative geometry
    R_vec = p_target - p_interceptor           # LOS vector (interceptor -> target)
    R = float(np.linalg.norm(R_vec))           # range (scalar)
    R_hat = R_vec / max(R, _EPS)               # LOS unit vector

    # 2. Relative velocity
    V_rel = v_target - v_interceptor           # relative velocity
    V_c = compute_closing_velocity(V_rel, R_hat)   # positive when closing

    # 3. Time-to-go estimate
    t_go = compute_time_to_go(R, V_c, t_go_min, t_go_max)

    # 4. LOS angular velocity vector
    omega_LOS = compute_los_rate(R_vec, V_rel)

    # 5. Target acceleration perpendicular to LOS
    a_T_perp = _perp(a_target, R_hat)

    # 6. APN command (ZEM-equivalent, correct: see module docstring)
    a_pn = N_prime * _perp(V_rel, R_hat) / t_go    # classical PN term
    a_aug = (N_prime / 2.0) * a_T_perp             # augmentation term
    a_cmd = a_pn + a_aug

    # 7. Zero Effort Miss (diagnostics)
    ZEM, _ = compute_zem(V_rel, a_target, t_go, R_hat)

    info: Dict[str, float] = {
        "range": R,
        "closing_vel": V_c,
        "t_go": t_go,
        "LOS_rate": float(np.linalg.norm(omega_LOS)),
        "LOS_rate_vec": omega_LOS,
        "ZEM": ZEM,
        "a_pn": a_pn,
        "a_aug": a_aug,
    }
    return a_cmd, info
