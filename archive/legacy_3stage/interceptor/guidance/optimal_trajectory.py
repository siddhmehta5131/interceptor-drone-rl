"""optimal_trajectory.py -- Compute the optimal intercept trajectory (critic oracle).

Given the interceptor's current state and the COMPLETE target path (all future
positions and velocities), compute an energy-efficient intercept trajectory.

This is fed to the critic as ground-truth "what a perfect policy would do" so
the value function learns faster.  The actor never sees any of it.

Method
------
For a known target path ``p_T(t)`` we:
  1. search a grid of candidate intercept times ``t_f`` for the earliest
     feasible intercept (straight-line reachability under ``v_max``/``a_max``);
  2. build a straight-line, cubic-Hermite time-scaled trajectory from the
     interceptor to the predicted intercept point that matches the boundary
     velocities ``v_int(0)`` and ``v_T(t_f)``;
  3. uniformly scale ``t_f`` upward until the acceleration/speed limits hold.

For a static target this reduces to a straight-line trapezoidal profile in
which the interceptor accelerates toward the target and decelerates to rest at
it (the plan's "bang-coast-bang or direct approach").

The oracle is deliberately closed-form and dependency-light (NumPy only) so it
can run inside the training loop without a solver.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

__all__ = ["compute_optimal_intercept"]

_EPS = 1e-9


# ---------------------------------------------------------------------------
# Target-path sampling helpers
# ---------------------------------------------------------------------------

def _sample_target(target_path: Any, t: float, order: int = 2) -> np.ndarray:
    """Return ``[pos, vel, accel]`` (each 3,) of the target at time ``t``.

    ``target_path`` may be:
      * an object with ``position_at`` / ``velocity_at`` / ``acceleration_at``
        (e.g. ``PolynomialTargetPath``);
      * a callable ``f(t) -> pos``;
      * a tuple/list ``(times, positions, velocities, accelerations)`` arrays.
    """
    if hasattr(target_path, "position_at"):
        pos = np.asarray(target_path.position_at(t), dtype=float)
        vel = (np.asarray(target_path.velocity_at(t), dtype=float)
               if hasattr(target_path, "velocity_at") else np.zeros(3))
        acc = (np.asarray(target_path.acceleration_at(t), dtype=float)
               if hasattr(target_path, "acceleration_at") else np.zeros(3))
        return np.concatenate([pos, vel, acc])
    if callable(target_path):
        pos = np.asarray(target_path(t), dtype=float)
        return np.concatenate([pos, np.zeros(3), np.zeros(3)])
    # (times, positions, velocities, accelerations) tables
    times, positions, velocities, accelerations = target_path
    pos = np.array([np.interp(t, times, positions[:, i]) for i in range(3)])
    vel = np.array([np.interp(t, times, velocities[:, i]) for i in range(3)])
    acc = np.array([np.interp(t, times, accelerations[:, i]) for i in range(3)])
    return np.concatenate([pos, vel, acc])


# ---------------------------------------------------------------------------
# 1D trapezoidal travel-time (accel -> cruise -> decel) starting at speed s0
# ---------------------------------------------------------------------------

def _trapezoid_reach_time(d: float, s0: float, v_max: float, a_max: float) -> float:
    """Minimum time to travel distance ``d`` from initial speed ``s0`` (>0 toward d).

    Uses a trapezoidal speed profile capped at ``v_max`` / ``a_max``.  Final
    speed is unconstrained (an intercept does not need to stop).
    """
    d = max(d, 0.0)
    a = max(a_max, _EPS)
    v_peak = max(v_max, s0)
    # time to accelerate from s0 to v_peak, and distance covered meanwhile
    t_acc = max(0.0, (v_peak - s0) / a)
    d_acc = s0 * t_acc + 0.5 * a * t_acc ** 2
    if d <= d_acc:
        # never reaches cruise; solve s0*t + 0.5*a*t^2 = d
        disc = s0 ** 2 + 2.0 * a * d
        return max(0.0, (-s0 + np.sqrt(max(disc, 0.0))) / a)
    d_cruise = d - d_acc
    return t_acc + d_cruise / v_peak


def _trapezoid_profile_stop(d: float, s0: float, v_max: float, a_max: float,
                            dt: float, t_f: float) -> Dict[str, np.ndarray]:
    """Straight-line profile that accelerates toward the target then stops.

    Produces ``positions``/``velocities``/``accelerations`` of length
    ``n = round(t_f/dt) + 1`` along a 1D coordinate ``s`` (metres from start).
    The acceleration phase is symmetric to the deceleration phase so final
    speed is zero.
    """
    n = max(int(round(t_f / dt)) + 1, 2)
    ts = np.linspace(0.0, t_f, n)
    s = np.zeros(n)
    v = np.zeros(n)
    acc = np.zeros(n)
    if d < _EPS:
        return {"t": ts, "s": s, "v": v, "a": acc}
    a_cap = max(a_max, _EPS)
    v_max = max(v_max, _EPS)
    # sym. accel/decel with cruise: choose peak speed so distance d is covered.
    # distance for triangular profile with peak speed vp: d = vp^2/a (accel+decel)
    vp = np.sqrt(a_cap * d)
    if vp <= v_max:
        t_acc = vp / a_cap
        t_total = 2.0 * t_acc
        cruise = False
        t_cruise = 0.0
    else:
        vp = v_max
        t_acc = vp / a_cap
        t_cruise = max(0.0, (d - vp ** 2 / a_cap) / vp)
        t_total = 2.0 * t_acc + t_cruise
        cruise = True
    if t_total < _EPS:
        t_total = _EPS
    # NOTE: we sample the actual profile over t_total; if t_total < t_f the
    # interceptor holds at rest at the target for the remainder of the horizon.
    for i, t in enumerate(ts):
        if t >= t_total:
            s[i] = d
            v[i] = 0.0
            acc[i] = 0.0
            continue
        if t < t_acc:
            s[i] = 0.5 * a_cap * t ** 2
            v[i] = a_cap * t
            a_i = a_cap
        elif cruise and t < t_acc + t_cruise:
            s[i] = 0.5 * a_cap * t_acc ** 2 + vp * (t - t_acc)
            v[i] = vp
            a_i = 0.0
        else:
            t_dec = t - (t_acc + t_cruise)
            d_before = 0.5 * a_cap * t_acc ** 2 + vp * t_cruise
            s[i] = d_before + vp * t_dec - 0.5 * a_cap * t_dec ** 2
            v[i] = max(0.0, vp - a_cap * t_dec)
            a_i = -a_cap
        acc[i] = a_i
    return {"t": ts, "s": s, "v": v, "a": acc}


# ---------------------------------------------------------------------------
# Cubic-Hermite straight-line profile matching both boundary velocities
# ---------------------------------------------------------------------------

def _hermite_profile(p0: np.ndarray, v0: np.ndarray, pf: np.ndarray,
                     vf: np.ndarray, t_f: float, dt: float) -> Dict[str, np.ndarray]:
    """Cubic Hermite position profile with given endpoint velocities."""
    n = max(int(round(t_f / dt)) + 1, 2)
    ts = np.linspace(0.0, t_f, n)
    tau = ts / max(t_f, _EPS)
    h00 = 2 * tau ** 3 - 3 * tau ** 2 + 1
    h10 = tau ** 3 - 2 * tau ** 2 + tau
    h01 = -2 * tau ** 3 + 3 * tau ** 2
    h11 = tau ** 3 - tau ** 2
    pos = (np.outer(h00, p0) + np.outer(h10 * t_f, v0)
           + np.outer(h01, pf) + np.outer(h11 * t_f, vf))
    # derivatives wrt t
    dh00 = (6 * tau ** 2 - 6 * tau) / max(t_f, _EPS)
    dh10 = (3 * tau ** 2 - 4 * tau + 1)
    dh01 = (-6 * tau ** 2 + 6 * tau) / max(t_f, _EPS)
    dh11 = (3 * tau ** 2 - 2 * tau)
    vel = (np.outer(dh00, p0) + np.outer(dh10, v0)
           + np.outer(dh01, pf) + np.outer(dh11, vf))
    d2h00 = (12 * tau - 6) / max(t_f, _EPS) ** 2
    d2h10 = (6 * tau - 4) / max(t_f, _EPS)
    d2h01 = (-12 * tau + 6) / max(t_f, _EPS) ** 2
    d2h11 = (6 * tau - 2) / max(t_f, _EPS)
    acc = (np.outer(d2h00, p0) + np.outer(d2h10, v0)
           + np.outer(d2h01, pf) + np.outer(d2h11, vf))
    return {"t": ts, "positions": pos, "velocities": vel, "accelerations": acc}


def _pack(t: np.ndarray, positions: np.ndarray, velocities: np.ndarray,
          accelerations: np.ndarray, cost: float, method: str) -> Dict[str, Any]:
    return {
        "optimal_time": float(t[-1]) if len(t) else 0.0,
        "times": t,
        "optimal_positions": positions,
        "optimal_velocities": velocities,
        "optimal_accelerations": accelerations,
        "optimal_accel": accelerations[0] if len(accelerations) else np.zeros(3),
        "optimal_vel": velocities[0] if len(velocities) else np.zeros(3),
        "optimal_cost": float(cost),
        "method": method,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_optimal_intercept(
    p_interceptor: np.ndarray,
    v_interceptor: np.ndarray,
    target_path: Any,
    v_max: float,
    a_max: float,
    dt: float,
    t_max: float = 20.0,
    is_static: Optional[bool] = None,
) -> Dict[str, Any]:
    """Compute an energy-efficient optimal intercept trajectory.

    Returns a dict with keys ``optimal_time``, ``optimal_positions`` (T, 3),
    ``optimal_velocities`` (T, 3), ``optimal_accelerations`` (T, 3),
    ``optimal_cost`` plus convenience scalars ``optimal_accel`` / ``optimal_vel``
    (the first row, i.e. what the oracle would command *now*).
    """
    p_int = np.asarray(p_interceptor, dtype=float)
    v_int = np.asarray(v_interceptor, dtype=float)
    dt = float(dt)

    if is_static is None:
        # Detect a static target by comparing the path at two times.
        try:
            s0 = _sample_target(target_path, 0.0)
            s1 = _sample_target(target_path, 1.0)
            is_static = bool(np.allclose(s0, s1, atol=1e-6))
        except Exception:
            is_static = False

    if is_static:
        return _optimal_static(p_int, v_int, target_path, v_max, a_max, dt)
    return _optimal_known_future(p_int, v_int, target_path, v_max, a_max, dt, t_max)


def _optimal_static(p_int: np.ndarray, v_int: np.ndarray, target_path: Any,
                    v_max: float, a_max: float, dt: float) -> Dict[str, Any]:
    """Straight-line trapezoidal profile to a static target."""
    p_target = _sample_target(target_path, 0.0)[:3]
    delta = p_target - p_int
    d = float(np.linalg.norm(delta))
    if d < _EPS:
        z = np.zeros((2, 3))
        return _pack(np.array([0.0, dt]), z, z.copy(), z.copy(), 0.0, "static")
    u = delta / d
    s0 = float(max(0.0, np.dot(v_int, u)))
    t_f = max(_trapezoid_reach_time(d, s0, v_max, a_max), dt)
    prof = _trapezoid_profile_stop(d, s0, v_max, a_max, dt, t_f)
    n = len(prof["t"])
    positions = p_int[None, :] + np.outer(prof["s"], u)
    velocities = np.outer(prof["v"], u)
    accelerations = np.outer(prof["a"], u)
    cost = float(np.sum(prof["a"] ** 2) * dt)
    out = _pack(prof["t"], positions, velocities, accelerations, cost, "static")
    return out


def _optimal_known_future(p_int: np.ndarray, v_int: np.ndarray, target_path: Any,
                          v_max: float, a_max: float, dt: float,
                          t_max: float) -> Dict[str, Any]:
    """Earliest feasible intercept of a known future target path."""
    # 1. Grid-search the earliest feasible intercept time.
    t_grid = np.arange(dt, t_max + dt, dt)
    t_f = None
    predicted = None
    for t in t_grid:
        st = _sample_target(target_path, float(t))
        d = float(np.linalg.norm(st[:3] - p_int))
        s0 = float(max(0.0, np.dot(v_int, (st[:3] - p_int) / max(d, _EPS))))
        t_reach = _trapezoid_reach_time(d, s0, v_max, a_max)
        if t_reach <= t + 1e-9:
            t_f = float(t)
            predicted = st
            break
    if t_f is None:
        st = _sample_target(target_path, float(t_grid[-1]))
        t_f = float(t_grid[-1])
        predicted = st

    pf = predicted[:3]
    vf = predicted[3:6]

    # 2. Build a cubic Hermite profile; scale t_f up until limits hold.
    traj = _hermite_profile(p_int, v_int, pf, vf, t_f, dt)
    for _ in range(50):
        amax = float(np.max(np.linalg.norm(traj["accelerations"], axis=1)))
        vmax = float(np.max(np.linalg.norm(traj["velocities"], axis=1)))
        over_a = amax > a_max + 1e-6
        over_v = vmax > v_max + 1e-6
        if not (over_a or over_v):
            break
        t_f *= 1.10
        # Re-sample the target at the (later) intercept time and rebuild.
        st = _sample_target(target_path, t_f)
        traj = _hermite_profile(p_int, v_int, st[:3], st[3:6], t_f, dt)

    cost = float(np.sum(np.linalg.norm(traj["accelerations"], axis=1) ** 2) * dt)
    return _pack(traj["t"], traj["positions"], traj["velocities"],
                 traj["accelerations"], cost, "hermite")
