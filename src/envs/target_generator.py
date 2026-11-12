"""target_generator.py -- target kinematics for the 3-stage curriculum.

Adapted from the pre-pivot codebase (plan §5.2).  The highly-reusable
:class:`PolynomialTargetPath` (closed-form, monotone, drone-independent) is
kept intact; the drone coupling in the reactive (evasive) path is replaced by
an explicit *interceptor* reference so the target can flee the thing chasing
it.  Target types:

* ``static``            -- fixed target, zero velocity (stage 1).
* ``order_1``           -- linear polynomial path (stage 2).
* ``order_2``           -- quadratic polynomial path (stage 2).
* ``order_3``           -- cubic polynomial path (stage 2).
* ``evasive``           -- reactive re-sampled manoeuvres with a flee bias
                           away from the interceptor (stage 3).

Polynomial paths are closed-form and *independent of the interceptor* (so the
target is predictable from position + velocity alone).  The path is
``pos(t) = start + delta * p(t / T)`` with ``p(0)=0``, ``p(1)=1`` and ``p`` a
monotone shape polynomial:

    order 1:  p(tau) = tau
    order 2:  p(tau) = a*tau^2 + (1-a)*tau
    order 3:  p(tau) = a*tau^3 + b*tau^2 + (1-a-b)*tau

``delta`` is scaled down uniformly if the peak path speed would exceed
``spawn.path_speed_cap``.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .stage_config import SpawnConfig, StageConfig

__all__ = ["TargetGenerator", "PolynomialTargetPath", "POLYNOMIAL_TARGET_TYPES"]

# Kinematic segment durations (seconds) for acceleration/jerk re-sampling.
_ACCEL_SEGMENT_RANGE = (0.5, 1.5)
_JERK_SEGMENT_RANGE = (0.3, 1.0)

# Soft speed cap for the REACTIVE target type (stage 3).
_MAX_TARGET_SPEED = 30.0

# Closed-form polynomial target types (stage 2).
POLYNOMIAL_TARGET_TYPES = ("order_1", "order_2", "order_3")

# Peak-speed integration resolution for the polynomial speed cap.
_PEAK_SAMPLES = 201

# Edge of the world the polynomial endpoint is clamped to (fraction of radius).
_END_BOUND_FRAC = 0.9


class PolynomialTargetPath:
    """Closed-form target trajectory for polynomial targets.

    Parameters
    ----------
    start, end:
        World-frame positions.  ``start`` is where the target spawned;
        ``end`` is sampled at reset and clamped inside the world bounds.
    duration:
        Path duration ``T`` in seconds (the episode length).
    order:
        1, 2 or 3 -- the degree of the shape polynomial.
    rng:
        Source for the randomised shape coefficients.
    speed_cap:
        Hard cap in m/s.  ``delta`` is scaled down uniformly so that
        ``max |dp/dtau| * |delta| / T <= speed_cap``.
    shape:
        ``"random"`` samples the coefficients, ``"linear"`` forces
        ``p(tau) = tau`` regardless of ``order``.
    """

    def __init__(
        self,
        start: np.ndarray,
        end: np.ndarray,
        duration: float,
        order: int,
        rng: Optional[np.random.Generator] = None,
        speed_cap: float = 8.0,
        shape: str = "random",
    ):
        rng = rng if rng is not None else np.random.default_rng()
        self.order = int(order) if order in (1, 2, 3) else 1
        self.T = float(max(1e-6, duration))
        self.shape = shape

        start = np.asarray(start, dtype=np.float64).reshape(3)
        end = np.asarray(end, dtype=np.float64).reshape(3)
        self.start = start.copy()
        self.end = end.copy()

        self.a = 0.0
        self.b = 0.0
        if shape == "random" and self.order >= 2:
            self._sample_shape(rng)

        delta = self.end - self.start
        if speed_cap > 0.0:
            peak = self._peak_abs_pprime()
            peak_speed = peak * float(np.linalg.norm(delta)) / self.T
            if peak_speed > speed_cap and peak_speed > 0.0:
                delta = delta * (speed_cap / peak_speed)
        self.delta = delta
        self.effective_end = self.start + self.delta

    # ------------------------------------------------------------------ shape
    def _sample_shape(self, rng: np.random.Generator) -> None:
        tau = np.linspace(0.0, 1.0, _PEAK_SAMPLES)
        if self.order == 2:
            for _ in range(64):
                a = float(rng.uniform(-0.8, 0.8))
                vals = a * tau * tau + (1.0 - a) * tau
                if np.all(vals >= -1e-9) and np.all(np.diff(vals) >= -1e-9):
                    self.a = a
                    return
            self.a = 0.0
            return
        for _ in range(64):
            a = float(rng.uniform(-0.8, 0.8))
            b = float(rng.uniform(-1.0, 1.0))
            vals = a * tau ** 3 + b * tau * tau + (1.0 - a - b) * tau
            if np.all(vals >= -1e-9) and np.all(np.diff(vals) >= -1e-9):
                self.a = a
                self.b = b
                return
        self.a = 0.0
        self.b = 0.0

    def _p(self, tau: np.ndarray) -> np.ndarray:
        tau = np.asarray(tau, dtype=np.float64)
        if self.order == 1 or self.shape == "linear":
            return tau
        if self.order == 2:
            return self.a * tau * tau + (1.0 - self.a) * tau
        return self.a * tau ** 3 + self.b * tau * tau + (1.0 - self.a - self.b) * tau

    def _pprime(self, tau: np.ndarray) -> np.ndarray:
        tau = np.asarray(tau, dtype=np.float64)
        if self.order == 1 or self.shape == "linear":
            return np.ones_like(tau)
        if self.order == 2:
            return 2.0 * self.a * tau + (1.0 - self.a)
        return 3.0 * self.a * tau * tau + 2.0 * self.b * tau + (1.0 - self.a - self.b)

    def _psecond(self, tau: np.ndarray) -> np.ndarray:
        tau = np.asarray(tau, dtype=np.float64)
        if self.order == 1 or self.shape == "linear":
            return np.zeros_like(tau)
        if self.order == 2:
            return np.full_like(tau, 2.0 * self.a)
        return 6.0 * self.a * tau + 2.0 * self.b

    def _peak_abs_pprime(self) -> float:
        tau = np.linspace(0.0, 1.0, _PEAK_SAMPLES)
        return float(np.max(np.abs(self._pprime(tau))))

    # ----------------------------------------------------------------- access
    def _tau(self, t: float) -> float:
        return float(np.clip(t / self.T, 0.0, 1.0))

    def position_at(self, t: float) -> np.ndarray:
        tau = self._tau(t)
        return self.start + self.delta * float(self._p(tau))

    def velocity_at(self, t: float) -> np.ndarray:
        tau = self._tau(t)
        return self.delta * (float(self._pprime(tau)) / self.T)

    def acceleration_at(self, t: float) -> np.ndarray:
        tau = self._tau(t)
        return self.delta * (float(self._psecond(tau)) / (self.T * self.T))

    def jerk_at(self, t: float) -> np.ndarray:
        if self.order != 3 or self.shape == "linear":
            return np.zeros(3, dtype=np.float64)
        return self.delta * (6.0 * self.a / (self.T ** 3) * np.ones(3))

    @property
    def peak_speed(self) -> float:
        return self._peak_abs_pprime() * float(np.linalg.norm(self.delta)) / self.T


class TargetGenerator:
    """Owns the target state and advances its kinematics per stage config."""

    def __init__(self, stage: StageConfig, rng: Optional[np.random.Generator] = None):
        self.stage = stage
        self.rng = rng if rng is not None else np.random.default_rng()
        self.type = stage.target_type
        self.spawn: SpawnConfig = stage.spawn
        self.radius = float(stage.world_radius)
        self.kill_radius = float(stage.intercept.kill_radius)
        # All 3-stage target types can be killed (static included).
        self.kills_enabled = True

        self._pos = np.zeros(3, dtype=np.float64)
        self._vel = np.zeros(3, dtype=np.float64)
        self._acc = np.zeros(3, dtype=np.float64)
        self._jerk = np.zeros(3, dtype=np.float64)
        self._seg_timer = 0.0
        self._seg_dur = 1.0
        self._visible = bool(stage.target_visible)

        self._path: Optional[PolynomialTargetPath] = None
        self._elapsed = 0.0
        self._moving = self.type != "static"
        # Cached (pos, vel, acc) of the true future used by the oracle/predictor.
        self._reactive = self.type == "evasive"

    # ------------------------------------------------------------------ API
    @property
    def pos(self) -> np.ndarray:
        return self._pos

    @property
    def vel(self) -> np.ndarray:
        return self._vel

    @property
    def acc(self) -> np.ndarray:
        return self._acc

    @property
    def jerk(self) -> np.ndarray:
        return self._jerk

    @property
    def visible(self) -> bool:
        return self._visible

    @property
    def path(self) -> Optional[PolynomialTargetPath]:
        return self._path

    @property
    def polynomial(self) -> bool:
        return self.type in POLYNOMIAL_TARGET_TYPES

    @property
    def elapsed(self) -> float:
        return float(self._elapsed)

    def state_at(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """True target ``(pos, vel, acc)`` at absolute time ``t`` along the path.

        For polynomial targets this is exact (the oracle uses it).  For
        reactive targets only the current segment is known, so ``t<=elapsed``
        returns the current state and future extrapolation is linear.
        """
        if self._path is not None:
            return (self._path.position_at(t), self._path.velocity_at(t),
                    self._path.acceleration_at(t))
        dt = max(float(t) - self._elapsed, 0.0)
        pos = self._pos + self._vel * dt + 0.5 * self._acc * dt * dt
        vel = self._vel + self._acc * dt
        return pos, vel, self._acc.copy()

    def reset(self, interceptor_pos: np.ndarray,
              episode_length_s: Optional[float] = None) -> Dict[str, np.ndarray]:
        """Spawn the target relative to the interceptor's spawn position."""
        self._seg_timer = 0.0
        self._elapsed = 0.0
        self._path = None

        pos = self._sample_spawn_rel(interceptor_pos)
        self._pos = pos
        self._acc = np.zeros(3, dtype=np.float64)
        self._jerk = np.zeros(3, dtype=np.float64)

        if self.polynomial:
            duration = float(episode_length_s) if episode_length_s else 20.0
            order = {"order_1": 1, "order_2": 2, "order_3": 3}[self.type]
            cap = float(self._mean_of(self.spawn.path_speed_cap))
            shape = self.spawn.path_shape or "random"
            self._path = PolynomialTargetPath(
                start=pos,
                end=self._sample_path_end(interceptor_pos, duration, cap),
                duration=duration, order=order, rng=self.rng,
                speed_cap=cap, shape=shape,
            )
            self._pos = self._path.position_at(0.0)
            self._vel = self._path.velocity_at(0.0)
            self._acc = self._path.acceleration_at(0.0)
            self._jerk = self._path.jerk_at(0.0)
            return self.state_dict()

        if self._moving:
            cone = self.spawn.cone_half_angle_deg
            d = self._sample_cone_dir(45.0 if cone <= 0 else cone)
            lo, hi = self.spawn.target_speed_range
            self._vel = d * float(self.rng.uniform(lo, hi))
            # Seed a small initial accel for reactive targets.
            if self._reactive:
                a_lo, a_hi = self.spawn.target_accel_range
                if a_hi > 0.0:
                    ad = self.rng.standard_normal(3)
                    ad /= np.linalg.norm(ad) + 1e-12
                    self._acc = ad * float(self.rng.uniform(a_lo, a_hi))
        else:
            self._vel = np.zeros(3, dtype=np.float64)
        return self.state_dict()

    def step(self, dt: float, interceptor_pos: np.ndarray) -> Dict[str, np.ndarray]:
        """Advance the target by ``dt`` seconds; return the state dict."""
        if not self._moving:
            return self.state_dict()

        if self._path is not None:
            self._elapsed += dt
            self._pos = self._path.position_at(self._elapsed)
            self._vel = self._path.velocity_at(self._elapsed)
            self._acc = self._path.acceleration_at(self._elapsed)
            self._jerk = self._path.jerk_at(self._elapsed)
            return self.state_dict()

        # ---- reactive (evasive) target: re-sampled manoeuvres ---------------
        self._elapsed += dt
        self._seg_timer += dt
        if self._seg_timer >= self._seg_dur:
            self._seg_timer = 0.0
            self._resample_maneuver(interceptor_pos)

        self._vel = self._vel + self._acc * dt
        self._pos = self._pos + self._vel * dt
        self._acc = self._acc + self._jerk * dt
        a_max = self.spawn.target_accel_g_limit * 9.81
        if a_max > 0.0:
            an = float(np.linalg.norm(self._acc))
            if an > a_max:
                self._acc = self._acc * (a_max / an)

        self._contain()
        return self.state_dict()

    def distance_to(self, interceptor_pos: np.ndarray) -> float:
        return float(np.linalg.norm(interceptor_pos - self._pos))

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {
            "pos": self._pos.copy(),
            "vel": self._vel.copy(),
            "acc": self._acc.copy(),
            "jerk": self._jerk.copy(),
            "visible": bool(self._visible),
            "kills_enabled": bool(self.kills_enabled),
            "polynomial": bool(self._path is not None),
            "elapsed": float(self._elapsed),
        }

    # -------------------------------------------------------------- internals
    @staticmethod
    def _mean_of(value) -> float:
        """Midpoint of a ``(lo, hi)`` range (or the value itself)."""
        if value is None:
            return 0.0
        if isinstance(value, dict):
            return 0.5 * (float(value["min"]) + float(value["max"]))
        try:
            lo, hi = float(value[0]), float(value[1])
            return 0.5 * (lo + hi)
        except (TypeError, IndexError, ValueError):
            return float(value)

    def _sample_path_end(self, interceptor_pos: np.ndarray, duration: float,
                         speed_cap: float) -> np.ndarray:
        sp = self.spawn
        lo, hi = sp.path_end_distance
        dist = float(self.rng.uniform(lo, hi))
        cone = sp.cone_half_angle_deg
        d = self._sample_cone_dir(45.0 if cone <= 0 else cone)
        end = np.asarray(interceptor_pos, dtype=np.float64) + d * dist

        bound = self.radius * _END_BOUND_FRAC
        end[0] = float(np.clip(end[0], -bound, bound))
        end[1] = float(np.clip(end[1], -bound, bound))
        az_lo, az_hi = sp.altitude_range
        end[2] = float(np.clip(end[2], az_lo, az_hi))
        return end

    def _sample_spawn_rel(self, interceptor_pos: np.ndarray) -> np.ndarray:
        sp = self.spawn
        d_range = sp.target_distance_range
        cone_deg = sp.cone_half_angle_deg if sp.cone_half_angle_deg > 0 else 90.0

        az_lo, az_hi = sp.altitude_range
        dirn = np.array([1.0, 0.0, 0.0])
        dist = float(self.rng.uniform(*d_range))
        for _ in range(200):
            dirn = self._sample_cone_dir(cone_deg)
            dist = float(self.rng.uniform(*d_range))
            pos = np.asarray(interceptor_pos, dtype=np.float64) + dirn * dist
            if az_lo <= pos[2] <= az_hi:
                if sp.lateral_offset_max > 0.0:
                    pos[1] += float(self.rng.uniform(-sp.lateral_offset_max,
                                                     sp.lateral_offset_max))
                return pos
        pos = np.asarray(interceptor_pos, dtype=np.float64) + dirn * dist
        pos[2] = float(np.clip(pos[2], az_lo, az_hi))
        return pos

    def _sample_cone_dir(self, cone_deg: float) -> np.ndarray:
        """Uniform direction inside a cone around world +x (forward)."""
        cos_min = float(np.cos(np.deg2rad(cone_deg)))
        for _ in range(500):
            d = self.rng.standard_normal(3)
            d /= np.linalg.norm(d) + 1e-12
            if d[0] >= cos_min:
                return d
        return np.array([1.0, 0.0, 0.0])

    def _resample_maneuver(self, interceptor_pos: np.ndarray) -> None:
        sp = self.spawn
        j_max = sp.target_jerk_limit

        j = self.rng.standard_normal(3)
        j /= np.linalg.norm(j) + 1e-12
        j = j * j_max
        if self.rng.uniform() < sp.evasive_probability:
            # flee directly away from the interceptor + lateral randomness
            flee = self._pos - np.asarray(interceptor_pos, dtype=np.float64)
            fn = float(np.linalg.norm(flee))
            if fn > 1e-6:
                flee = flee / fn
            lateral = self.rng.standard_normal(3)
            lateral -= lateral @ flee * flee
            ln = float(np.linalg.norm(lateral))
            if ln > 1e-6:
                lateral = lateral / ln
            j = (flee + 0.6 * lateral) * j_max
        self._jerk = j
        self._seg_dur = float(self.rng.uniform(*_JERK_SEGMENT_RANGE))

        a_max = sp.target_accel_g_limit * 9.81
        if a_max > 0.0:
            an = float(np.linalg.norm(self._acc))
            if an > a_max * 0.75:
                self._acc = self._acc * (a_max * 0.75 / an)

    def _contain(self) -> None:
        """Keep the reactive target inside the world radius (radial reflection)."""
        x, y = self._pos[0], self._pos[1]
        r_xy = float(np.hypot(x, y))
        if r_xy > self.radius:
            n = np.array([x, y], dtype=np.float64) / (r_xy + 1e-12)
            vh = np.array([self._vel[0], self._vel[1]], dtype=np.float64)
            vn = float(vh @ n)
            if vn > 0.0:
                vh = vh - 2.0 * vn * n
            scale = self.radius / r_xy
            self._pos[0] = x * scale
            self._pos[1] = y * scale
            self._vel[0], self._vel[1] = vh
        vn = float(np.linalg.norm(self._vel))
        if vn > _MAX_TARGET_SPEED:
            self._vel = self._vel * (_MAX_TARGET_SPEED / vn)
