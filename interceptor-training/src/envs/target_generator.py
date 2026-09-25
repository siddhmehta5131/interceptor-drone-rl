"""target_generator.py  --  target / waypoint kinematics (plan §5).

Target order semantics (plan stages 2-8):

* ``none``             -- no target (stage 1).
* ``static_waypoint``  -- fixed point to fly toward; kill is disabled
                          (stage 2 success = final distance < threshold).
* ``static``           -- fixed target (stages 3-4).
* ``order_1``          -- constant velocity.
* ``order_2``          -- constant *acceleration* segments: magnitude capped
                          by ``target_accel_g_limit`` (G -> m/s^2) and
                          direction re-sampled every ~0.5-1.5 s.
* ``order_3``          -- constant *jerk* segments: magnitude capped by
                          ``target_jerk_limit`` (m/s^3), acceleration capped
                          by ``target_accel_g_limit``; re-sampled every
                          ~0.3-1.0 s.
* ``evasive``          -- like order_3 but with probability
                          ``evasive_probability`` per segment the jerk/accel
                          is chosen to flee directly away from the drone.

The target is contained inside the stage's world radius (horizontal velocity
component is reflected radially at the boundary).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .stage_config import LOOKAHEAD_HORIZON_S, SpawnConfig, StageConfig

__all__ = ["TargetGenerator"]

# Kinematic segment durations (seconds) for acceleration/jerk re-sampling.
_ACCEL_SEGMENT_RANGE = (0.5, 1.5)
_JERK_SEGMENT_RANGE = (0.3, 1.0)

# Soft speed cap: moving targets are not allowed to exceed this many m/s.
_MAX_TARGET_SPEED = 30.0


class TargetGenerator:
    """Owns the target state and advances its kinematics per stage config."""

    def __init__(self, stage: StageConfig, rng: Optional[np.random.Generator] = None):
        self.stage = stage
        self.rng = rng if rng is not None else np.random.default_rng()
        self.type = stage.target_type
        self.spawn = stage.spawn
        self.radius = float(stage.world_radius)
        self.kill_radius = float(stage.intercept.kill_radius)
        self.kills_enabled = stage.target_type not in ("none", "static_waypoint")

        self._pos = np.zeros(3, dtype=np.float64)
        self._vel = np.zeros(3, dtype=np.float64)
        self._acc = np.zeros(3, dtype=np.float64)
        self._jerk = np.zeros(3, dtype=np.float64)
        self._seg_timer = 0.0
        self._seg_dur = 1.0
        self._visible = bool(stage.target_visible)
        self._moving = self.type not in ("none", "static", "static_waypoint")

    # ------------------------------------------------------------------ API

    @property
    def pos(self) -> np.ndarray:
        return self._pos

    @property
    def vel(self) -> np.ndarray:
        return self._vel

    @property
    def visible(self) -> bool:
        return self._visible

    def reset(self, drone_pos: np.ndarray) -> Dict[str, np.ndarray]:
        """Spawn the target relative to the drone's spawn position."""
        self._seg_timer = 0.0
        if self.type in ("none",):
            self._pos = np.array(drone_pos, dtype=np.float64)
            self._vel = np.zeros(3, dtype=np.float64)
            self._acc = np.zeros(3, dtype=np.float64)
            self._jerk = np.zeros(3, dtype=np.float64)
            return self.state_dict()

        pos = self._sample_spawn_rel(drone_pos)
        self._pos = pos
        self._acc = np.zeros(3, dtype=np.float64)
        self._jerk = np.zeros(3, dtype=np.float64)

        if self._moving:
            d = self._sample_cone_dir(45.0 if self.spawn.cone_half_angle_deg <= 0 else self.spawn.cone_half_angle_deg)
            lo, hi = self.spawn.speed_range
            speed = float(self.rng.uniform(lo, hi))
            self._vel = d * speed
        else:
            self._vel = np.zeros(3, dtype=np.float64)
        return self.state_dict()

    def step(self, dt: float, drone_pos: np.ndarray) -> Dict[str, np.ndarray]:
        """Advance the target by ``dt`` seconds; return the state dict."""
        if not self._moving:
            return self.state_dict()

        self._seg_timer += dt
        if self._seg_timer >= self._seg_dur:
            self._seg_timer = 0.0
            self._resample_maneuver(drone_pos)

        if self.type in ("order_2", "order_3", "evasive"):
            self._vel = self._vel + self._acc * dt
        self._pos = self._pos + self._vel * dt
        if self.type in ("order_3", "evasive"):
            self._acc = self._acc + self._jerk * dt
            a_max = self.spawn.target_accel_g_limit * 9.81
            if a_max > 0.0:
                an = float(np.linalg.norm(self._acc))
                if an > a_max:
                    self._acc = self._acc * (a_max / an)

        self._contain()
        return self.state_dict()

    def predicted_pos(self) -> np.ndarray:
        """Look-ahead predicted position (plan §8 lookahead ablation).

        Second-order prediction (v*T + 0.5*a*T^2) plus the jerk term
        (1/6*j*T^3) for order_3 / evasive segments.
        """
        if not self.stage.intercept.lookahead_enabled:
            return self._pos.copy()
        if not self._moving:
            return self._pos.copy()
        T = LOOKAHEAD_HORIZON_S
        return self._pos + self._vel * T + 0.5 * self._acc * T * T + (1.0 / 6.0) * self._jerk * T * T * T

    def distance_to(self, drone_pos: np.ndarray) -> float:
        return float(np.linalg.norm(drone_pos - self._pos))

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {
            "pos": self._pos.copy(),
            "vel": self._vel.copy(),
            "acc": self._acc.copy(),
            "visible": bool(self._visible),
            "kills_enabled": bool(self.kills_enabled),
        }

    # -------------------------------------------------------------- internals

    def _sample_spawn_rel(self, drone_pos: np.ndarray) -> np.ndarray:
        sp: SpawnConfig = self.spawn
        if self.type == "static_waypoint":
            d_range = sp.waypoint_distance_range
        else:
            d_range = sp.target_distance_range
        cone_deg = sp.cone_half_angle_deg if sp.cone_half_angle_deg > 0 else 90.0

        az_lo, az_hi = sp.altitude_range
        for _ in range(200):
            dirn = self._sample_cone_dir(cone_deg)
            dist = float(self.rng.uniform(*d_range))
            pos = drone_pos + dirn * dist
            # vertical constraint
            if az_lo <= pos[2] <= az_hi:
                if sp.lateral_offset_max > 0.0:
                    pos[1] += float(self.rng.uniform(-sp.lateral_offset_max, sp.lateral_offset_max))
                return pos
        # fallback: project up to the altitude band
        pos = drone_pos + dirn * dist
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
        # fallback: go straight forward
        return np.array([1.0, 0.0, 0.0])

    def _resample_maneuver(self, drone_pos: np.ndarray) -> None:
        sp: SpawnConfig = self.spawn
        a_max = sp.target_accel_g_limit * 9.81

        if self.type == "order_2":
            # choose a new acceleration segment
            self._acc = self._sample_accel(a_max)
            self._seg_dur = float(self.rng.uniform(*_ACCEL_SEGMENT_RANGE))
        elif self.type in ("order_3", "evasive"):
            j_max = sp.target_jerk_limit
            j = self.rng.standard_normal(3)
            j /= np.linalg.norm(j) + 1e-12
            j = j * j_max
            if self.type == "evasive" and self.rng.uniform() < sp.evasive_probability:
                # flee directly away from the drone + lateral randomness
                flee = self._pos - drone_pos
                fn = np.linalg.norm(flee)
                if fn > 1e-6:
                    flee = flee / fn
                lateral = self.rng.standard_normal(3)
                lateral -= lateral @ flee * flee
                ln = np.linalg.norm(lateral)
                if ln > 1e-6:
                    lateral = lateral / ln
                j = (flee + 0.6 * lateral) * j_max
            self._jerk = j
            self._seg_dur = float(self.rng.uniform(*_JERK_SEGMENT_RANGE))
            if self.type == "evasive" and a_max > 0.0:
                # evasive targets keep modest speeds; re-clamp current acc
                an = float(np.linalg.norm(self._acc))
                if an > a_max * 0.75:
                    self._acc = self._acc * (a_max * 0.75 / an)

    def _sample_accel(self, a_max: float) -> np.ndarray:
        if a_max <= 0.0:
            return np.zeros(3)
        d = self.rng.standard_normal(3)
        d /= np.linalg.norm(d) + 1e-12
        return d * a_max

    def _contain(self) -> None:
        """Keep the target within the world radius (reflect radial velocity).

        When the horizontal radius exceeds the world radius the target is
        placed EXACTLY on the boundary (``pos *= radius/r``) and the outward
        radial velocity component is reflected.
        """
        x, y = self._pos[0], self._pos[1]
        r_xy = float(np.hypot(x, y))
        if r_xy > self.radius:
            # reflect the horizontal velocity about the radial direction
            n = np.array([x, y], dtype=np.float64) / (r_xy + 1e-12)
            vh = np.array([self._vel[0], self._vel[1]], dtype=np.float64)
            vn = float(vh @ n)
            if vn > 0.0:
                vh = vh - 2.0 * vn * n
            scale = self.radius / r_xy
            self._pos[0] = x * scale
            self._pos[1] = y * scale
            self._vel[0], self._vel[1] = vh
        # soft speed cap (keep the target catchable)
        vn = float(np.linalg.norm(self._vel))
        if vn > _MAX_TARGET_SPEED:
            self._vel = self._vel * (_MAX_TARGET_SPEED / vn)