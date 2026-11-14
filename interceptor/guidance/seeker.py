"""Range-dependent seeker noise, dropout and alpha-beta-gamma filtering (P1.7).

Ported verbatim from notebook v2 cell 3, with the RNG injectable so every
environment owns an explicit ``numpy.random.Generator`` (plan P1.11).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ..common.sources import METADATA_KEYS, load_yaml
from .platform import InterceptorLimits
from .tracker import KalmanTracker

__all__ = ["SensorNoiseConfig", "SeekerSensorFilter", "load_sensor_noise_config",
           "SEEKER_YAML_FIELDS"]

SEEKER_YAML = Path(__file__).resolve().parents[2] / "configs" / "seeker.yaml"

#: ``configs/seeker.yaml`` key -> ``SensorNoiseConfig`` field (BUGS #25).
SEEKER_YAML_FIELDS = {
    "enabled": "enabled",
    "position_sigma_base": "pos_sigma_base_m",
    "position_sigma_angular": "pos_sigma_angular_rad",
    "velocity_sigma_fraction": "vel_sigma_frac",
    "acceleration_sigma_fraction": "acc_sigma_frac",
    "dropout_probability": "dropout_prob",
    "ema_alpha_position": "ema_alpha_pos",
    "ema_alpha_velocity": "ema_alpha_vel",
    "ema_alpha_acceleration": "ema_alpha_acc",
}

#: The original notebook runs in NED; the consolidated world is z-up.  Sensor
#: noise is isotropic, so mapping each Gaussian draw by the same proper
#: rotation keeps the statistics unchanged while making fixture replay exact.
_FRAME_SIGN = np.array([1.0, -1.0, -1.0])


@dataclass
class SensorNoiseConfig:
    """Seeker measurement error model."""

    enabled: bool = True
    pos_sigma_base_m: float = 0.5
    pos_sigma_angular_rad: float = 0.0025
    vel_sigma_frac: float = 0.03
    acc_sigma_frac: float = 0.18
    dropout_prob: float = 0.04
    ema_alpha_pos: float = 0.65
    ema_alpha_vel: float = 0.55
    ema_alpha_acc: float = 0.35


@lru_cache(maxsize=8)
def _load_cached(path: str) -> SensorNoiseConfig:
    raw = load_yaml(path)
    unknown = sorted(k for k in raw if k not in SEEKER_YAML_FIELDS and k not in METADATA_KEYS)
    if unknown:
        # A documented key the code does not implement would silently do nothing.
        raise KeyError(f"{path}: keys not used by SensorNoiseConfig: {unknown}")
    kwargs = {}
    for key, field in SEEKER_YAML_FIELDS.items():
        if key not in raw:
            raise KeyError(f"{path}: missing key {key!r}")
        leaf = raw[key]
        kwargs[field] = leaf["value"] if isinstance(leaf, dict) else leaf
    kwargs["enabled"] = bool(kwargs["enabled"])
    for field in kwargs:
        if field != "enabled":
            kwargs[field] = float(kwargs[field])
    return SensorNoiseConfig(**kwargs)


def load_sensor_noise_config(path: "str | Path | None" = None) -> SensorNoiseConfig:
    """Build the seeker noise model from ``configs/seeker.yaml`` (the source of record)."""
    return replace(_load_cached(str(Path(path) if path else SEEKER_YAML)))


class SeekerSensorFilter:
    """Noisy seeker + recursive tracking filter + constant-acceleration predict."""

    def __init__(self, cfg: SensorNoiseConfig, limits: InterceptorLimits,
                 rng: Optional[np.random.Generator] = None,
                 filter_type: str = "abg", tau_man: float = 1.0):
        if filter_type not in ("abg", "kalman"):
            raise ValueError(f"filter_type must be 'abg' or 'kalman', got {filter_type!r}")
        self.filter_type = filter_type
        self.kf = KalmanTracker(limits.a_target_max, tau_man) if filter_type == "kalman" else None
        self.cfg = cfg
        self.limits = limits
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.est_pos = np.zeros(3)
        self.est_vel = np.zeros(3)
        self.est_acc = np.zeros(3)
        self.raw_pos = np.zeros(3)
        self.raw_vel = np.zeros(3)
        self.raw_acc = np.zeros(3)
        self.visible = True
        self._initialized = False

    def reset(self, true_pos: np.ndarray, true_vel: np.ndarray, true_acc: np.ndarray) -> None:
        self.est_pos = np.asarray(true_pos, dtype=float).copy()
        self.est_vel = np.asarray(true_vel, dtype=float).copy()
        self.est_acc = np.asarray(true_acc, dtype=float).copy()
        self.raw_pos = self.est_pos.copy()
        self.raw_vel = self.est_vel.copy()
        self.raw_acc = self.est_acc.copy()
        self.visible = True
        self._initialized = True
        if self.kf is not None:
            self.kf.reset(true_pos, true_vel, true_acc, self._sigmas(0.0, 1.0))

    def _sigmas(self, dist: float, noise_scale: float) -> Tuple[float, float, float]:
        """EQ-SEEK-1/2 standard deviations (position, velocity, acceleration)."""
        return (noise_scale * (self.cfg.pos_sigma_base_m + self.cfg.pos_sigma_angular_rad * dist),
                noise_scale * self.cfg.vel_sigma_frac * self.limits.v_target_nominal,
                noise_scale * self.cfg.acc_sigma_frac * self.limits.a_target_max)

    def observe(self, interceptor_pos: np.ndarray, true_pos: np.ndarray,
                true_vel: np.ndarray, true_acc: np.ndarray, dt: float,
                noise_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        if not self._initialized:
            self.reset(true_pos, true_vel, true_acc)

        if not self.cfg.enabled or noise_scale <= 1e-6:
            self.est_pos = np.asarray(true_pos, dtype=float).copy()
            self.est_vel = np.asarray(true_vel, dtype=float).copy()
            self.est_acc = np.asarray(true_acc, dtype=float).copy()
            self.raw_pos = self.est_pos.copy()
            self.raw_vel = self.est_vel.copy()
            self.raw_acc = self.est_acc.copy()
            self.visible = True
            return self.est_pos.copy(), self.est_vel.copy(), self.est_acc.copy(), True

        if self.rng.random() < self.cfg.dropout_prob * noise_scale:
            self.visible = False
            if self.kf is not None:
                self.kf.predict(dt)
                self.est_pos, self.est_vel, self.est_acc = self.kf.pos, self.kf.vel, self.kf.acc
                return self.est_pos.copy(), self.est_vel.copy(), self.est_acc.copy(), False
            self.est_pos = self.est_pos + self.est_vel * dt + 0.5 * self.est_acc * dt ** 2
            self.est_vel = self.est_vel + self.est_acc * dt
            return self.est_pos.copy(), self.est_vel.copy(), self.est_acc.copy(), False

        self.visible = True
        dist = float(np.linalg.norm(np.asarray(true_pos) - np.asarray(interceptor_pos)))
        sigma_p = noise_scale * (self.cfg.pos_sigma_base_m + self.cfg.pos_sigma_angular_rad * dist)
        sigma_v = noise_scale * self.cfg.vel_sigma_frac * self.limits.v_target_nominal
        sigma_a = noise_scale * self.cfg.acc_sigma_frac * self.limits.a_target_max

        self.raw_pos = np.asarray(true_pos) + _FRAME_SIGN * self.rng.normal(0.0, sigma_p, size=3)
        self.raw_vel = np.asarray(true_vel) + _FRAME_SIGN * self.rng.normal(0.0, sigma_v, size=3)
        self.raw_acc = np.asarray(true_acc) + _FRAME_SIGN * self.rng.normal(0.0, sigma_a, size=3)

        if self.kf is not None:
            self.kf.predict(dt)
            self.kf.update(self.raw_pos, self.raw_vel, self.raw_acc, (sigma_p, sigma_v, sigma_a))
            self.est_pos, self.est_vel, self.est_acc = self.kf.pos, self.kf.vel, self.kf.acc
            return self.est_pos.copy(), self.est_vel.copy(), self.est_acc.copy(), True

        pred_pos = self.est_pos + self.est_vel * dt + 0.5 * self.est_acc * dt ** 2
        pred_vel = self.est_vel + self.est_acc * dt
        self.est_pos = self.cfg.ema_alpha_pos * self.raw_pos + (1.0 - self.cfg.ema_alpha_pos) * pred_pos
        self.est_vel = self.cfg.ema_alpha_vel * self.raw_vel + (1.0 - self.cfg.ema_alpha_vel) * pred_vel
        self.est_acc = self.cfg.ema_alpha_acc * self.raw_acc + (1.0 - self.cfg.ema_alpha_acc) * self.est_acc
        return self.est_pos.copy(), self.est_vel.copy(), self.est_acc.copy(), True

    @staticmethod
    def kalman_predict(pos: np.ndarray, vel: np.ndarray, acc: np.ndarray,
                       dt_ahead: float) -> Tuple[np.ndarray, np.ndarray]:
        ahead = float(dt_ahead)
        return (np.asarray(pos) + np.asarray(vel) * ahead + 0.5 * np.asarray(acc) * ahead ** 2,
                np.asarray(vel) + np.asarray(acc) * ahead)
