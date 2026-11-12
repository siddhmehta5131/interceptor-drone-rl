"""Range-dependent seeker noise, dropout and alpha-beta-gamma filtering (P1.7).

Ported verbatim from notebook v2 cell 3, with the RNG injectable so every
environment owns an explicit ``numpy.random.Generator`` (plan P1.11).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .platform import InterceptorLimits

__all__ = ["SensorNoiseConfig", "SeekerSensorFilter"]

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


class SeekerSensorFilter:
    """Noisy seeker + recursive tracking filter + constant-acceleration predict."""

    def __init__(self, cfg: SensorNoiseConfig, limits: InterceptorLimits,
                 rng: Optional[np.random.Generator] = None):
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
