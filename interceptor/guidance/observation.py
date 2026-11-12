"""Asymmetric observation builder for the unified guidance environment.

Actor: 85 values = four 16-value frames (m=3, p=2) + three 7-value future
samples (n=3, q=5).  Critic: 16 privileged values.  Normalisation scales come
from the loaded :class:`~interceptor.guidance.platform.InterceptorLimits`, so
one implementation serves all three platforms (fix F18).
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from .platform import InterceptorLimits
from .seeker import SeekerSensorFilter

__all__ = ["AsymmetricObsBuilder", "FRAME_DIM", "FUTURE_SAMPLE_DIM", "PRIVILEGED_DIM"]


class AsymmetricObsBuilder:
    FRAME_DIM = 16
    FUTURE_SAMPLE_DIM = 7
    PRIVILEGED_DIM = 16

    def __init__(self, limits: InterceptorLimits, m: int = 3, p: int = 2,
                 n: int = 3, q: int = 5):
        self.limits = limits
        self.m, self.p, self.n, self.q = int(m), int(p), int(n), int(q)
        self.actor_dim = self.FRAME_DIM * (1 + self.m) + self.FUTURE_SAMPLE_DIM * self.n
        self.total_dim = self.actor_dim + self.PRIVILEGED_DIM
        self._history = deque(maxlen=self.m * self.p + 1)

    def reset(self) -> None:
        self._history.clear()

    def _frame(self, p_M, v_M, est_p_T, est_v_T, visible) -> np.ndarray:
        lim = self.limits
        rel_pos = np.asarray(est_p_T) - np.asarray(p_M)
        rel_vel = np.asarray(est_v_T) - np.asarray(v_M)
        rng = float(np.linalg.norm(rel_pos))
        rhat = rel_pos / max(rng, 1e-6)
        closing = float(-np.dot(rel_vel, rhat))
        r_eff = max(rng, 15.0)
        los_rate = np.cross(rel_pos, rel_vel) / r_eff ** 2
        t_go = float(np.clip(rng / max(closing, 0.1), 0.0, lim.max_time))
        return np.clip(np.concatenate([
            rel_pos / lim.r_init_max,
            rel_vel / lim.v_max,
            np.asarray(v_M) / lim.v_max,
            [rng / lim.r_init_max, closing / lim.v_max],
            los_rate,
            [t_go / lim.max_time, 1.0 if visible else 0.0],
        ]), -10.0, 10.0).astype(np.float32)

    def build(self, *, p_M, v_M, est_p_T, est_v_T, est_a_T, visible: bool,
              dt: float, time_remaining: float, optimal_accel, optimal_vel,
              true_future_pos, true_future_vel, true_acc) -> np.ndarray:
        lim = self.limits
        self._history.appendleft(self._frame(p_M, v_M, est_p_T, est_v_T, visible))
        frames = list(self._history)
        history = np.concatenate([
            frames[k * self.p] if k * self.p < len(frames) else frames[-1]
            for k in range(self.m + 1)
        ])
        future = []
        for k in range(self.n):
            ahead = float(self.q * (k + 1)) * dt
            fp, fv = SeekerSensorFilter.kalman_predict(est_p_T, est_v_T, est_a_T, ahead)
            rel_p = (fp - (np.asarray(p_M) + np.asarray(v_M) * ahead)) / lim.r_init_max
            rel_v = (fv - np.asarray(v_M)) / lim.v_max
            future.append(np.concatenate([rel_p, rel_v, [1.0 if visible else 0.0]]))
        actor = np.clip(np.concatenate([history] + future), -10.0, 10.0)
        priv = np.clip(np.concatenate([
            [time_remaining / lim.max_time],
            np.asarray(optimal_accel) / lim.a_n_max,
            np.asarray(optimal_vel) / lim.v_max,
            (np.asarray(true_future_pos) - np.asarray(p_M)) / lim.r_init_max,
            (np.asarray(true_future_vel) - np.asarray(v_M)) / lim.v_max,
            np.asarray(true_acc) / max(lim.a_target_max, 1.0),
        ]), -10.0, 10.0)
        return np.concatenate([actor, priv]).astype(np.float32)
