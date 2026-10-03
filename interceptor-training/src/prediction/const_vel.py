"""prediction/const_vel.py  --  zero-jerk linear extrapolation.

``p(t + h) = p(t) + v(t) * h`` -- the simplest predictor that is exact for a
constant-velocity target (stage 5).  It ignores the position history, which
makes it the natural default when the observation history is empty.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Predictor


class ConstVelPredictor(Predictor):
    name = "const_vel"

    def predict(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        *,
        history: Optional[np.ndarray] = None,
        future_samples: int = 0,
        future_skip: int = 1,
        dt: float = 0.01,
    ):
        p = np.asarray(pos, dtype=np.float64).reshape(3)
        v = np.asarray(vel, dtype=np.float64).reshape(3)
        t = self._times(future_samples, future_skip, dt)          # (n+1,)
        pred_pos = p[None, :] + t[:, None] * v[None, :]            # (n+1, 3)
        pred_vel = np.repeat(v[None, :], pred_pos.shape[0], axis=0)
        return pred_pos, pred_vel
