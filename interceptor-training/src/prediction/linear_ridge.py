"""prediction/linear_ridge.py  --  ridge regression on the position history.

Fits ``p(t) = c0 + c1 t + c2 t^2`` (with ``c2`` dropped when there is not
enough history) by ridge-regularised least squares over the observed target
positions, then extrapolates.  The regulariser is *tiny* by default
(``1e-3``): it only exists to keep the normal equations invertible when the
history is degenerate (e.g. a stationary target), never to bias a good fit.

The predictor always re-fits on the ``pos`` / ``vel`` it is handed, so it stays
correct under history truncation and target re-spawns.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Predictor


class LinearRidgePredictor(Predictor):
    name = "linear_ridge"

    def __init__(
        self,
        ridge: float = 1e-3,
        history_window: int = 16,
        degree: int = 2,
    ):
        if ridge < 0.0:
            raise ValueError("ridge must be >= 0")
        if history_window < 2:
            raise ValueError("history_window must be >= 2")
        if degree not in (1, 2):
            raise ValueError("degree must be 1 or 2")
        self.ridge = float(ridge)
        self.history_window = int(history_window)
        self.degree = int(degree)

    # ------------------------------------------------------------------ API
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
        t_future = self._times(future_samples, future_skip, dt)    # (n+1,)

        # -- assemble (times, positions) pairs ------------------------------
        times = [0.0]
        pts = [p]
        if history is not None and len(history) > 0:
            hist = np.asarray(history, dtype=np.float64).reshape(-1, 3)[-self.history_window:]
            n = hist.shape[0]
            for i in range(n):
                times.append(float(-(n - i) * dt))
                pts.append(hist[i])

        t = np.asarray(times, dtype=np.float64)[:, None]           # (m, 1)
        y = np.asarray(pts, dtype=np.float64)                      # (m, 3)

        deg = self.degree
        if t.shape[0] < deg + 1:
            deg = 1
        if t.shape[0] < 2:
            return self._const_vel(p, v, t_future)

        # -- design matrix: [1, t, t^2] (drop t^2 when under-determined) ----
        A = np.concatenate([t ** k for k in range(deg + 1)], axis=1)  # (m, deg+1)
        m, ncol = A.shape
        penalty = self.ridge * np.eye(ncol, dtype=np.float64)
        penalty[0, 0] = 0.0  # do not shrink the intercept
        try:
            coeff = np.linalg.solve(A.T @ A + penalty, A.T @ y)
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            return self._const_vel(p, v, t_future)

        # -- extrapolate, then blend in the observed velocity at t = 0 ------
        Af = np.concatenate([t_future[:, None] ** k for k in range(deg + 1)], axis=1)
        pred_pos = Af @ coeff                                        # (n+1, 3)
        deriv = np.zeros_like(Af)
        deriv[:, 0] = 0.0
        for k in range(1, ncol):
            deriv[:, k] = k * t_future ** (k - 1)
        pred_vel = deriv @ coeff

        # The fit is anchored on noisy samples; trust the measured velocity for
        # the first sample so the observation is continuous with the current
        # frame.
        pred_pos[0] = p
        pred_vel[0] = v
        return pred_pos, pred_vel

    # ------------------------------------------------------------- internals
    @staticmethod
    def _const_vel(p, v, t_future):
        pred_pos = p[None, :] + t_future[:, None] * v[None, :]
        pred_vel = np.repeat(v[None, :], pred_pos.shape[0], axis=0)
        return pred_pos, pred_vel
