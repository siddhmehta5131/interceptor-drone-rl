"""prediction/kalman.py -- constant-acceleration Kalman target predictor.

Uses ``filterpy``'s ``KalmanFilter`` when available (see ``requirements.txt``);
otherwise falls back to the equivalent closed-form constant-acceleration
extrapolation.  The state is ``[x, y, z, vx, vy, vz]`` and the observed target
acceleration drives the control input, giving

``pos' = pos + vel*dt + 0.5*acc*dt^2`` and ``vel' = vel + acc*dt``.
"""

from __future__ import annotations

import numpy as np

from .base import PREDICTORS, Predictor

__all__ = ["KalmanPredictor"]

try:  # pragma: no cover - optional dependency
    from filterpy.kalman import KalmanFilter as _KF
    _HAVE_FILTERPY = True
except Exception:  # pragma: no cover
    _KF = None
    _HAVE_FILTERPY = False


class KalmanPredictor(Predictor):
    """Constant-acceleration predictor backed by a filterpy Kalman filter."""

    name = "kalman"

    def __init__(self):
        self._kf = None
        if _HAVE_FILTERPY:
            self._kf = _KF(dim_x=6, dim_z=3, dim_u=3)

    def reset(self) -> None:
        return None

    def predict(self, pos, vel, acc, dt_ahead: float):
        ahead = float(dt_ahead)
        pos = np.asarray(pos, dtype=np.float64)
        vel = np.asarray(vel, dtype=np.float64)
        acc = np.asarray(acc, dtype=np.float64)
        if self._kf is None:
            return self._analytic(pos, vel, acc, ahead)

        kf = self._kf
        I3 = np.eye(3)
        Z3 = np.zeros((3, 3))
        dt2 = 0.5 * ahead * ahead
        kf.F = np.block([[I3, I3 * ahead], [Z3, I3]])
        kf.B = np.block([[I3 * dt2], [I3 * ahead]])
        kf.u = acc
        kf.x = np.concatenate([pos, vel])
        kf.predict()
        x = np.asarray(kf.x, dtype=np.float64).ravel()
        return x[0:3].copy(), x[3:6].copy()

    @staticmethod
    def _analytic(pos, vel, acc, ahead):
        return pos + vel * ahead + 0.5 * acc * ahead * ahead, vel + acc * ahead


PREDICTORS["kalman"] = KalmanPredictor
