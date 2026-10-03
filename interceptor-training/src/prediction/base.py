"""prediction/base.py  --  the predictor interface (plan D-9).

A predictor turns the *observed* target state (position, velocity, and an
optional position history) into a stack of future positions and velocities.
Row ``0`` of each output is the "now" estimate so the observation can include
an explicit ``t = 0`` sample.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class Predictor:
    """Base class for target-velocity predictors."""

    #: short identifier, matching ``ObservationConfig.predictor``
    name: str = "base"

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
        """Return ``(pred_pos, pred_vel)`` with ``future_samples + 1`` rows.

        Parameters
        ----------
        pos, vel:
            Current target position / velocity in the world frame.
        history:
            Optional ``(n, 3)`` array of *past* positions, oldest first.
        future_samples:
            Number of future steps to predict (row 0 is the current step).
        future_skip:
            Physical time between successive predicted samples, in steps.
        dt:
            Physics timestep in seconds.
        """
        raise NotImplementedError

    # -- shared helper ----------------------------------------------------
    @staticmethod
    def _times(future_samples: int, future_skip: int, dt: float) -> np.ndarray:
        """Offsets in seconds for rows ``0..future_samples`` (row 0 == now)."""
        n = int(max(0, future_samples))
        k = int(max(1, future_skip))
        return np.arange(n + 1, dtype=np.float64) * (k * float(dt))

    def reset(self) -> None:
        """Clear any per-episode state.  Stateless predictors need nothing."""
        return None

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r})"
