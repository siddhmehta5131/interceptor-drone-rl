"""prediction/base.py -- target-state predictor interface.

A predictor extrapolates the current target state ``(pos, vel, acc)`` forward by
a horizon ``dt_ahead`` seconds.  It is deliberately tiny: the future
observation block only needs a short-horizon estimate of where the target will
be, and the APN guidance supplies the rest of the geometry.
"""

from __future__ import annotations

from typing import Dict, Tuple, Type

import numpy as np

__all__ = ["Predictor", "make_predictor", "PREDICTORS"]


class Predictor:
    """Base class for target-state predictors."""

    #: short identifier, matching ``ObsStageConfig.predictor``
    name: str = "base"

    def reset(self) -> None:
        """Clear any per-episode state.  Stateless predictors need nothing."""
        return None

    def predict(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        acc: np.ndarray,
        dt_ahead: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(predicted_pos, predicted_vel)`` at ``now + dt_ahead``."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r})"


#: Registry populated by the concrete predictor modules.
PREDICTORS: Dict[str, Type[Predictor]] = {}


def make_predictor(name: str) -> Predictor:
    """Instantiate a predictor by name (defaults to ``const_vel``)."""
    # Import lazily so importing this module has no side effects.
    from . import const_vel as _cv  # noqa: F401
    try:
        from . import kalman as _kf  # noqa: F401
    except Exception:  # pragma: no cover - optional dependency
        _kf = None
    key = (name or "const_vel").lower()
    cls = PREDICTORS.get(key, PREDICTORS["const_vel"])
    return cls()
