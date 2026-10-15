"""prediction/  --  target-velocity predictors (plan D-9).

Two predictors ship with the project:

* ``const_vel``  -- zero-jerk linear extrapolation from the last observed
  target velocity.  Cheap, deterministic, and exactly right for the
  constant-velocity target of stage 5.
* ``linear_ridge`` -- ridge regression over the observed position history,
  fitting ``p(t) = c0 + c1 t + c2 t^2`` and extrapolating.  This tracks the
  accelerating targets of stages 6-7 far better than ``const_vel`` while
  degrading gracefully to linear extrapolation when history is short.

Both expose the same interface so ``ObservationConfig.predictor`` can select
one by name:

    pred_pos, pred_vel = predictor.predict(pos, vel, history=hist, times=F)

where ``pred_pos`` has shape ``(future_samples + 1, 3)`` and ``pred_vel`` the
same, with row ``0`` being the *current* (now) estimate.  ``future_samples + 1``
lets the observation carry ``t = 0`` explicitly.
"""

from __future__ import annotations

from .base import Predictor
from .const_vel import ConstVelPredictor
from .linear_ridge import LinearRidgePredictor

__all__ = [
    "Predictor",
    "ConstVelPredictor",
    "LinearRidgePredictor",
    "get_predictor",
    "PREDICTOR_REGISTRY",
]

#: name -> class, used by ``ObservationConfig`` validation.
PREDICTOR_REGISTRY = {
    "const_vel": ConstVelPredictor,
    "linear_ridge": LinearRidgePredictor,
}


def get_predictor(name: str, **kwargs) -> Predictor:
    """Instantiate a predictor by name; unknown names raise ``ValueError``."""
    key = str(name).strip().lower()
    if key not in PREDICTOR_REGISTRY:
        raise ValueError(
            f"unknown predictor {name!r}; expected one of "
            f"{sorted(PREDICTOR_REGISTRY)}"
        )
    return PREDICTOR_REGISTRY[key](**kwargs)
