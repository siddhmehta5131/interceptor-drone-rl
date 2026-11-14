"""prediction/const_vel.py -- constant-velocity target predictor.

The simplest useful predictor: assume the target keeps its current velocity.
"""

from __future__ import annotations

import numpy as np

from .base import PREDICTORS, Predictor

__all__ = ["ConstantVelocityPredictor"]


class ConstantVelocityPredictor(Predictor):
    """Extrapolate the target with a fixed velocity (ignores acceleration)."""

    name = "const_vel"

    def predict(self, pos, vel, acc, dt_ahead: float):
        ahead = float(dt_ahead)
        return pos + vel * ahead, vel


PREDICTORS["const_vel"] = ConstantVelocityPredictor
