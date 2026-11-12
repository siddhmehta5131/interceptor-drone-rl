"""Crazyflie 2.x 6-DOF physics (z-up world frame).

Ported verbatim from the archived simulator
(``rl-drone-flight-simulator/interceptor-training/src/physics``, commit
``493d370``).  Behaviour is pinned by ``tests/unit/test_fixture_parity_sim.py``
against fixtures recorded from the original code (plan P1.5).  Model changes
(parameterised timestep, refitted motor and aero models, Dryden wind) are
plan phase P3 and must update those fixtures deliberately.
"""

from .constants import *  # noqa: F401,F403
from . import aero, pipeline, quaternion  # noqa: F401

__all__ = ["aero", "pipeline", "quaternion"]
