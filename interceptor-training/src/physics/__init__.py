"""Physics engine for the interceptor training pipeline.

Ported from hover_env.py (scalar, single-agent) with optional
domain-randomization / wind / ground-effect extensions (off by default).
"""

from .constants import *  # noqa: F401,F403
from . import aero, pipeline, quaternion  # noqa: F401

__all__ = ["aero", "pipeline", "quaternion"]