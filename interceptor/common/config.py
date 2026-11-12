"""Single public configuration entry point (plan P1.11).

The implementation lives in :mod:`interceptor.utils.config_loader` for
backwards compatibility; this module is the canonical import path for new code
and re-exports the whole API without duplicating loader logic.
"""

from __future__ import annotations

from ..utils.config_loader import (  # noqa: F401
    APNSpec, ConfigError, FUTURE_SOURCES, InterceptorSpec, ObservationConfig,
    ON_CAPPED_CHOICES, PREDICTORS, TrainConfig, is_range, load_config,
    parse_range,
)

__all__ = [
    "ConfigError", "ObservationConfig", "InterceptorSpec", "APNSpec",
    "TrainConfig", "load_config", "parse_range", "is_range",
    "FUTURE_SOURCES", "PREDICTORS", "ON_CAPPED_CHOICES",
]
