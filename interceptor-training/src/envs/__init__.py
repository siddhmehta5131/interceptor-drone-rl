"""Envs package -- curriculum stage environments for the interceptor drone."""

from .stage_config import (
    STAGES,
    InterceptConfig,
    RewardConfig,
    SpawnConfig,
    StageConfig,
    make_obs_config,
)

__all__ = [
    "STAGES",
    "InterceptConfig",
    "RewardConfig",
    "SpawnConfig",
    "StageConfig",
    "make_obs_config",
]