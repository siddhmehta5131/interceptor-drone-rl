"""Envs package -- curriculum stage environments for the interceptor drone."""

from .stage_config import (
    FACING_CUTOFF_M,
    STAGES,
    TARGET_STAGE_Z_HI,
    InterceptConfig,
    ObsStageConfig,
    RewardConfig,
    SpawnConfig,
    StageConfig,
    make_obs_config,
)

from .obs_builder import (
    FUTURE_SAMPLE_DIM,
    PRIVILEGED_FIELDS,
    STAGE1_FRAME_DIM,
    TARGET_FRAME_DIM,
    observation_dim,
    privileged_dim,
)

__all__ = [
    "FACING_CUTOFF_M",
    "FUTURE_SAMPLE_DIM",
    "PRIVILEGED_FIELDS",
    "STAGE1_FRAME_DIM",
    "STAGES",
    "TARGET_FRAME_DIM",
    "TARGET_STAGE_Z_HI",
    "InterceptConfig",
    "ObsStageConfig",
    "RewardConfig",
    "SpawnConfig",
    "StageConfig",
    "make_obs_config",
    "observation_dim",
    "privileged_dim",
]