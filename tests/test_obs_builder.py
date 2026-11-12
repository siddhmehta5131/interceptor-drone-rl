"""Unit tests for the observation builder (plan §16.1.3)."""

from __future__ import annotations

import numpy as np

from interceptor.envs.obs_builder import (
    FRAME_DIM,
    FUTURE_SAMPLE_DIM,
    PRIVILEGED_DIM,
    ObsBuilder,
)
from interceptor.envs.stage_config import ObsStageConfig


def _build(builder, i=0.0):
    pos = np.array([i, 0.0, 0.0])
    return builder.build(
        interceptor_pos=pos,
        interceptor_vel=np.zeros(3),
        target_pos=pos + np.array([10.0, 0.0, 0.0]),
        target_vel=np.zeros(3),
        target_acc=np.zeros(3),
        target_visible=True,
        dt=0.02,
        time_remaining=3.0,
        optimal_accel=np.zeros(3),
        optimal_vel=np.zeros(3),
        target_true_future_pos=pos,
        target_true_future_vel=np.zeros(3),
        target_true_accel=np.zeros(3),
    )


def test_module_dimension_constants():
    assert FRAME_DIM == 16
    assert FUTURE_SAMPLE_DIM == 7
    assert PRIVILEGED_DIM == 16


def test_frame_block_shape():
    builder = ObsBuilder(ObsStageConfig())
    builder.reset()
    frame = builder.compute_frame(
        np.zeros(3), np.zeros(3), np.array([10.0, 0, 0]), np.zeros(3), True
    )
    assert frame.shape == (16,)
    assert np.all(np.isfinite(frame))


def test_actor_and_privileged_dims():
    builder = ObsBuilder(ObsStageConfig())
    builder.reset()
    obs, priv = _build(builder)
    assert obs.shape == (85,)
    assert priv.shape == (16,)
    assert np.all(np.isfinite(obs)) and np.all(np.isfinite(priv))


def test_actor_dim_matches_config():
    cfg = ObsStageConfig()
    builder = ObsBuilder(cfg)
    assert builder.actor_dim == cfg.actor_dim == 85


def test_visible_flag_encodes_missing_target():
    builder = ObsBuilder(ObsStageConfig())
    builder.reset()
    frame_visible = builder.compute_frame(
        np.zeros(3), np.zeros(3), np.array([10.0, 0, 0]), np.zeros(3), True
    )
    builder.reset()
    frame_hidden = builder.compute_frame(
        np.zeros(3), np.zeros(3), np.array([10.0, 0, 0]), np.zeros(3), False
    )
    assert frame_visible[-1] == 1.0
    assert frame_hidden[-1] == 0.0
