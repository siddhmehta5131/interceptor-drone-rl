"""Unit tests for observation history stacking (plan §16.1.8)."""

from __future__ import annotations

import numpy as np

from src.envs.obs_builder import FRAME_DIM, ObsBuilder
from src.envs.stage_config import STAGES

_TARGET_POS = np.array([30.0, 0.0, 0.0])
_TARGET_VEL = np.zeros(3)
_TARGET_ACC = np.zeros(3)


def _make_builder() -> ObsBuilder:
    return ObsBuilder(STAGES["stage_1"].obs)


def _build(builder: ObsBuilder, interceptor_pos: np.ndarray) -> np.ndarray:
    actor, _ = builder.build(
        interceptor_pos=interceptor_pos,
        interceptor_vel=np.array([15.0, 0.0, 0.0]),
        target_pos=_TARGET_POS,
        target_vel=_TARGET_VEL,
        target_acc=_TARGET_ACC,
        target_visible=True,
        dt=0.02,
        time_remaining=3.0,
        optimal_accel=np.zeros(3),
        optimal_vel=np.zeros(3),
        target_true_future_pos=_TARGET_POS,
        target_true_future_vel=_TARGET_VEL,
        target_true_accel=_TARGET_ACC,
    )
    return actor


def test_build_shapes():
    builder = _make_builder()
    builder.reset()
    actor = _build(builder, np.zeros(3))
    assert actor.shape == (85,)


def test_newest_frame_is_first_block():
    builder = _make_builder()
    builder.reset()
    ip = np.zeros(3)
    actor = _build(builder, ip)
    expected = builder._normalise_frame(
        builder.compute_frame(ip, np.array([15.0, 0.0, 0.0]),
                              _TARGET_POS, _TARGET_VEL, True)
    )
    np.testing.assert_allclose(actor[:FRAME_DIM], expected, atol=1e-6)


def test_history_block_contains_past_frames():
    builder = _make_builder()
    builder.reset()
    # Populate enough history (need > m*p = 6 frames for all slots).
    for i in range(10):
        _build(builder, np.array([float(i), 0.0, 0.0]))
    actor = _build(builder, np.array([10.0, 0.0, 0.0]))
    history = actor[: FRAME_DIM * 4].reshape(4, FRAME_DIM)
    # The four sampled frames should not all be identical.
    assert not np.allclose(history[0], history[3])


def test_history_reset_clears_state():
    builder = _make_builder()
    builder.reset()
    for i in range(8):
        _build(builder, np.array([float(i), 0.0, 0.0]))
    builder.reset()
    actor = _build(builder, np.zeros(3))
    history = actor[: FRAME_DIM * 4].reshape(4, FRAME_DIM)
    # After reset only one frame exists so all history slots are identical.
    for k in range(4):
        np.testing.assert_allclose(history[k], history[0], atol=1e-9)
