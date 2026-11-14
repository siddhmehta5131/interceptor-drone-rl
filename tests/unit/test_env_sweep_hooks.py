"""Sweep hooks on the guidance env: manoeuvre shape scale and reward-weight scaling.

The defaults (``maneuver_shape_scale=1``, ``reward_scale=None``) must leave every
trajectory bit-identical, so the golden fixtures and headline results do not move;
non-default values must actually change what they claim to change.
"""
import numpy as np
import pytest

from interceptor.guidance.env import UnifiedInterceptEnv
from interceptor.guidance.maneuvers import ManeuverType, PlatformType

STEPS = 120


def _rollout(seed=3, maneuver=ManeuverType.SINUSOIDAL_WEAVE, **kw):
    env = UnifiedInterceptEnv(PlatformType.QUADCOPTER, mode="apn_filtered",
                              target_set="evasive", fixed_maneuver=maneuver,
                              seed=seed, **kw)
    env.reset(seed=seed)
    tgt, rew = [], []
    for _ in range(STEPS):
        _, r, term, trunc, _ = env.step(np.zeros(3, dtype=np.float32))
        tgt.append(env.target.p_W.copy())
        rew.append(r)
        if term or trunc:
            break
    return np.array(tgt), np.array(rew)


def test_defaults_leave_trajectory_unchanged():
    base_t, base_r = _rollout()
    t, r = _rollout(maneuver_shape_scale=1.0, reward_scale=None)
    np.testing.assert_array_equal(base_t, t)
    np.testing.assert_array_equal(base_r, r)


def test_unit_reward_scale_is_identity():
    _, base_r = _rollout()
    _, r = _rollout(reward_scale={"w_progress": 1.0, "c_time": 1.0})
    np.testing.assert_array_equal(base_r, r)


def test_shape_scale_changes_target_path():
    base_t, _ = _rollout()
    t, _ = _rollout(maneuver_shape_scale=1.5)
    n = min(len(base_t), len(t))
    assert np.max(np.linalg.norm(base_t[:n] - t[:n], axis=1)) > 0.5


def test_shape_scale_stored_on_target():
    env = UnifiedInterceptEnv(PlatformType.QUADCOPTER, mode="apn_filtered",
                              target_set="evasive", seed=0, maneuver_shape_scale=0.5)
    env.reset(seed=0)
    assert env.target.shape_scale == 0.5


def test_reward_scale_multiplies_weights_and_changes_reward():
    env = UnifiedInterceptEnv(reward_scale={"w_progress": 2.0})
    ref = UnifiedInterceptEnv()
    assert env.reward_fn.w_progress == pytest.approx(2.0 * ref.reward_fn.w_progress)
    assert env.reward_fn.c_time == ref.reward_fn.c_time
    base_t, base_r = _rollout()
    t, r = _rollout(reward_scale={"w_progress": 2.0})
    np.testing.assert_array_equal(base_t, t)  # reward weights never touch dynamics
    assert not np.allclose(base_r, r)


def test_reward_scale_unknown_weight_raises():
    with pytest.raises((TypeError, AttributeError)):
        UnifiedInterceptEnv(reward_scale={"w_not_a_weight": 2.0})
