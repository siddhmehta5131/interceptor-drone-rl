"""Unit tests for the reward function (plan §16.1.4)."""

from __future__ import annotations

import numpy as np

from interceptor.envs.reward import RewardFunction
from interceptor.envs.stage_config import RewardConfig


def _rf():
    return RewardFunction(RewardConfig(), v_max=50.0, a_max=15.0, kill_radius=1.0)


def _compute(rf, **overrides):
    base = dict(
        distance=10.0, prev_distance=10.0, closing_vel=0.0, LOS_rate_norm=0.0,
        a_total_norm=0.0, a_rl_norm=0.0, action=np.zeros(3), prev_action=np.zeros(3),
        ZEM=0.0, step_count=10, max_steps=200, killed=False, out_of_bounds=False,
    )
    base.update(overrides)
    return rf.compute(**base)


def test_kill_gives_large_positive_reward_and_terminates():
    reward, terminated, breakdown = _compute(
        _rf(), distance=0.2, prev_distance=5.0, step_count=5, killed=True
    )
    assert terminated is True
    assert reward > 20.0
    assert breakdown.kill > 0.0


def test_out_of_bounds_penalty_terminates():
    reward, terminated, breakdown = _compute(_rf(), out_of_bounds=True)
    assert terminated is True
    assert reward < 0.0
    assert breakdown.crash < 0.0


def test_progress_rewards_approaching():
    reward, _, breakdown = _compute(_rf(), distance=5.0, prev_distance=10.0)
    assert breakdown.progress > 0.0


def test_energy_and_rl_residual_are_penalties():
    _, _, breakdown = _compute(_rf(), a_total_norm=15.0, a_rl_norm=10.0)
    assert breakdown.energy < 0.0
    assert breakdown.rl_residual < 0.0


def test_smoothness_penalises_action_change():
    _, _, breakdown = _compute(
        _rf(), action=np.ones(3), prev_action=-np.ones(3)
    )
    assert breakdown.smooth < 0.0


def test_zem_penalty_scales_with_miss():
    _, _, small = _compute(_rf(), distance=10.0, ZEM=0.1)
    _, _, big = _compute(_rf(), distance=10.0, ZEM=5.0)
    assert big.zem < small.zem <= 0.0
