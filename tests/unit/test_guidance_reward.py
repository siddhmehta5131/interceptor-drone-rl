"""P5.9: potential-based guidance reward, terminal terms and horizon discount."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from interceptor.guidance.env import UnifiedInterceptEnv
from interceptor.guidance.maneuvers import PlatformType
from interceptor.guidance.platform import PLATFORMS
from interceptor.guidance.reward import (
    HORIZON_STEPS,
    GuidanceReward,
    discounted_terminal,
    horizon_gamma,
    load_reward,
    notebook_reward,
)
from interceptor.guidance.seeker import SensorNoiseConfig
from interceptor.guidance.targets import StaticTarget

ROOT = Path(__file__).resolve().parents[2]


def _reward_zero_aux(**overrides):
    """Reward with auxiliary, action and time terms disabled."""
    base = dict(w_align=0.0, w_los=0.0, w_zem=0.0, w_res_rl=0.0,
                w_res_pure=0.0, w_smooth=0.0, c_time=0.0)
    base.update(overrides)
    return GuidanceReward(**base)


def _compute(rf, **overrides):
    base = dict(
        distance=10.0, prev_distance=10.0,
        los_vec=np.zeros(3), los_rate_vec=np.zeros(3), zem_norm=0.0,
        v_hat=np.zeros(3), action=np.zeros(3), prev_action=np.zeros(3),
        residual_weight=0.0, step_count=10, max_steps=200, r_init_max=100.0,
        killed=False, crashed=False, diverged=False, min_miss_distance=np.inf,
    )
    base.update(overrides)
    return rf.compute(**base)


def test_config_reward_matches_defaults_and_is_sourced():
    data = yaml.safe_load((ROOT / "configs/guidance.yaml").read_text(encoding="utf-8"))
    block = data["reward"]
    rf = load_reward(ROOT / "configs/guidance.yaml")
    for name, leaf in block.items():
        assert set(("value", "unit", "source")) <= set(leaf), name
        assert leaf["unit"] and leaf["source"]
        assert getattr(rf, name) == pytest.approx(float(leaf["value"]))
    # The shipped values are the code defaults.
    defaults = GuidanceReward()
    for name in block:
        assert getattr(rf, name) == pytest.approx(getattr(defaults, name), abs=1e-5)


def test_horizon_discount_keeps_ten_percent_at_the_first_step():
    gamma = horizon_gamma(HORIZON_STEPS)
    assert gamma ** HORIZON_STEPS == pytest.approx(0.1, rel=1e-9)
    # The plan's F16 note: a 150-point kill at step 0.
    assert discounted_terminal(gamma, HORIZON_STEPS, 150.0) == pytest.approx(15.0, abs=0.05)
    assert discounted_terminal(0.995, HORIZON_STEPS, 150.0) < 1.0


def test_potential_progress_telescopes_with_gamma_one():
    rf = _reward_zero_aux(gamma=1.0)
    total = 0.0
    prev = 100.0
    for dist in (80.0, 60.0, 40.0):
        _, terms = _compute(rf, distance=dist, prev_distance=prev)
        total += terms["progress"]
        prev = dist
    expected = rf.potential(40.0, 100.0) - rf.potential(100.0, 100.0)
    assert total == pytest.approx(expected)


def test_progress_is_positive_when_closing_and_negative_when_opening():
    rf = _reward_zero_aux()
    _, closing = _compute(rf, distance=90.0, prev_distance=100.0)
    _, opening = _compute(rf, distance=110.0, prev_distance=100.0)
    assert closing["progress"] > 0.0
    assert opening["progress"] < 0.0


def test_terminal_terms_have_the_expected_sign_and_magnitude():
    rf = _reward_zero_aux(gamma=0.0)
    _, kill = _compute(rf, killed=True, step_count=0, max_steps=1000)
    assert kill["terminal"] == pytest.approx(200.0)
    _, late = _compute(rf, killed=True, step_count=1000, max_steps=1000)
    assert late["terminal"] == pytest.approx(150.0)

    _, crash = _compute(rf, crashed=True)
    assert crash["terminal"] == pytest.approx(-80.0)

    _, diverge = _compute(rf, diverged=True, min_miss_distance=45.0, r_init_max=450.0)
    assert diverge["terminal"] == pytest.approx(-2.5)


def test_auxiliary_and_action_terms_are_penalties():
    rf = GuidanceReward()
    _, terms = _compute(
        rf, distance=100.0, prev_distance=100.0,
        los_vec=np.array([1.0, 0.0, 0.0]), v_hat=np.array([1.0, 0.0, 0.0]),
        los_rate_vec=np.array([0.1, 0.0, 0.0]), zem_norm=5.0,
        action=np.ones(3), prev_action=-np.ones(3), residual_weight=0.04,
        r_init_max=100.0)
    assert terms["align"] == pytest.approx(0.35)
    assert terms["los"] < 0.0
    assert terms["zem"] < 0.0
    assert terms["residual"] < 0.0
    assert terms["smooth"] < 0.0
    assert terms["time"] == pytest.approx(-0.005)


def test_total_equals_the_sum_of_the_terms():
    rf = GuidanceReward()
    reward, terms = _compute(
        rf, distance=95.0, prev_distance=100.0, killed=True,
        min_miss_distance=3.0)
    assert reward == pytest.approx(sum(terms.values()))


def test_env_reports_terms_and_uses_the_horizon_discount():
    env = UnifiedInterceptEnv(platform=PlatformType.QUADCOPTER,
                              mode="apn_rl", target_set="evasive", seed=5)
    assert env.gamma == pytest.approx(horizon_gamma(HORIZON_STEPS))
    env.reset()
    _, reward, _, _, info = env.step(np.zeros(3))
    terms = info["reward_terms"]
    assert reward == pytest.approx(sum(terms.values()))
    assert set(terms) == {"progress", "align", "los", "zem", "residual",
                          "smooth", "time", "terminal"}


def test_legacy_notebook_reward_is_available_only_for_fixture_replay():
    kwargs = dict(
        distance=90.0, prev_distance=100.0, los_vec=np.zeros(3),
        los_rate_vec=np.zeros(3), zem_norm=0.0, v_hat=np.zeros(3),
        action=np.zeros(3), prev_action=np.zeros(3), residual_weight=0.0,
        step_count=10, max_steps=200, r_init_max=100.0, v_max=50.0,
        dt=0.02, killed=False, crashed=False, diverged=False,
        min_miss_distance=np.inf)
    legacy, _ = notebook_reward(**kwargs)
    assert legacy == pytest.approx(2.5 * 10.0 / (50.0 * 0.02) - 0.005)


def test_env_kill_terminal_bonus_is_reported():
    limits = replace(PLATFORMS[PlatformType.QUADCOPTER], dt=0.2,
                     kill_radius=0.1, v_max=100.0, v_target_nominal=100.0)
    env = UnifiedInterceptEnv(platform=PlatformType.QUADCOPTER,
                              mode="apn_filtered", target_set="static",
                              limits=limits, noise_config=SensorNoiseConfig(enabled=False),
                              seed=3)
    env.reset()
    env.p_M = np.array([0.0, 0.0, 100.0])
    env.v_M = np.array([100.0, 0.0, 0.0])
    env.a_M = np.zeros(3)
    env.target = StaticTarget(np.array([10.0, 0.0, 100.0]))
    env.min_miss_distance = np.inf
    env.prev_dist = float(np.linalg.norm(env.target.p_W - env.p_M))
    _, _, terminated, _, info = env.step(np.zeros(3))
    assert info["killed"] and terminated
    assert info["reward_terms"]["terminal"] >= 150.0
