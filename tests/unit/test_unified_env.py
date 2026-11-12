"""Behavioural tests for the unified guidance environment (P1.8/P5 contract)."""

from dataclasses import replace

import numpy as np
import pytest

from interceptor.guidance.env import TARGET_SETS, UnifiedInterceptEnv
from interceptor.guidance.maneuvers import ManeuverType, PlatformType
from interceptor.guidance.observation import AsymmetricObsBuilder
from interceptor.guidance.platform import PLATFORMS
from interceptor.guidance.seeker import SensorNoiseConfig
from interceptor.guidance.targets import StaticTarget

PLATFORMS_UNDER_TEST = [PlatformType.QUADCOPTER, PlatformType.FIXED_WING,
                        PlatformType.MISSILE]


@pytest.mark.parametrize("platform", PLATFORMS_UNDER_TEST)
@pytest.mark.parametrize("target_set", TARGET_SETS)
def test_observation_dims_and_target_sets(platform, target_set):
    env = UnifiedInterceptEnv(platform=platform, mode="apn_filtered",
                              target_set=target_set, seed=7)
    obs, info = env.reset()
    assert env.observation_space.shape == (101,)
    assert env.action_space.shape == (3,)
    assert env.obs_builder.actor_dim == 85
    assert env.obs_builder.PRIVILEGED_DIM == 16
    assert env.obs_builder.total_dim == 85 + 16
    assert obs.shape == (101,)
    assert np.all(np.isfinite(obs))
    assert info["platform"] == platform.value

    p0 = env.target.p_W.copy()
    obs, reward, terminated, truncated, info = env.step(np.zeros(3))
    assert obs.shape == (101,)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert set(("a_des", "v_des", "omega_des", "miss_distance", "killed",
                "crashed", "diverged", "steps")) <= set(info)
    if target_set == "static":
        np.testing.assert_array_equal(env.target.p_W, p0)
        assert np.linalg.norm(env.target.v_W) == 0.0
    else:
        assert np.linalg.norm(env.target.p_W - p0) > 0.0


@pytest.mark.parametrize("platform", PLATFORMS_UNDER_TEST)
def test_a3_zero_action_is_bit_identical_to_a2(platform):
    """A3 with a zero residual must reproduce A2 exactly (plan P5)."""
    common = dict(platform=platform, target_set="evasive",
                  fixed_maneuver=ManeuverType.BARREL_ROLL, seed=1234)
    a2 = UnifiedInterceptEnv(mode="apn_filtered", **common)
    a3 = UnifiedInterceptEnv(mode="apn_rl", **common)
    obs2, _ = a2.reset()
    obs3, _ = a3.reset()
    np.testing.assert_array_equal(obs2, obs3)
    for step in range(80):
        obs2, r2, t2, tr2, i2 = a2.step(np.zeros(3))
        obs3, r3, t3, tr3, i3 = a3.step(np.zeros(3))
        np.testing.assert_array_equal(obs2, obs3)
        assert r2 == r3
        assert (t2, tr2) == (t3, tr3)
        np.testing.assert_array_equal(i2["velocity_output"], i3["velocity_output"])
        np.testing.assert_array_equal(i2["a_des"], i3["a_des"])
        np.testing.assert_array_equal(i2["a_apn"], i3["a_apn"])
        np.testing.assert_array_equal(i2["a_rl"], i3["a_rl"])
        if t2 or tr2:
            break
    np.testing.assert_array_equal(a2.get_trajectory_log()["interceptor"],
                                  a3.get_trajectory_log()["interceptor"])
    np.testing.assert_array_equal(a2.get_trajectory_log()["target"],
                                  a3.get_trajectory_log()["target"])


@pytest.mark.parametrize("offset,should_kill", [(0.0, True), (5.0, False)])
def test_kill_uses_continuous_closest_approach(offset, should_kill):
    """A fast crossing whose endpoints are far apart still kills on CPA."""
    limits = replace(PLATFORMS[PlatformType.QUADCOPTER],
                     dt=0.2, kill_radius=0.1, v_max=100.0,
                     v_target_nominal=100.0)
    env = UnifiedInterceptEnv(platform=PlatformType.QUADCOPTER,
                              mode="apn_filtered", target_set="static",
                              limits=limits,
                              noise_config=SensorNoiseConfig(enabled=False), seed=3)
    env.reset()
    env.p_M = np.array([0.0, 0.0, 100.0])
    env.v_M = np.array([100.0, 0.0, 0.0])
    env.a_M = np.zeros(3)
    env.target = StaticTarget(np.array([10.0, offset, 100.0]))
    env.min_miss_distance = np.inf
    env.prev_dist = float(np.linalg.norm(env.target.p_W - env.p_M))
    _, _, terminated, _, info = env.step(np.zeros(3))
    if should_kill:
        assert info["killed"] and terminated
        assert info["miss_distance"] < 1e-9
    else:
        assert not info["killed"] and not terminated
        # Both step endpoints are outside the kill radius; the continuous CPA
        # is smaller than the initial offset because APN curves toward it.
        assert info["distance"] > limits.kill_radius
        assert 0.0 < info["miss_distance"] < 5.0


def test_legacy_mode_arguments_map_to_modes():
    raw = UnifiedInterceptEnv(use_apn=True, pure_apn_mode=True, seed=1)
    hybrid = UnifiedInterceptEnv(use_apn=True, pure_apn_mode=False, seed=1)
    pure = UnifiedInterceptEnv(use_apn=False, pure_apn_mode=False, seed=1)
    assert (raw.mode, hybrid.mode, pure.mode) == ("apn_raw", "apn_rl", "pure_rl")
