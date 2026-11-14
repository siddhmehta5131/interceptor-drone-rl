"""P5.8: per-platform observation scales, dimensions and value bounds.

The actor frame stays 16 values, the history block 64 (4 frames) and the
future block 21 (3 samples of 7), so the actor is 85 values; the privileged
critic block is 16, giving 101 in total for every platform.
"""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from interceptor.guidance.env import UnifiedInterceptEnv
from interceptor.guidance.maneuvers import PlatformType
from interceptor.guidance.observation import (
    AsymmetricObsBuilder,
    ObservationScales,
    load_observation_scales,
)
from interceptor.guidance.platform import PLATFORMS, load_platforms

ROOT = Path(__file__).resolve().parents[2]
PLATFORM_LIST = [PlatformType.QUADCOPTER, PlatformType.FIXED_WING,
                 PlatformType.MISSILE]


def _config_scales():
    return load_observation_scales(ROOT / "configs/platforms.yaml")


def test_every_configured_scale_is_sourced():
    data = yaml.safe_load((ROOT / "configs/platforms.yaml").read_text(encoding="utf-8"))
    block = data["observation_scales"]
    names = {f.name for f in fields(ObservationScales)}
    for platform in PLATFORM_LIST:
        entry = block[platform.value]
        assert set(entry) == names
        for name, leaf in entry.items():
            assert isinstance(leaf, dict), (platform.value, name)
            assert set(("value", "unit", "source")) <= set(leaf), (platform.value, name)
            assert float(leaf["value"]) > 0.0
            assert leaf["unit"] and leaf["source"]


def test_loaded_scales_match_the_config_values():
    data = yaml.safe_load((ROOT / "configs/platforms.yaml").read_text(encoding="utf-8"))
    out = _config_scales()
    for platform in PLATFORM_LIST:
        got = out[platform]
        want = data["observation_scales"][platform.value]
        for name in (f.name for f in fields(ObservationScales)):
            assert getattr(got, name) == pytest.approx(float(want[name]["value"]))


@pytest.mark.parametrize("platform", PLATFORM_LIST, ids=[p.value for p in PLATFORM_LIST])
def test_observation_dims_are_85_16_101_for_all_platforms(platform):
    env = UnifiedInterceptEnv(platform=platform, mode="apn_filtered",
                              target_set="static", seed=11)
    obs, _ = env.reset()
    assert env.obs_builder.actor_dim == 85
    assert env.obs_builder.PRIVILEGED_DIM == 16
    assert env.obs_builder.total_dim == 101
    assert env.observation_space.shape == (101,)
    assert obs.shape == (101,)
    # 64 history + 21 future + 16 privileged.
    assert AsymmetricObsBuilder.FRAME_DIM == 16
    assert AsymmetricObsBuilder.FUTURE_SAMPLE_DIM == 7
    assert AsymmetricObsBuilder.PRIVILEGED_DIM == 16


@pytest.mark.parametrize("platform", PLATFORM_LIST, ids=[p.value for p in PLATFORM_LIST])
def test_random_episode_values_stay_within_bounds(platform):
    env = UnifiedInterceptEnv(platform=platform, mode="apn_rl",
                              target_set="evasive", noise_scale=2.0, seed=19)
    rng = np.random.default_rng(19)
    obs, _ = env.reset()
    assert np.all(np.abs(obs) <= 10.0)
    for _ in range(40):
        obs, _, terminated, truncated, _ = env.step(
            rng.uniform(-1.0, 1.0, size=3))
        assert np.all(np.isfinite(obs))
        assert np.all(np.abs(obs) <= 10.0)
        if terminated or truncated:
            break


@pytest.mark.parametrize("platform", PLATFORM_LIST, ids=[p.value for p in PLATFORM_LIST])
def test_config_scales_and_config_limits_keep_observations_bounded(platform):
    """The shipped config block, not just the derived fallback, is usable."""
    limits = load_platforms(ROOT / "configs/platforms.yaml")
    env = UnifiedInterceptEnv(
        platform=platform, mode="apn_filtered", target_set="evasive",
        limits=limits[platform], obs_scales=_config_scales()[platform],
        seed=23)
    rng = np.random.default_rng(23)
    obs, _ = env.reset()
    assert np.all(np.abs(obs) <= 10.0)
    for _ in range(30):
        obs, _, terminated, truncated, _ = env.step(
            rng.uniform(-1.0, 1.0, size=3))
        assert np.all(np.abs(obs) <= 10.0)
        if terminated or truncated:
            break


def test_builder_applies_supplied_scales_and_keeps_privileged_layout():
    lim = PLATFORMS[PlatformType.QUADCOPTER]
    base = ObservationScales.from_limits(lim)
    scales = replace(
        base, rel_pos=10.0, rel_vel=2.0, interceptor_vel=2.0, range=10.0,
        closing_vel=2.0, t_go=10.0, future_pos=10.0, future_vel=2.0,
        time_remaining=10.0, optimal_accel=5.0, optimal_vel=2.0,
        target_accel=2.0,
    )
    builder = AsymmetricObsBuilder(lim, scales=scales)
    obs = builder.build(
        p_M=np.zeros(3), v_M=np.array([1.0, 0.0, 0.0]),
        est_p_T=np.array([10.0, 0.0, 0.0]),
        est_v_T=np.array([-1.0, 0.0, 0.0]), est_a_T=np.zeros(3),
        visible=True, dt=0.02, time_remaining=5.0,
        optimal_accel=np.array([5.0, 0.0, 0.0]),
        optimal_vel=np.array([2.0, 0.0, 0.0]),
        true_future_pos=np.array([12.0, 0.0, 0.0]),
        true_future_vel=np.array([2.0, 0.0, 0.0]),
        true_acc=np.array([2.0, 0.0, 0.0]),
    )
    assert obs.shape == (101,)

    frame = obs[:16]
    np.testing.assert_allclose(frame[0:3], [1.0, 0.0, 0.0])    # rel_pos/r
    np.testing.assert_allclose(frame[3:6], [-1.0, 0.0, 0.0])   # rel_vel/v
    np.testing.assert_allclose(frame[6:9], [0.5, 0.0, 0.0])    # v_M/v
    assert frame[9] == pytest.approx(1.0)                      # range/r
    assert frame[10] == pytest.approx(1.0)                     # closing/v
    np.testing.assert_allclose(frame[11:14], 0.0)              # LOS rate
    assert frame[14] == pytest.approx(0.5)                     # t_go/time
    assert frame[15] == pytest.approx(1.0)                     # visible

    # Future sample 0 at 0.1 s: range 9.9 - 0.1, velocity -1 - 1.
    np.testing.assert_allclose(obs[64:67], [0.98, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(obs[67:70], [-1.0, 0.0, 0.0], atol=1e-6)

    priv = obs[85:]
    assert priv.shape == (16,)
    assert priv[0] == pytest.approx(0.5)                       # time/10
    np.testing.assert_allclose(priv[1:4], [1.0, 0.0, 0.0])     # optimal accel
    np.testing.assert_allclose(priv[4:7], [1.0, 0.0, 0.0])     # optimal vel
    np.testing.assert_allclose(priv[7:10], [1.2, 0.0, 0.0])    # true future pos
    np.testing.assert_allclose(priv[10:13], [0.5, 0.0, 0.0])   # true future vel
    np.testing.assert_allclose(priv[13:16], [1.0, 0.0, 0.0])   # true accel
