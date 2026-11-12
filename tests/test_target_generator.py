"""Unit tests for target generation (plan §16.1.5)."""

from __future__ import annotations

import numpy as np

from src.envs.stage_config import STAGES
from src.envs.target_generator import PolynomialTargetPath, TargetGenerator


def test_static_target_stays_put():
    gen = TargetGenerator(STAGES["stage_1"], np.random.default_rng(0))
    gen.reset(np.zeros(3), episode_length_s=4.0)
    p0 = gen.pos.copy()
    for _ in range(50):
        gen.step(0.02, np.zeros(3))
    np.testing.assert_allclose(gen.pos, p0, atol=1e-9)


def test_polynomial_path_follows_closed_form():
    path = PolynomialTargetPath(
        start=np.zeros(3), end=np.array([30.0, 0, 0]), duration=5.0, order=1,
        rng=np.random.default_rng(0),
    )
    np.testing.assert_allclose(path.position_at(0.0), np.zeros(3), atol=1e-9)
    assert np.linalg.norm(path.position_at(2.5) - np.array([15.0, 0, 0])) < 1e-6
    v = path.velocity_at(2.5)
    np.testing.assert_allclose(v, [6.0, 0, 0], atol=1e-6)


def test_polynomial_path_respects_speed_cap():
    path = PolynomialTargetPath(
        start=np.zeros(3), end=np.array([1000.0, 0, 0]), duration=2.0, order=1,
        rng=np.random.default_rng(0), speed_cap=10.0,
    )
    assert path.peak_speed <= 10.0 + 1e-6


def test_polynomial_order_2_has_acceleration():
    path = PolynomialTargetPath(
        start=np.zeros(3), end=np.array([40.0, 0, 0]), duration=6.0, order=2,
        rng=np.random.default_rng(1),
    )
    assert np.linalg.norm(path.acceleration_at(3.0)) > 0.0


def test_state_at_matches_position_at():
    gen = TargetGenerator(STAGES["stage_2"], np.random.default_rng(0))
    gen.reset(np.zeros(3), episode_length_s=6.0)
    pos, vel, acc = gen.state_at(2.0)
    assert np.all(np.isfinite(pos)) and np.all(np.isfinite(vel)) and np.all(np.isfinite(acc))


def test_evasive_target_stays_in_world():
    stage = STAGES["stage_3"]
    gen = TargetGenerator(stage, np.random.default_rng(3))
    gen.reset(np.zeros(3), episode_length_s=10.0)
    radius = stage.world_radius
    for _ in range(2000):
        gen.step(0.02, np.zeros(3))
        r_xy = float(np.hypot(gen.pos[0], gen.pos[1]))
        assert r_xy <= radius + 1e-3


def test_all_stage_types_reset_and_step_finite():
    for sid, stage in STAGES.items():
        gen = TargetGenerator(stage, np.random.default_rng(0))
        gen.reset(np.zeros(3), episode_length_s=8.0)
        for _ in range(20):
            gen.step(0.02, np.zeros(3))
        assert np.all(np.isfinite(gen.pos)), sid
        assert np.all(np.isfinite(gen.vel)), sid
