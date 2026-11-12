"""Unit tests for the optimal-trajectory oracle (plan §16.1.7)."""

from __future__ import annotations

import numpy as np

from src.envs.target_generator import PolynomialTargetPath
from src.guidance.optimal_trajectory import compute_optimal_intercept


class _StaticPath:
    """Minimal target-path protocol: a fixed point in space."""

    def __init__(self, pos):
        self._pos = np.asarray(pos, dtype=np.float64)

    def position_at(self, t):
        return self._pos.copy()

    def velocity_at(self, t):
        return np.zeros(3)

    def acceleration_at(self, t):
        return np.zeros(3)


def test_static_target_oracle_finite():
    result = compute_optimal_intercept(
        p_interceptor=np.zeros(3),
        v_interceptor=np.zeros(3),
        target_path=_StaticPath([30.0, 0.0, 0.0]),
        v_max=50.0,
        a_max=15.0,
        dt=0.02,
    )
    assert np.isfinite(result["optimal_time"])
    assert np.all(np.isfinite(result["optimal_positions"]))
    assert np.all(np.isfinite(result["optimal_velocities"]))
    assert np.all(np.isfinite(result["optimal_accelerations"]))


def test_oracle_result_keys():
    result = compute_optimal_intercept(
        p_interceptor=np.zeros(3),
        v_interceptor=np.array([10.0, 0.0, 0.0]),
        target_path=_StaticPath([30.0, 5.0, 0.0]),
        v_max=50.0,
        a_max=15.0,
        dt=0.02,
    )
    for key in ("optimal_time", "optimal_positions", "optimal_velocities",
                "optimal_accelerations", "optimal_accel", "optimal_vel",
                "optimal_cost", "method"):
        assert key in result


def test_optimal_accel_within_bounds():
    result = compute_optimal_intercept(
        p_interceptor=np.zeros(3),
        v_interceptor=np.zeros(3),
        target_path=_StaticPath([40.0, 10.0, 5.0]),
        v_max=50.0,
        a_max=15.0,
        dt=0.02,
    )
    a = np.linalg.norm(result["optimal_accel"])
    assert a <= 15.0 + 1e-6


def test_polynomial_target_oracle_finite():
    path = PolynomialTargetPath(
        start=np.array([30.0, 0.0, 0.0]),
        end=np.array([30.0, 40.0, 10.0]),
        duration=6.0,
        order=1,
        rng=np.random.default_rng(0),
        speed_cap=20.0,
    )
    result = compute_optimal_intercept(
        p_interceptor=np.zeros(3),
        v_interceptor=np.array([15.0, 0.0, 0.0]),
        target_path=path,
        v_max=50.0,
        a_max=15.0,
        dt=0.02,
    )
    assert np.isfinite(result["optimal_time"])
    assert np.all(np.isfinite(result["optimal_positions"]))
    assert np.all(np.isfinite(result["optimal_velocities"]))
