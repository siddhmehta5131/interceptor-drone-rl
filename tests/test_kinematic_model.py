"""Unit tests for point-mass interceptor kinematics (plan §16.1.2)."""

from __future__ import annotations

import numpy as np

from src.guidance.kinematic_model import KinematicModel, InterceptorState


def test_reset_returns_expected_state():
    km = KinematicModel(v_max=50.0, a_max=15.0, omega_max=3.0, dt=0.02)
    state = km.reset(position=np.array([1.0, 2.0, 3.0]), velocity=np.array([5.0, 0, 0]))
    assert isinstance(state, InterceptorState)
    np.testing.assert_allclose(state.position, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(state.velocity, [5.0, 0, 0])


def test_constant_acceleration_integration():
    km = KinematicModel(v_max=50.0, a_max=15.0, omega_max=3.0, dt=0.02)
    state = km.reset(position=np.zeros(3), velocity=np.zeros(3))
    for _ in range(50):  # 1 s at a_max
        state = km.step(state, np.array([15.0, 0, 0]))
    assert abs(np.linalg.norm(state.velocity) - 15.0) < 1.0
    assert abs(state.position[0] - 7.5) < 0.5


def test_acceleration_is_clamped():
    km = KinematicModel(v_max=50.0, a_max=15.0, omega_max=3.0, dt=0.02)
    state = km.reset(position=np.zeros(3), velocity=np.zeros(3))
    state = km.step(state, np.array([1000.0, 0, 0]))  # way over a_max
    assert np.linalg.norm(state.velocity) <= 15.0 + 1e-6


def test_velocity_is_clamped():
    km = KinematicModel(v_max=20.0, a_max=100.0, omega_max=3.0, dt=0.02)
    state = km.reset(position=np.zeros(3), velocity=np.array([19.0, 0, 0]))
    for _ in range(50):
        state = km.step(state, np.array([100.0, 0, 0]))
    assert np.linalg.norm(state.velocity) <= 20.0 + 1e-6


def test_orientation_remains_unit_quaternion():
    km = KinematicModel(v_max=50.0, a_max=15.0, omega_max=3.0, dt=0.02)
    state = km.reset(position=np.zeros(3), velocity=np.array([10.0, 0, 0]))
    for _ in range(100):
        state = km.step(state, np.array([0.0, 5.0, 0.0]), omega_cmd=np.array([0.0, 0.0, 0.5]))
    assert abs(np.linalg.norm(state.orientation) - 1.0) < 1e-6
