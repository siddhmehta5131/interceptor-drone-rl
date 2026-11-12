"""Unit tests for the APN guidance law (plan §16.1.1)."""

from __future__ import annotations

import numpy as np

from src.guidance.apn import (
    compute_apn_command,
    compute_closing_velocity,
    compute_los_rate,
    compute_time_to_go,
    compute_zem,
)


def test_head_on_static_target_gives_zero_command():
    a_cmd, info = compute_apn_command(
        np.zeros(3), np.array([20.0, 0, 0]), np.array([50.0, 0, 0]), np.zeros(3), np.zeros(3)
    )
    assert np.linalg.norm(a_cmd) < 1e-6
    assert info["closing_vel"] > 0.0
    assert info["range"] == 50.0


def test_crossing_target_leads_laterally():
    a_cmd, info = compute_apn_command(
        np.zeros(3), np.array([20.0, 0, 0]), np.array([40.0, 0, 0]), np.array([0, 8.0, 0]), np.zeros(3)
    )
    assert a_cmd[1] > 0.0
    assert info["LOS_rate"] > 0.0


def test_accelerating_target_uses_augmentation():
    _, info_plain = compute_apn_command(
        np.zeros(3), np.array([20.0, 0, 0]), np.array([40.0, 0, 5.0]),
        np.array([0, 0, 0]), np.zeros(3),
    )
    _, info_aug = compute_apn_command(
        np.zeros(3), np.array([20.0, 0, 0]), np.array([40.0, 0, 5.0]),
        np.array([0, 0, 0]), np.array([0.0, 0.0, 6.0]),
    )
    assert np.linalg.norm(info_aug["a_aug"]) > np.linalg.norm(info_plain["a_aug"])


def test_los_rate_zero_when_collinear():
    R = np.array([10.0, 0, 0])
    V_rel = np.array([-5.0, 0, 0])
    assert np.linalg.norm(compute_los_rate(R, V_rel)) < 1e-9


def test_closing_velocity_sign():
    R_hat = np.array([1.0, 0, 0])
    assert compute_closing_velocity(np.array([-5.0, 0, 0]), R_hat) > 0.0
    assert compute_closing_velocity(np.array([5.0, 0, 0]), R_hat) < 0.0


def test_time_to_go_bounded():
    assert compute_time_to_go(50.0, 25.0) == 2.0
    # opening geometry must not explode
    assert compute_time_to_go(50.0, -1.0, t_go_min=0.1, t_go_max=20.0) <= 20.0


def test_zem_zero_for_perfect_intercept():
    ZEM, ZEM_vec = compute_zem(
        V_rel=np.array([-20.0, 0, 0]), a_target=np.zeros(3), t_go=2.0,
        R_hat=np.array([1.0, 0, 0]),
    )
    assert abs(ZEM) < 1e-9
    assert np.linalg.norm(ZEM_vec) < 1e-9


def test_zem_nonzero_for_miss():
    ZEM, _ = compute_zem(
        V_rel=np.array([-20.0, 5.0, 0]), a_target=np.zeros(3), t_go=2.0,
        R_hat=np.array([1.0, 0, 0]),
    )
    assert ZEM > 0.0
