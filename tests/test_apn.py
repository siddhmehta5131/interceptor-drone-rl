"""Unit tests for the APN guidance law (plan §16.1.1)."""

from __future__ import annotations

import numpy as np

from interceptor.guidance.apn import (
    compute_apn_command,
    compute_axial_acceleration,
    compute_closing_velocity,
    compute_los_rate,
    compute_time_to_go,
    compute_zem,
)
from interceptor.guidance.kinematics import (
    compute_3d_apn_command,
    compute_engagement_kinematics,
)
from interceptor.guidance.maneuvers import PlatformType
from interceptor.guidance.platform import PLATFORMS


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


# --------------------------------------------------------------------- P5.5
# Axial speed-management term (EQ-APN-3), tested separately from the
# perpendicular APN law so arm A1/A2 differ by exactly this factor.


def test_axial_term_inactive_at_high_speed_and_closing_rate():
    lim = PLATFORMS[PlatformType.QUADCOPTER]
    v_M = np.array([0.99 * lim.v_max, 0.0, 0.0])
    a_axial = compute_axial_acceleration(
        v_M, lim, np.array([1.0, 0.0, 0.0]),
        closing_speed=0.90 * lim.v_max)
    assert np.allclose(a_axial, 0.0)


def test_axial_term_omnidirectional_points_along_los():
    lim = PLATFORMS[PlatformType.QUADCOPTER]
    los = np.array([0.0, 1.0, 0.0])
    a_axial = compute_axial_acceleration(
        np.array([10.0, 0.0, 0.0]), lim, los, closing_speed=0.0)
    assert np.allclose(a_axial, 0.8 * lim.a_t_max * los)
    assert np.linalg.norm(a_axial) <= lim.a_t_max + 1e-9


def test_axial_term_non_holonomic_points_along_velocity():
    lim = PLATFORMS[PlatformType.FIXED_WING]
    v_M = np.array([0.0, lim.v_min, 0.0])
    a_axial = compute_axial_acceleration(
        v_M, lim, np.array([1.0, 0.0, 0.0]), closing_speed=0.0)
    assert np.allclose(a_axial, lim.a_t_max * np.array([0.0, 1.0, 0.0]))


def test_axial_term_activates_on_one_condition():
    lim = PLATFORMS[PlatformType.QUADCOPTER]
    los = np.array([1.0, 0.0, 0.0])
    # Full speed but too little closing rate: active.
    a_slow_close = compute_axial_acceleration(
        lim.v_max * los, lim, los, closing_speed=0.0)
    # Slow but closing fast: active too.
    a_slow_speed = compute_axial_acceleration(
        0.5 * lim.v_max * los, lim, los, closing_speed=0.9 * lim.v_max)
    assert np.allclose(a_slow_close, 0.8 * lim.a_t_max * los)
    assert np.allclose(a_slow_speed, 0.8 * lim.a_t_max * los)


def test_axial_default_closing_speed_is_velocity_projection():
    lim = PLATFORMS[PlatformType.QUADCOPTER]
    v_M = np.array([10.0, 2.0, 0.0])
    los = np.array([1.0, 0.0, 0.0])
    default = compute_axial_acceleration(v_M, lim, los)
    explicit = compute_axial_acceleration(
        v_M, lim, los, closing_speed=float(np.dot(v_M, los)))
    assert np.allclose(default, explicit)


def test_apn_command_switch_adds_exactly_the_axial_term():
    lim = PLATFORMS[PlatformType.QUADCOPTER]
    kin = compute_engagement_kinematics(
        np.array([100.0, 0.0, 0.0]), np.zeros(3), np.zeros(3),
        np.zeros(3), np.array([20.0, 0.0, 0.0]))
    a_with = compute_3d_apn_command(
        kin, np.zeros(3), np.array([20.0, 0.0, 0.0]), lim,
        include_axial_boost=True)
    a_without = compute_3d_apn_command(
        kin, np.zeros(3), np.array([20.0, 0.0, 0.0]), lim,
        include_axial_boost=False)
    expected = compute_axial_acceleration(
        np.array([20.0, 0.0, 0.0]), lim, kin.los_vec,
        closing_speed=kin.v_cl)
    assert np.allclose(a_with - a_without, expected)
    assert np.linalg.norm(expected) > 0.0

