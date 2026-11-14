"""PolynomialTargetPath (non-evasive target set); moved from the archived test_target_generator."""

import numpy as np

from interceptor.guidance.polynomial_path import PolynomialTargetPath


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
