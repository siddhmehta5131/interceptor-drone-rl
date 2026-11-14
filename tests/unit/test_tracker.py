"""P5.4: Kalman tracker consistency and accuracy (EQ-KF-1..5)."""
import numpy as np
import pytest

from interceptor.guidance.platform import PLATFORMS
from interceptor.guidance.seeker import SensorNoiseConfig, SeekerSensorFilter
from interceptor.guidance.tracker import KalmanTracker, transition_matrices


def _run(platform, filter_type, seed, steps=400, dropout=0.04):
    lim = PLATFORMS[platform]
    cfg = SensorNoiseConfig(dropout_prob=dropout)
    f = SeekerSensorFilter(cfg, lim, np.random.default_rng(seed), filter_type=filter_type)
    dt = lim.dt
    rng = np.random.default_rng(seed + 1)
    p = np.array([500.0, 0, 100.0]); v = np.array([lim.v_target_nominal, 0, 0.0])
    a = np.zeros(3); t_acc = 0.0
    f.reset(p, v, a)
    err_est, err_raw, nis = [], [], []
    for k in range(steps):
        # slowly varying smooth manoeuvre acceleration, well inside the limit
        a = 0.3 * lim.a_target_max * np.array([0, np.sin(0.5 * k * dt), np.cos(0.3 * k * dt)])
        p = p + v * dt + 0.5 * a * dt ** 2; v = v + a * dt
        ep, _, _, vis = f.observe(np.zeros(3), p, v, a, dt)
        if vis:
            err_est.append(np.linalg.norm(ep - p)); err_raw.append(np.linalg.norm(f.raw_pos - p))
            if f.kf is not None:
                nis.append(f.kf.last_nis)
    return np.array(err_est), np.array(err_raw), np.array(nis)


def test_transition_matrices_shape_and_symmetry():
    F, Q = transition_matrices(0.02, 4.0)
    assert np.allclose(Q, Q.T) and np.all(np.linalg.eigvalsh(Q) >= -1e-15)
    assert np.allclose(F[0], [1, 0.02, 2e-4])


@pytest.mark.parametrize("platform", list(PLATFORMS))
def test_estimate_error_below_raw_measurement_error(platform):
    e, r, _ = _run(platform, "kalman", 3)
    assert np.mean(e[50:]) < np.mean(r[50:])


@pytest.mark.parametrize("platform", list(PLATFORMS))
def test_normalised_innovation_consistent(platform):
    # Mean NIS over many steps should be near 3 (chi^2, 3 dof per axis group
    # after averaging); the process model is mismatched so allow a band.
    nis = np.concatenate([_run(platform, "kalman", s)[2] for s in range(5)])
    assert 1.5 < np.mean(nis) < 4.5


def test_coasts_through_dropout():
    e, _, _ = _run(next(iter(PLATFORMS)), "kalman", 5, dropout=0.5)
    assert np.all(np.isfinite(e))


def test_default_filter_is_unchanged_abg():
    lim = PLATFORMS[next(iter(PLATFORMS))]
    assert SeekerSensorFilter(SensorNoiseConfig(), lim).filter_type == "abg"
