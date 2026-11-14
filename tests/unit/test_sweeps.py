"""P9 helpers and env hooks."""
import numpy as np
import pytest

from interceptor.eval import sweeps
from interceptor.guidance.env import UnifiedInterceptEnv
from interceptor.guidance.maneuvers import ManeuverType, PlatformType
from interceptor.guidance.observation import AsymmetricObsBuilder
from interceptor.guidance.platform import PLATFORMS


def test_kill_rate_vs_radius_is_cdf():
    out = sweeps.kill_rate_vs_radius([1, 2, 3, 4], [0.5, 2, 10])
    assert [o["kill_rate"] for o in out] == [0.0, 0.5, 1.0]


def test_limits_for_scales_target():
    lim = PLATFORMS[PlatformType.QUADCOPTER]
    q = PlatformType.QUADCOPTER
    assert sweeps.limits_for(q, "target_accel_ratio", 0.5).a_target_max == pytest.approx(0.5 * lim.a_n_max)
    assert sweeps.limits_for(q, "target_speed_ratio", 0.5).v_target_nominal == pytest.approx(0.5 * lim.v_max)
    assert sweeps.limits_for(q, "noise_scale", 3) is lim


def test_residual_components_pythagoras():
    pos, tgt = np.zeros((3, 3)), np.tile([10.0, 0, 0], (3, 1))
    rl = np.array([[1.0, 2, 0], [0, 0, 3], [4, 0, 0]])
    par, perp = sweeps.residual_components(pos, tgt, np.zeros((3, 3)), rl)
    np.testing.assert_allclose(par, [1, 0, 4])
    np.testing.assert_allclose(perp, [2, 3, 0])


def test_sweep_row_shape():
    rows = sweeps.run_sweep("noise_scale", PlatformType.QUADCOPTER, "apn_filtered",
                            values=[0.0], behaviors=list(ManeuverType)[:1], episodes=1)
    assert rows[0]["episodes"] == 1 and 0 <= rows[0]["kill_rate"] <= 1


def test_ablation_zeroes_blocks_without_changing_dim():
    lim = PLATFORMS[PlatformType.QUADCOPTER]
    kw = dict(p_M=np.zeros(3), v_M=np.array([5.0, 0, 0]), est_p_T=np.array([100.0, 5, 0]),
              est_v_T=np.array([-3.0, 1, 0]), est_a_T=np.zeros(3), visible=True, dt=0.02,
              time_remaining=10, optimal_accel=np.zeros(3), optimal_vel=np.zeros(3),
              true_future_pos=np.zeros(3), true_future_vel=np.zeros(3), true_acc=np.zeros(3))
    full = AsymmetricObsBuilder(lim).build(**kw)
    for ab, sl in (("history", slice(16, 64)), ("prediction", slice(64, 85)),
                   ("privileged", slice(85, 101))):
        o = AsymmetricObsBuilder(lim, ablate=(ab,)).build(**kw)
        assert o.shape == full.shape and np.all(o[sl] == 0)
        np.testing.assert_array_equal(o[:16], full[:16])
        assert np.any(full[sl] != 0)
    with pytest.raises(ValueError):
        AsymmetricObsBuilder(lim, ablate=("nope",))


def test_env_residual_authority_and_gamma():
    e = UnifiedInterceptEnv(mode="apn_rl", residual_authority=0.5, gamma=0.99, seed=0)
    assert e.residual_authority == 0.5 and e.reward_fn.gamma == 0.99
    assert UnifiedInterceptEnv(seed=0).residual_authority == 0.35
