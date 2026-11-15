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


@pytest.mark.parametrize("platform", list(PlatformType))
def test_limits_for_scales_interceptor_response(platform):
    """Sweeps that defend the project-choice response limits (tuned_values_plan family C/G)."""
    lim = PLATFORMS[platform]
    lag = sweeps.limits_for(platform, "interceptor_lag_scale", 2.0)
    assert lag.tau_accel == pytest.approx(2.0 * lim.tau_accel) and lag.max_jerk == lim.max_jerk
    ax = sweeps.limits_for(platform, "axial_limit_scale", 0.06)
    assert ax.a_t_max == pytest.approx(0.06 * lim.a_t_max)
    assert ax.a_t_min == pytest.approx(0.06 * lim.a_t_min) and ax.a_n_max == lim.a_n_max
    assert sweeps.limits_for(platform, "jerk_limit_scale", 0.5).max_jerk == pytest.approx(0.5 * lim.max_jerk)
    for name in ("interceptor_lag_scale", "axial_limit_scale", "jerk_limit_scale"):
        assert 1.0 in sweeps.SWEEPS[name][1]  # nominal point is in every grid


def test_seeker_gain_scale_caps_at_one():
    from interceptor.guidance.seeker import load_sensor_noise_config
    base = load_sensor_noise_config()
    cfg = sweeps.noise_config_for("seeker_gain_scale", 1.5)
    assert cfg.ema_alpha_pos == pytest.approx(min(1.0, 1.5 * base.ema_alpha_pos))
    assert cfg.ema_alpha_acc == pytest.approx(1.5 * base.ema_alpha_acc)
    assert max(cfg.ema_alpha_pos, cfg.ema_alpha_vel, cfg.ema_alpha_acc) <= 1.0
    assert sweeps.noise_config_for("noise_scale", 2.0) is None


def test_new_response_sweeps_run():
    for name in ("interceptor_lag_scale", "axial_limit_scale", "jerk_limit_scale", "seeker_gain_scale"):
        rows = sweeps.run_sweep(name, PlatformType.FIXED_WING, "apn_filtered",
                                values=[sweeps.SWEEPS[name][1][0]],
                                behaviors=list(ManeuverType)[:1], episodes=1)
        assert rows[0]["episodes"] == 1


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


def test_env_takes_n_prime_from_sweep_table():
    """BUGS #35: with no N_nav the env uses the P5.11 winner; explicit values win."""
    from interceptor.guidance.config import resolve_n_prime
    for platform in PlatformType:
        for mode in ("apn_raw", "apn_filtered", "apn_rl"):
            env = UnifiedInterceptEnv(platform=platform, mode=mode, seed=0)
            assert env.N_nav == resolve_n_prime(platform, mode)
    assert UnifiedInterceptEnv(platform=PlatformType.QUADCOPTER, N_nav=3.0, seed=0).N_nav == 3.0
