"""P6.3: one constructor builds PPO/SAC/TD3 with the same network."""

from __future__ import annotations

import pytest

from interceptor.fc.env import FlightControllerEnv
from interceptor.guidance.env import UnifiedInterceptEnv
from interceptor.guidance.maneuvers import PlatformType
from training.algos import ALGOS, build_model, load_training_config


@pytest.fixture(scope="module")
def cfg():
    return load_training_config()


@pytest.mark.parametrize("algo", ALGOS)
def test_cited_hyperparameters_are_recorded(cfg, algo):
    hp = cfg["algorithms"][algo]
    assert hp["learning_rate"] == pytest.approx(3e-4)
    assert hp["gamma"] == pytest.approx(0.99)
    assert cfg["network"]["pi"] == [256, 256, 128]
    assert cfg["network"]["vf"] == [256, 256, 128]


@pytest.mark.parametrize("algo", ALGOS)
def test_values_equal_sb3_defaults_except_project_choices(cfg, algo):
    """configs/training.yaml cites SB3 2.9.0 defaults line by line; only the
    TD3 learning rate (PROJECT_CHOICE) and TD3 exploration noise (paper value,
    SB3 has none) may differ from the installed library."""
    import inspect
    from stable_baselines3 import PPO, SAC, TD3
    cls = {"ppo": PPO, "sac": SAC, "td3": TD3}[algo]
    defaults = {k: v.default for k, v in inspect.signature(cls.__init__).parameters.items()}
    allowed = {("td3", "learning_rate"), ("td3", "action_noise_std")}
    for key, value in cfg["algorithms"][algo].items():
        if (algo, key) in allowed:
            continue
        assert defaults[key] == value, f"{algo}.{key}: yaml {value} != SB3 {defaults[key]}"


def test_td3_gets_gaussian_exploration_noise(cfg):
    """Fujimoto et al. (2018) Sec. 6.1: exploration noise N(0, 0.1) (BUGS #31)."""
    import numpy as np
    from stable_baselines3.common.noise import NormalActionNoise
    env = FlightControllerEnv(domain_randomization=False, wind=False)
    model = build_model("td3", env, seed=0, config=cfg)
    assert isinstance(model.action_noise, NormalActionNoise)
    half = (env.action_space.high - env.action_space.low) / 2.0
    np.testing.assert_allclose(model.action_noise._sigma, 0.1 * half)
    assert "action_noise_std" in cfg["algorithms"]["td3"]  # build_model must not mutate cfg


@pytest.mark.parametrize("algo", ALGOS)
def test_builds_with_matching_network(cfg, algo):
    env = FlightControllerEnv(domain_randomization=False, wind=False)
    model = build_model(algo, env, seed=0, config=cfg)
    arch = model.policy.net_arch
    assert list(arch["pi"]) == cfg["network"]["pi"]
    q_key = "vf" if "vf" in arch else "qf"
    assert list(arch[q_key]) == cfg["network"]["vf"]


def test_asymmetric_ppo_policy(cfg):
    from interceptor.training.asymmetric_policy import AsymmetricActorCriticPolicy
    env = FlightControllerEnv(domain_randomization=False, wind=False)
    model = build_model("ppo", env, seed=0, config=cfg, asymmetric=True,
                        actor_obs_dim=33)
    assert isinstance(model.policy, AsymmetricActorCriticPolicy)
    assert model.policy.actor_obs_dim == 33
    assert model.observation_space.shape == (49,)


def test_asymmetric_checkpoint_round_trip(tmp_path, cfg):
    from stable_baselines3 import PPO
    env = FlightControllerEnv(domain_randomization=False, wind=False)
    model = build_model("ppo", env, seed=0, config=cfg, asymmetric=True,
                        actor_obs_dim=33)
    path = tmp_path / "asym.zip"
    model.save(path)
    loaded = PPO.load(path, device="cpu")
    assert loaded.policy.actor_obs_dim == 33
@pytest.mark.parametrize("algo", ALGOS)
@pytest.mark.parametrize("task", ["fc", "guidance"])
@pytest.mark.slow
def test_trains_2000_steps_on_both_environments(algo, task):
    """Acceptance for P6.3: every algorithm trains on every task."""
    if task == "fc":
        env = FlightControllerEnv(domain_randomization=False, wind=False)
    else:
        env = UnifiedInterceptEnv(platform=PlatformType.QUADCOPTER, mode="apn_rl",
                                  target_set="static", N_nav=5.0, seed=0)
    model = build_model(algo, env, seed=0)
    model.learn(total_timesteps=2000)
    assert model.num_timesteps >= 2000
