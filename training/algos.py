"""Uniform PPO / SAC / TD3 construction (plan P6.3).

The three algorithms are built from Stable-Baselines3 with the same MLP
widths, a symmetric (non-privileged) MLP policy, and the hyperparameters
recorded in ``configs/training.yaml`` (SB3 2.9.0 library defaults, which match
PPO Table 3 / SAC Table 1 / TD3 Sec. 6.1; TD3 learning rate 3e-4 and the shared
[256, 256, 128] network are project choices; each line of that file names its
source).  Keeping the constructor in one place is what makes the P6.4
comparison fair.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "training.yaml"

__all__ = ["ALGOS", "load_training_config", "build_model", "algo_from_config"]

ALGOS = ("ppo", "sac", "td3")


def load_training_config(path: Optional[Path] = None) -> dict:
    path = Path(path) if path is not None else CONFIG_PATH
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _activation(name: str):
    import torch.nn as nn
    table = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}
    try:
        return table[str(name).lower()]
    except KeyError as exc:
        raise ValueError(f"unknown activation {name!r}") from exc


def _policy_kwargs(net: dict, activation: str) -> dict:
    fn = _activation(activation)
    return {
        "net_arch": {"pi": list(net["pi"]), "vf": list(net["vf"])},
        "activation_fn": fn,
    }


def _q_policy_kwargs(net: dict, activation: str) -> dict:
    fn = _activation(activation)
    return {
        "net_arch": {"pi": list(net["pi"]), "qf": list(net["vf"])},
        "activation_fn": fn,
    }


def build_model(
    algo: str,
    env: Any,
    *,
    seed: int = 0,
    config: Optional[dict] = None,
    tensorboard_log: Optional[str] = None,
    device: str = "cpu",
    asymmetric: bool = False,
    actor_obs_dim: Optional[int] = None,
):
    """Build one SB3 model with the shared network and configured hyperparameters.

    ``asymmetric=True`` selects the privileged-critic PPO policy (P7.3); the
    actor sees the first ``actor_obs_dim`` columns, the critic all of them.
    """
    key = str(algo).lower()
    if key not in ALGOS:
        raise ValueError(f"algo must be one of {ALGOS}, got {algo!r}")
    cfg = config or load_training_config()
    net = cfg["network"]
    activation = net.get("activation", "relu")
    hp = dict(cfg["algorithms"][key])

    from stable_baselines3 import PPO, SAC, TD3

    if asymmetric:
        if key != "ppo":
            raise ValueError("the asymmetric critic is only implemented for PPO")
        if not actor_obs_dim:
            raise ValueError("actor_obs_dim is required for an asymmetric policy")
        from interceptor.training.asymmetric_policy import AsymmetricActorCriticPolicy
        kwargs = _policy_kwargs(net, activation)
        kwargs["actor_obs_dim"] = int(actor_obs_dim)
        return PPO(
            AsymmetricActorCriticPolicy, env, seed=seed, device=device, verbose=0,
            tensorboard_log=tensorboard_log, policy_kwargs=kwargs, **hp)

    if key == "ppo":
        return PPO(
            "MlpPolicy", env, seed=seed, device=device, verbose=0,
            tensorboard_log=tensorboard_log,
            policy_kwargs=_policy_kwargs(net, activation), **hp)
    if key == "sac":
        return SAC(
            "MlpPolicy", env, seed=seed, device=device, verbose=0,
            tensorboard_log=tensorboard_log,
            policy_kwargs=_q_policy_kwargs(net, activation), **hp)
    # TD3 explores with Gaussian action noise N(0, sigma * max_action) (Fujimoto et al.
    # 2018, Sec. 6.1; sfujim/TD3 main.py L44 and L122). SB3's TD3 adds no noise unless an
    # ActionNoise object is passed (BUGS #31), so it is built here from the config.
    noise_std = hp.pop("action_noise_std", None)
    if noise_std:
        import numpy as np
        from stable_baselines3.common.noise import NormalActionNoise
        space = env.action_space
        half_range = (np.asarray(space.high, dtype=np.float64) - np.asarray(space.low, dtype=np.float64)) / 2.0
        hp["action_noise"] = NormalActionNoise(
            mean=np.zeros(space.shape), sigma=float(noise_std) * half_range)
    return TD3(
        "MlpPolicy", env, seed=seed, device=device, verbose=0,
        tensorboard_log=tensorboard_log,
        policy_kwargs=_q_policy_kwargs(net, activation), **hp)


def algo_from_config(config: Optional[dict] = None) -> str:
    """Read the P6.5 decision, falling back to PPO until the comparison runs."""
    cfg = config or load_training_config()
    return str(cfg.get("selected_algo", "ppo")).lower()
