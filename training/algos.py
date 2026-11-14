"""Uniform PPO / SAC / TD3 construction (plan P6.3).

The three algorithms are built from Stable-Baselines3 with the same MLP
widths, a symmetric (non-privileged) MLP policy, and the RL Baselines3 Zoo
default hyperparameters recorded in ``configs/training.yaml``.  Keeping the
constructor in one place is what makes the P6.4 comparison fair.
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
    """Build one SB3 model with the shared network and Zoo hyperparameters.

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
    return TD3(
        "MlpPolicy", env, seed=seed, device=device, verbose=0,
        tensorboard_log=tensorboard_log,
        policy_kwargs=_q_policy_kwargs(net, activation), **hp)


def algo_from_config(config: Optional[dict] = None) -> str:
    """Read the P6.5 decision, falling back to PPO until the comparison runs."""
    cfg = config or load_training_config()
    return str(cfg.get("selected_algo", "ppo")).lower()
