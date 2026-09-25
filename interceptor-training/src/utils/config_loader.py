"""config_loader.py -- run configuration loading, validation, canonical hashing.

The run YAML (``configs/default_run.yaml``) has three top-level sections:

``global:``      seed, device, data_dir, n_parallel_envs,
                 checkpoint_interval_steps, tensorboard, execution_mode,
                 rollback policy and the optional ``smoke`` block.
``stages:``      per-stage overrides merged on top of the built-in
                 :data:`~src.envs.stage_config.STAGES` table (e.g. to enable
                 ``domain_randomize``/``wind_enabled``/``ground_effect`` or
                 override ``rollback_threshold``).  Keys are ``stage_1``...
                 ``stage_8``; an absent key keeps the built-in stage config.
``configs:``     one block per training algorithm/baseline (see plan §4.3):
                 ``algo`` (PPO|SAC|TD3), ``policy``, ``net_arch``,
                 ``activation``, ``obs_history.{frames,skip}``, ``stages``
                 (list of stage numbers), ``hyperparameters`` (SB3 kwargs)
                 and optional ``action_noise_sigma`` (TD3 only).

``config_hash()`` returns a SHA-256 that identifies the *training-relevant*
content of a run config (seed, configs, stage overrides, env count, ckpt
interval).  It deliberately excludes host-specific paths (``data_dir``) so a
resumed run on a different volume still matches.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..envs.stage_config import STAGES, StageConfig

__all__ = ["RunConfig", "load_run_config", "config_hash"]

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_ALGOS = {"PPO", "SAC", "TD3"}
_POLICIES = {"MlpPolicy"}
_ACTIVATIONS = {"ReLU", "Tanh", "ELU"}
_ACTIVATION_ALIASES = {"relu": "ReLU", "tanh": "Tanh", "elu": "ELU"}


class ConfigError(ValueError):
    """Raised for invalid run configuration."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def _canonicalize_config(c: Dict[str, Any]) -> Dict[str, Any]:
    """Accept BOTH the plan §4.3 YAML schema and the internal canonical schema.

    Plan schema: ``algorithm``, ``policy_kwargs: {net_arch, activation_fn}``,
    ``obs_history: {m, s}``, ``hyperparameters`` (SB3 kwarg names).
    Canonical schema: ``algo``, top-level ``net_arch``/``activation``,
    ``obs_history: {frames, skip}``.
    """
    out = dict(c)
    if "algorithm" in out and "algo" not in out:
        out["algo"] = out["algorithm"]
    pk = out.get("policy_kwargs")
    if isinstance(pk, dict):
        if "net_arch" in pk and "net_arch" not in out:
            out["net_arch"] = pk["net_arch"]
        if "activation_fn" in pk and "activation" not in out:
            out["activation"] = pk["activation_fn"]
    # activation names are Title-cased canonically; accept plan-schema spellings
    if isinstance(out.get("activation"), str):
        out["activation"] = _ACTIVATION_ALIASES.get(out["activation"].lower(),
                                                    out["activation"])
    # a bare shared net_arch list (SB3 style) is expanded to the {pi, head}
    # dict form the validator and network builder expect
    na = out.get("net_arch")
    if isinstance(na, list) and na:
        head = "vf" if out.get("algo", "PPO") == "PPO" else "qf"
        out["net_arch"] = {"pi": list(na), head: list(na)}
    oh = out.get("obs_history")
    if isinstance(oh, dict):
        if "m" in oh and "frames" not in oh:
            oh["frames"] = oh["m"]
        if "s" in oh and "skip" not in oh:
            oh["skip"] = oh["s"]
        out["obs_history"] = oh
    return out


def _validate_global(g: Dict[str, Any]) -> None:
    for key in ("seed", "device", "data_dir", "n_parallel_envs",
                "checkpoint_interval_steps", "tensorboard",
                "execution_mode"):
        _require(key in g, f"global.{key} is required")
    _require(isinstance(g["seed"], int), "global.seed must be an int")
    _require(g["device"] in ("auto", "cpu", "cuda"),
             f"global.device must be auto|cpu|cuda, got {g['device']!r}")
    _require(isinstance(g["n_parallel_envs"], int) and g["n_parallel_envs"] >= 1,
             "global.n_parallel_envs must be >= 1")
    _require(isinstance(g["checkpoint_interval_steps"], int)
             and g["checkpoint_interval_steps"] > 0,
             "global.checkpoint_interval_steps must be > 0")
    _require(g["execution_mode"] in ("sequential",),
             f"global.execution_mode must be 'sequential', got {g['execution_mode']!r}")


def _validate_config(cname: str, c: Dict[str, Any], known_stages: List[int]) -> None:
    _require(c.get("algo") in _ALGOS, f"configs.{cname}.algo must be one of {_ALGOS}")
    _require(c.get("policy") in _POLICIES,
             f"configs.{cname}.policy must be MlpPolicy")
    net_arch = c.get("net_arch")
    algo = c.get("algo")
    # PPO uses dict(pi=[...], vf=[...]); SAC/TD3 use dict(pi=[...], qf=[...])
    head_key = "vf" if algo == "PPO" else "qf"
    _require(isinstance(net_arch, dict) and ("pi" in net_arch) and (head_key in net_arch),
             f"configs.{cname}.net_arch must be a dict with 'pi' and "
             f"'{head_key}' lists (got keys {sorted(net_arch) if isinstance(net_arch, dict) else None})")
    for k in ("pi", head_key):
        _require(isinstance(net_arch[k], list) and all(isinstance(h, int) and h > 0
                                                       for h in net_arch[k]),
                 f"configs.{cname}.net_arch.{k} must be a list of positive ints")
    _require(c.get("activation") in _ACTIVATIONS,
             f"configs.{cname}.activation must be one of {_ACTIVATIONS}")
    obs = c.get("obs_history", {})
    frames, skip = obs.get("frames", 0), obs.get("skip", 1)
    _require(isinstance(frames, int) and frames >= 0 and isinstance(skip, int) and skip >= 1,
             f"configs.{cname}.obs_history needs int frames>=0 and int skip>=1")
    stages = c.get("stages")
    _require(isinstance(stages, list) and stages,
             f"configs.{cname}.stages must be a non-empty list")
    for s in stages:
        _require(int(s) in known_stages,
                 f"configs.{cname}.stages contains unknown stage {s!r}")
    _require(isinstance(c.get("hyperparameters", {}), dict),
             f"configs.{cname}.hyperparameters must be a dict")


def _coerce_stage_overrides(stage: StageConfig, overrides: Dict[str, Any]) -> StageConfig:
    """Deep-merge a stage override dict onto a StageConfig (in place copy).

    Both the plan-stage YAML key names and the internal field names are
    accepted: ``radius`` -> ``world_radius``, ``success_threshold`` ->
    ``threshold``; everything else maps 1:1 onto the StageConfig fields or
    the ``reward`` / ``spawn`` / ``intercept`` / ``obs`` sub-dataclasses.
    """
    out = copy.deepcopy(stage)
    renames = {"radius": "world_radius", "success_threshold": "threshold"}
    scalar_keys = (
        "name", "target_type", "target_visible", "world_radius",
        "max_episode_steps", "max_training_steps", "success_metric",
        "threshold", "window", "success_rate", "rollback_threshold",
        "domain_randomize", "wind_enabled", "ground_effect", "lookahead_enabled",
    )
    for key, value in overrides.items():
        k = renames.get(key, key)
        if k in scalar_keys:
            setattr(out, k, value)
    if "reward" in overrides and isinstance(overrides["reward"], dict):
        for k, v in overrides["reward"].items():
            setattr(out.reward, k, v)
    if "spawn" in overrides and isinstance(overrides["spawn"], dict):
        for k, v in overrides["spawn"].items():
            setattr(out.spawn, k, v)
    if "intercept" in overrides and isinstance(overrides["intercept"], dict):
        for k, v in overrides["intercept"].items():
            setattr(out.intercept, k, v)
    if "obs" in overrides and isinstance(overrides["obs"], dict):
        for k, v in overrides["obs"].items():
            setattr(out.obs, k, v)
    return out


# ---------------------------------------------------------------------------
# RunConfig
# ---------------------------------------------------------------------------

class RunConfig:
    """Validated run configuration with typed accessors."""

    def __init__(self, raw: Dict[str, Any], source: str = "<yaml>") -> None:
        _require("global" in raw, f"{source}: top-level 'global' section required")
        _require("configs" in raw, f"{source}: top-level 'configs' section required")
        self.source = source
        self.raw = raw
        self.global_cfg = dict(raw["global"])
        self.configs = dict(raw["configs"])

        # merge the default global rollback policy
        roll = self.global_cfg.setdefault("rollback", {})
        roll.setdefault("max_attempts", 2)
        roll.setdefault("retry_budget_scale", 0.5)
        roll.setdefault("threshold_scale", 0.5)
        roll.setdefault("min_steps_scale", 0.25)

        _validate_global(self.global_cfg)

        canonical = {cname: _canonicalize_config(c) for cname, c in self.configs.items()}
        self.configs = canonical

        known_stages = sorted(
            int(sid.split("_")[1]) for sid in STAGES if sid.startswith("stage_")
        )
        for cname, c in self.configs.items():
            _validate_config(cname, c, known_stages)

        # merged per-stage configs
        raw_stages = raw.get("stages", {})
        self.stages: Dict[str, StageConfig] = {}
        for sid, stage in STAGES.items():
            ov = raw_stages.get(sid, {})
            self.stages[sid] = _coerce_stage_overrides(stage, ov)
        self._stage_numbers = known_stages

    # -- accessors ----------------------------------------------------------
    @property
    def seed(self) -> int:
        return int(self.global_cfg["seed"])

    @property
    def device(self) -> str:
        return str(self.global_cfg["device"])

    @property
    def data_dir(self) -> Path:
        return Path(self.global_cfg["data_dir"])

    @property
    def n_parallel_envs(self) -> int:
        return int(self.global_cfg["n_parallel_envs"])

    @property
    def checkpoint_interval_steps(self) -> int:
        return int(self.global_cfg["checkpoint_interval_steps"])

    @property
    def tensorboard(self) -> bool:
        return bool(self.global_cfg.get("tensorboard", True))

    @property
    def rollback_cfg(self) -> Dict[str, Any]:
        return dict(self.global_cfg["rollback"])

    def smoke_block(self) -> Optional[Dict[str, Any]]:
        sm = self.global_cfg.get("smoke")
        return dict(sm) if sm else None

    def stage(self, number: int) -> StageConfig:
        return self.stages[f"stage_{number}"]

    def stage_numbers(self) -> List[int]:
        return list(self._stage_numbers)

    def config_names(self) -> List[str]:
        return list(self.configs.keys())

    def config(self, name: str) -> Dict[str, Any]:
        return self.configs[name]

    def config_hyperparams(self, name: str) -> Dict[str, Any]:
        return dict(self.configs[name].get("hyperparameters", {}))

    def config_stages(self, name: str) -> List[int]:
        return [int(s) for s in self.configs[name]["stages"]]

    def config_obs_history(self, name: str) -> Dict[str, int]:
        oh = self.configs[name].get("obs_history", {})
        return {"frames": int(oh.get("frames", 0)), "skip": int(oh.get("skip", 1))}

    # -- resume-side -----------------------------------------------------------------
    def training_signature_dict(self) -> Dict[str, Any]:
        """The subset of the config that defines a training run's identity.

        Includes the FULL effective per-stage config (reward weights, spawn
        ranges, thresholds, ...) so any training-relevant edit invalidates
        the resume hash.
        """
        return {
            "seed": self.seed,
            "n_parallel_envs": self.n_parallel_envs,
            "checkpoint_interval_steps": self.checkpoint_interval_steps,
            "tensorboard": self.tensorboard,
            "rollback": self.rollback_cfg,
            "stages": {
                sid: dataclasses.asdict(s)
                for sid, s in self.stages.items()
            },
            "configs": self.configs,
        }


def config_hash(cfg: RunConfig) -> str:
    """Canonical SHA-256 of the training-relevant run configuration."""
    payload = json.dumps(
        cfg.training_signature_dict(),
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_run_config(path: str) -> RunConfig:
    """Load and validate a run YAML from disk."""
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {path}")
    with open(p, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: YAML root must be a mapping")
    return RunConfig(raw, source=str(p))