"""config_loader.py -- load run-wide training settings from ``config.yaml``.

Produces a :class:`TrainConfig` that bundles:

* the ``global`` / ``rollback`` / ``on_capped`` run-wide settings,
* the observation layout (:class:`ObservationConfig`),
* the interceptor kinematic limits (:class:`InterceptorSpec`),
* the APN constants (:class:`APNSpec`),
* the effective per-stage table (``stages.yaml`` defaults merged with any
  ``config.yaml`` overrides), and
* the model definitions (when ``model.yaml`` is bound at load time).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ..envs.stage_config import (
    DEFAULT_STAGES, ObsStageConfig, StageConfig, load_stages,
)

__all__ = [
    "ConfigError",
    "ObservationConfig",
    "InterceptorSpec",
    "APNSpec",
    "TrainConfig",
    "load_config",
    "parse_range",
    "is_range",
    "FUTURE_SOURCES",
    "PREDICTORS",
    "ON_CAPPED_CHOICES",
]

FUTURE_SOURCES = ("true", "pred")
PREDICTORS = ("const_vel", "kalman")
ON_CAPPED_CHOICES = ("continue", "stop")


class ConfigError(ValueError):
    """Raised when a configuration file is malformed."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def is_range(value: Any) -> bool:
    if isinstance(value, Mapping):
        return "min" in value and "max" in value
    return isinstance(value, (list, tuple)) and len(value) == 2 and all(
        isinstance(v, (int, float)) for v in value)


def parse_range(value: Any, *, label: str = "range"):
    """Coerce a scalar / ``[lo, hi]`` / ``{min:, max:}`` into ``(lo, hi)``."""
    if value is None:
        raise ConfigError(f"{label}: value is required")
    if isinstance(value, Mapping):
        _require("min" in value and "max" in value,
                 f"{label}: range mapping needs 'min' and 'max'")
        lo, hi = float(value["min"]), float(value["max"])
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        lo, hi = float(value[0]), float(value[1])
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        lo = hi = float(value)
    else:
        raise ConfigError(f"{label}: cannot parse range from {value!r}")
    _require(lo <= hi, f"{label}: min ({lo}) must be <= max ({hi})")
    return lo, hi


@dataclass
class ObservationConfig:
    """Observation layout (plan §7)."""

    frame_dim: int = 16
    history_frames: int = 3
    history_skip: int = 2
    future_samples: int = 3
    future_skip: int = 5
    future_source: str = "pred"
    predictor: str = "const_vel"

    @property
    def m(self) -> int:
        return self.history_frames

    @property
    def p(self) -> int:
        return self.history_skip

    @property
    def n(self) -> int:
        return self.future_samples

    @property
    def q(self) -> int:
        return self.future_skip

    @property
    def has_future(self) -> bool:
        return self.future_samples > 0

    @property
    def actor_dim(self) -> int:
        return self.frame_dim * (1 + self.history_frames) + 7 * self.future_samples

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InterceptorSpec:
    """Interceptor kinematic limits (plan §6)."""

    v_max: float = 50.0
    a_max: float = 15.0
    omega_max: float = 3.0
    speed_ratio: float = 1.5
    dt: float = 0.02

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class APNSpec:
    """APN guidance constants (plan §2.3)."""

    N_prime: float = 4.0
    t_go_min: float = 0.05
    t_go_max: float = 20.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _merge(base: Mapping, override: Mapping) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _effective_obs(global_obs: ObservationConfig,
                   stage_obs: ObsStageConfig) -> ObsStageConfig:
    """Global layout wins for shape fields; the stage keeps its rl scale."""
    return ObsStageConfig(
        frame_dim=global_obs.frame_dim,
        history_frames=global_obs.history_frames,
        history_skip=global_obs.history_skip,
        future_samples=global_obs.future_samples,
        future_skip=global_obs.future_skip,
        future_source=global_obs.future_source,
        predictor=global_obs.predictor,
        rl_action_scale=float(stage_obs.rl_action_scale),
    )


@dataclass
class TrainConfig:
    """Everything a training run needs, minus the model definitions."""

    global_cfg: Dict[str, Any] = field(default_factory=dict)
    rollback: Dict[str, Any] = field(default_factory=dict)
    on_capped: str = "stop"
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    interceptor: InterceptorSpec = field(default_factory=InterceptorSpec)
    apn: APNSpec = field(default_factory=APNSpec)
    stages: Dict[str, StageConfig] = field(default_factory=dict)
    models: Dict[str, Any] = field(default_factory=dict)
    source_config_path: str = ""
    source_model_path: str = ""
    config_hash: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------- accessors
    @property
    def seed(self) -> int:
        return int(self.global_cfg.get("seed", 42))

    @property
    def device(self) -> str:
        return str(self.global_cfg.get("device", "cpu"))

    @property
    def data_dir(self) -> str:
        return str(self.global_cfg.get("data_dir", "data/runs"))

    @property
    def n_parallel_envs(self) -> int:
        return int(self.global_cfg.get("n_parallel_envs", 1))

    @property
    def checkpoint_interval_steps(self) -> int:
        return int(self.global_cfg.get("checkpoint_interval_steps", 100000))

    @property
    def tensorboard(self) -> bool:
        return bool(self.global_cfg.get("tensorboard", True))

    @property
    def execution_mode(self) -> str:
        return str(self.global_cfg.get("execution_mode", "train"))

    @property
    def rollback_cfg(self) -> Dict[str, Any]:
        return dict(self.rollback)

    def stage_numbers(self) -> List[int]:
        out = []
        for sid in self.stages:
            tail = sid.rsplit("_", 1)[-1]
            if tail.isdigit():
                out.append(int(tail))
        return sorted(out)

    def stage(self, number: int) -> StageConfig:
        sid = f"stage_{int(number)}"
        if sid not in self.stages:
            raise ConfigError(f"unknown stage {number}; known: {sorted(self.stages)}")
        return self.stages[sid]

    def model_ids(self) -> List[str]:
        return list(self.models.keys())

    def model(self, model_id: str):
        if model_id not in self.models:
            raise ConfigError(f"unknown model {model_id!r}; known: {self.model_ids()}")
        return self.models[model_id]

    def obs_history(self, model_id: str):
        m = self.model(model_id)
        return getattr(m, "obs_history", (self.observation.m, self.observation.p))

    # ------------------------------------------------------------- signatures
    def training_signature_dict(self) -> Dict[str, Any]:
        return {
            "observation": self.observation.to_dict(),
            "interceptor": self.interceptor.to_dict(),
            "apn": self.apn.to_dict(),
            "on_capped": self.on_capped,
            "stages": {sid: sc.target_type for sid, sc in sorted(self.stages.items())},
        }

    def recompute_hash(self) -> str:
        blob = json.dumps(self.training_signature_dict(), sort_keys=True)
        self.config_hash = hashlib.sha256(blob.encode()).hexdigest()[:16]
        return self.config_hash

    def bind_models(self, models: Mapping[str, Any]) -> "TrainConfig":
        self.models = dict(models)
        return self


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover
        raise ConfigError("PyYAML is required to load configuration files") from exc
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise ConfigError(f"failed to parse {path}: {exc}") from exc
    _require(isinstance(data, Mapping), f"{path}: top level must be a mapping")
    return dict(data)


def load_config(path: str = "configs/config.yaml", *,
                bind_models: Optional[Mapping[str, Any]] = None,
                model_source: str = "configs/model.yaml") -> TrainConfig:
    """Load a :class:`TrainConfig` from ``config.yaml``."""
    cfg_path = Path(path)
    _require(cfg_path.exists(), f"config file not found: {cfg_path}")
    raw = _load_yaml(cfg_path)

    global_cfg = dict(raw.get("global", {}))
    rollback = dict(raw.get("rollback", {}))
    on_capped = str(raw.get("on_capped", "stop"))
    _require(on_capped in ON_CAPPED_CHOICES,
             f"on_capped must be one of {ON_CAPPED_CHOICES}, got {on_capped!r}")

    obs_raw = dict(raw.get("observation", {}))
    observation = ObservationConfig(
        frame_dim=int(obs_raw.get("frame_dim", 16)),
        history_frames=int(obs_raw.get("history_frames", 3)),
        history_skip=int(obs_raw.get("history_skip", 2)),
        future_samples=int(obs_raw.get("future_samples", 3)),
        future_skip=int(obs_raw.get("future_skip", 5)),
        future_source=str(obs_raw.get("future_source", "pred")),
        predictor=str(obs_raw.get("predictor", "const_vel")),
    )
    _require(observation.frame_dim == 16,
             f"observation.frame_dim must be 16, got {observation.frame_dim}")
    _require(observation.future_source in FUTURE_SOURCES,
             f"future_source must be one of {FUTURE_SOURCES}")
    _require(observation.predictor in PREDICTORS,
             f"predictor must be one of {PREDICTORS}")

    int_raw = dict(raw.get("interceptor", {}))
    interceptor = InterceptorSpec(
        v_max=float(int_raw.get("v_max", 50.0)),
        a_max=float(int_raw.get("a_max", 15.0)),
        omega_max=float(int_raw.get("omega_max", 3.0)),
        speed_ratio=float(int_raw.get("speed_ratio", 1.5)),
        dt=float(int_raw.get("dt", 0.02)),
    )
    apn_raw = dict(raw.get("apn", {}))
    apn = APNSpec(
        N_prime=float(apn_raw.get("N_prime", 4.0)),
        t_go_min=float(apn_raw.get("t_go_min", 0.05)),
        t_go_max=float(apn_raw.get("t_go_max", 20.0)),
    )

    # Effective stage table: stages.yaml (or built-ins) + config overrides.
    stages = load_stages("configs/stages.yaml")
    if not stages:
        stages = dict(DEFAULT_STAGES)
    overrides = raw.get("stages", {}) or {}
    from ..envs.stage_config import stages_from_dict  # local import: avoid cycle
    for sid, ov in overrides.items():
        if sid not in stages:
            raise ConfigError(f"override for unknown stage {sid!r}")
        plain = {**asdict(stages[sid]), **dict(ov)}
        stages[sid] = stages_from_dict({sid: plain})[sid]

    # Apply the global observation layout to every stage.
    for sid, sc in stages.items():
        sc.obs = _effective_obs(observation, sc.obs)

    cfg = TrainConfig(
        global_cfg=global_cfg, rollback=rollback, on_capped=on_capped,
        observation=observation, interceptor=interceptor, apn=apn,
        stages=stages, source_config_path=str(cfg_path),
        source_model_path=str(model_source), raw=raw,
    )
    cfg.recompute_hash()

    if bind_models is None:
        try:
            from .model_loader import load_models
            if Path(model_source).exists():
                bind_models = load_models(model_source)
        except Exception:
            bind_models = None
    if bind_models is not None:
        cfg.bind_models(bind_models)
    return cfg
