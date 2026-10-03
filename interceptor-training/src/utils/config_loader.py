"""config_loader.py -- load training settings from a `config.yaml` file (plan A-2/A-4).

This module owns everything that is NOT model architecture:

* ``global``   -- run-wide settings (seed, device, data_dir, ...)
* ``rollback`` -- curriculum rollback policy
* ``on_capped``-- 'continue' | 'stop'
* ``observation`` -- history/future stacking layout and the future source
* ``target_alt``  -- Stage-1 hover altitude (number or ``{min:, max:}``)
* ``stages``      -- per-stage overrides merged onto the built-in STAGES table

Model architecture lives in :mod:`src.utils.model_loader`.  ``config_hash``
spans BOTH files (D-20) so nothing that affects training can silently change
underneath a resumed run.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from ..envs.stage_config import STAGES, StageConfig

__all__ = [
    "ConfigError",
    "ObservationConfig",
    "TrainConfig",
    "config_hash",
    "load_config",
    "parse_range",
    "FUTURE_SOURCES",
    "PREDICTORS",
]

#: Accepted ``observation.future_source`` values.
FUTURE_SOURCES = ("true", "pred")

#: Accepted ``observation.predictor`` values (plan F-4).
PREDICTORS = ("const_vel", "linear_ridge")

#: Accepted ``on_capped`` values (D-30).
ON_CAPPED_CHOICES = ("continue", "stop")

_ROLLBACK_DEFAULTS: Dict[str, float] = {
    "max_attempts": 2,
    "retry_budget_scale": 0.5,
    "threshold_scale": 0.5,
    "min_steps_scale": 0.25,
}

_GLOBAL_KEYS = (
    "seed",
    "device",
    "data_dir",
    "n_parallel_envs",
    "checkpoint_interval_steps",
    "tensorboard",
    "execution_mode",
)

#: Stage attributes that may be overridden from the ``stages:`` block.
_STAGE_SCALAR_KEYS = (
    "name",
    "target_type",
    "target_visible",
    "world_radius",
    "episode_length_s",
    "max_training_steps",
    "success_metric",
    "threshold",
    "window",
    "success_rate",
    "rollback_threshold",
    "domain_randomize",
    "wind_enabled",
    "ground_effect",
)

#: Nested stage sub-blocks that map 1:1 onto a StageConfig attribute.
_STAGE_SUBBLOCKS = ("reward", "spawn", "intercept", "obs")

#: Top-level shorthands that really live inside the ``spawn`` sub-block, so
#: ``stages.stage_5.path_speed_cap`` and ``stages.stage_5.spawn.path_speed_cap``
#: are equivalent.
_STAGE_SPAWN_ALIASES = ("path_end_distance", "path_speed_cap", "path_shape")


class ConfigError(ValueError):
    """Raised for any malformed or inconsistent configuration."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


# ---------------------------------------------------------------------------
# A-6 / D-48: "number or {min:, max:} range" parsing
# ---------------------------------------------------------------------------

def parse_range(value: Any, name: str) -> Tuple[float, float]:
    """Parse a fixed value or a ``{min:, max:}`` mapping into a ``(lo, hi)`` tuple.

    A bare number collapses to ``(v, v)`` so callers never need to special-case
    the fixed form.
    """
    if isinstance(value, bool):
        raise ConfigError(f"{name}: expected number or {{min:, max:}}, got bool")
    if isinstance(value, (int, float)):
        return (float(value), float(value))
    if isinstance(value, Mapping) and "min" in value and "max" in value:
        try:
            lo, hi = float(value["min"]), float(value["max"])
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{name}: min/max must be numbers, got {value!r}") from exc
        if lo > hi:
            raise ConfigError(f"{name}: min ({lo}) > max ({hi})")
        return (lo, hi)
    raise ConfigError(f"{name}: expected number or {{min:, max:}}, got {value!r}")


def is_range(value: Any) -> bool:
    """True when ``value`` is a ``{min:, max:}`` mapping (as opposed to fixed)."""
    return isinstance(value, Mapping) and "min" in value and "max" in value


# ---------------------------------------------------------------------------
# observation section (D-16, D-18, D-19, D-38)
# ---------------------------------------------------------------------------

@dataclass
class ObservationConfig:
    """History/future stacking layout.

    ``m = history_frames``, ``p = history_skip``, ``n = future_samples``,
    ``q = future_skip``.  The actor observation is
    ``frame_dim * (1 + m) + 7 * n``.
    """

    history_frames: int = 0
    history_skip: int = 1
    future_samples: int = 0
    future_skip: int = 1
    future_source: str = "true"
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "history_frames": self.history_frames,
            "history_skip": self.history_skip,
            "future_samples": self.future_samples,
            "future_skip": self.future_skip,
            "future_source": self.future_source,
            "predictor": self.predictor,
        }


def _parse_observation(raw: Any, source: str) -> ObservationConfig:
    if raw is None:
        raw = {}
    _require(
        isinstance(raw, Mapping),
        f"{source}: 'observation' must be a mapping, got {type(raw).__name__}",
    )
    o = raw or {}

    def _int(key: str, default: int, lo: int) -> int:
        v = o.get(key, default)
        _require(
            isinstance(v, int) and not isinstance(v, bool),
            f"{source}: observation.{key} must be an int, got {v!r}",
        )
        _require(v >= lo, f"{source}: observation.{key} must be >= {lo}, got {v}")
        return int(v)

    m = _int("history_frames", 0, 0)
    p = _int("history_skip", 1, 1)
    n = _int("future_samples", 0, 0)
    q = _int("future_skip", 1, 1)

    fs = str(o.get("future_source", "true")).strip().lower()
    _require(
        fs in FUTURE_SOURCES,
        f"{source}: observation.future_source must be one of {list(FUTURE_SOURCES)}, got {fs!r}",
    )

    pred = str(o.get("predictor", "const_vel")).strip().lower()
    _require(
        pred in PREDICTORS,
        f"{source}: observation.predictor must be one of {list(PREDICTORS)}, got {pred!r}",
    )
    if n == 0:
        # No future frames -> the predictor is never used; accept anything valid
        # but default it so `train_model` never has to special-case None.
        pred = "const_vel"

    return ObservationConfig(
        history_frames=m, history_skip=p, future_samples=n, future_skip=q,
        future_source=fs, predictor=pred,
    )


# ---------------------------------------------------------------------------
# TrainConfig
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Validated training settings, optionally bound to a set of models."""

    global_cfg: Dict[str, Any] = field(default_factory=dict)
    rollback: Dict[str, Any] = field(default_factory=lambda: dict(_ROLLBACK_DEFAULTS))
    on_capped: str = "continue"
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    target_alt: Any = 5.0
    stage_overrides: Dict[str, Any] = field(default_factory=dict)
    source_config_path: str = "<yaml>"
    source_model_path: str = ""
    models: Dict[str, Any] = field(default_factory=dict)
    stages: Dict[str, StageConfig] = field(default_factory=dict)
    config_hash: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)
    # private
    _config_only_hash: str = field(default="", repr=False)

    # -- accessors mirroring the old RunConfig surface -------------------
    @property
    def source(self) -> str:
        return self.source_config_path

    @property
    def seed(self) -> int:
        return int(self.global_cfg["seed"])

    @property
    def device(self) -> str:
        return str(self.global_cfg["device"])

    @property
    def data_dir(self) -> Path:
        return Path(str(self.global_cfg["data_dir"]))

    @property
    def n_parallel_envs(self) -> int:
        return int(self.global_cfg["n_parallel_envs"])

    @property
    def checkpoint_interval_steps(self) -> int:
        return int(self.global_cfg["checkpoint_interval_steps"])

    @property
    def tensorboard(self) -> bool:
        return bool(self.global_cfg["tensorboard"])

    @property
    def rollback_cfg(self) -> Dict[str, Any]:
        return self.rollback

    @property
    def target_alt_range(self) -> Tuple[float, float]:
        return parse_range(self.target_alt, "target_alt")

    @property
    def target_alt_is_range(self) -> bool:
        return is_range(self.target_alt)

    def stage_numbers(self) -> List[int]:
        return sorted(int(sid.split("_")[1]) for sid in self.stages if sid.startswith("stage_"))

    def stage(self, number: int) -> StageConfig:
        sid = f"stage_{int(number)}"
        if sid not in self.stages:
            raise ConfigError(f"{self.source_config_path}: unknown stage {number}")
        return self.stages[sid]

    def model_ids(self) -> List[str]:
        return list(self.models.keys())

    def model(self, model_id: str):
        if model_id not in self.models:
            raise ConfigError(
                f"{self.source_model_path or self.source_config_path}: "
                f"unknown model_id {model_id!r}; available: {sorted(self.models)}"
            )
        return self.models[model_id]

    def obs_history(self, model_id: str) -> Dict[str, int]:
        return dict(self.model(model_id).obs_history)

    # -- hashing ---------------------------------------------------------
    def training_signature_dict(self) -> Dict[str, Any]:
        from .model_loader import models_signature

        return {
            "global": {k: self.global_cfg[k] for k in _GLOBAL_KEYS},
            "rollback": {k: self.rollback[k] for k in sorted(self.rollback)},
            "on_capped": self.on_capped,
            "observation": self.observation.to_dict(),
            "target_alt": self.target_alt,
            "stages": {
                sid: _dataclass_plain(sc) for sid, sc in sorted(self.stages.items())
            },
            "models": models_signature(self.models),
        }

    def bind_models(self, models: Mapping[str, Any], model_source: str = "") -> "TrainConfig":
        """Attach model definitions and recompute the combined hash (D-20)."""
        self.models = dict(models)
        if model_source:
            self.source_model_path = str(model_source)
        if not self.config_hash or self.models or not self._config_only_hash:
            self._config_only_hash = _sha256(
                json.dumps(self.training_signature_dict(), sort_keys=True,
                           default=str, separators=(",", ":"))
            )
            self.config_hash = self._config_only_hash
        return self

    def recompute_hash(self) -> str:
        self._config_only_hash = _sha256(
            json.dumps(self.training_signature_dict(), sort_keys=True,
                       default=str, separators=(",", ":"))
        )
        self.config_hash = self._config_only_hash
        return self.config_hash


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _dataclass_plain(obj: Any) -> Any:
    """Recursively convert dataclasses / paths into JSON-safe plain data."""
    import dataclasses

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _dataclass_plain(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, Mapping):
        return {k: _dataclass_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_dataclass_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def config_hash(cfg: TrainConfig) -> str:
    """SHA-256 spanning the config file AND the model definitions (D-20)."""
    return cfg.config_hash or cfg.recompute_hash()


# ---------------------------------------------------------------------------
# validation + loading
# ---------------------------------------------------------------------------

def _validate_global(g: Any, source: str) -> Dict[str, Any]:
    _require(
        isinstance(g, Mapping),
        f"{source}: 'global' section is required and must be a mapping",
    )
    out = dict(g)
    for k in _GLOBAL_KEYS:
        _require(k in out, f"{source}: global.{k} is required")

    _require(
        isinstance(out["seed"], int) and not isinstance(out["seed"], bool),
        f"{source}: global.seed must be an int, got {out['seed']!r}",
    )
    dev = str(out["device"])
    _require(
        dev in ("auto", "cpu", "cuda"),
        f"{source}: global.device must be auto|cpu|cuda, got {dev!r}",
    )
    _require(bool(str(out["data_dir"]).strip()), f"{source}: global.data_dir must be non-empty")
    n_env = out["n_parallel_envs"]
    _require(
        isinstance(n_env, int) and not isinstance(n_env, bool) and n_env >= 1,
        f"{source}: global.n_parallel_envs must be an int >= 1, got {n_env!r}",
    )
    ci = out["checkpoint_interval_steps"]
    _require(
        isinstance(ci, int) and not isinstance(ci, bool) and ci > 0,
        f"{source}: global.checkpoint_interval_steps must be a positive int, got {ci!r}",
    )
    _require(
        isinstance(out["tensorboard"], bool),
        f"{source}: global.tensorboard must be a bool",
    )
    _require(
        str(out["execution_mode"]) == "sequential",
        f"{source}: global.execution_mode must be 'sequential', "
        f"got {out['execution_mode']!r}",
    )
    return out


def _validate_rollback(r: Any, source: str) -> Dict[str, Any]:
    out = dict(_ROLLBACK_DEFAULTS)
    if r is None:
        return out
    _require(
        isinstance(r, Mapping),
        f"{source}: 'rollback' section must be a mapping, got {type(r).__name__}",
    )
    for k, v in r.items():
        _require(k in _ROLLBACK_DEFAULTS, f"{source}: unknown rollback key {k!r}")
        out[k] = v

    ma = out["max_attempts"]
    _require(
        isinstance(ma, int) and not isinstance(ma, bool) and ma >= 0,
        f"{source}: rollback.max_attempts must be a non-negative int, got {ma!r}",
    )
    for k in ("retry_budget_scale", "threshold_scale", "min_steps_scale"):
        v = out[k]
        _require(
            isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= float(v) <= 1.0,
            f"{source}: rollback.{k} must be a number in [0, 1], got {v!r}",
        )
        out[k] = float(v)
    out["max_attempts"] = int(ma)
    return out


# Spawn fields that accept either a scalar or a two-element range.  They are
# normalised to ``(lo, hi)`` tuples so YAML lists and mappings behave the same.
_STAGE_RANGE_SPAWN_FIELDS = (
    "path_end_distance",
    "path_speed_cap",
    "target_distance_range",
    "waypoint_distance_range",
    "speed_range",
    "altitude_range",
)


def _assign_stage_field(obj, key: str, val: Any, label: str) -> None:
    """Set one override field, normalising spawn ranges to ``(lo, hi)``."""
    if isinstance(obj, SpawnConfig) and key in _STAGE_RANGE_SPAWN_FIELDS:
        val = parse_range(val, label)
    setattr(obj, key, val)


def _coerce_stage_overrides(stage: StageConfig, overrides: Mapping[str, Any],
                            *, sid: str, source: str) -> StageConfig:
    """Merge a ``stages:`` override block onto a built-in StageConfig."""
    sc = copy.deepcopy(stage)
    if not overrides:
        return sc
    if not isinstance(overrides, Mapping):
        raise ConfigError(f"{source}: stages.{sid} override must be a mapping")

    for key, val in overrides.items():
        if key == "max_episode_steps":
            # Legacy step count -> seconds, so old configs keep working.
            _require(
                isinstance(val, int) and not isinstance(val, bool) and val > 0,
                f"{source}: stages.{sid}.max_episode_steps must be a positive int",
            )
            from ..physics.constants import PH_DT

            sc.episode_length_s = float(val) * float(PH_DT)
        elif key == "radius":
            sc.world_radius = val
        elif key == "success_threshold":
            sc.threshold = val
        elif key in _STAGE_SPAWN_ALIASES:
            _assign_stage_field(sc.spawn, key, val, f"{source}: stages.{sid}.{key}")
        elif key in _STAGE_SUBBLOCKS:
            sub = getattr(sc, key)
            _require(
                isinstance(val, Mapping),
                f"{source}: stages.{sid}.{key} must be a mapping",
            )
            for k, v in val.items():
                _require(hasattr(sub, k), f"{source}: stages.{sid}.{key}.{k} is not a field")
                _assign_stage_field(sub, k, v, f"{source}: stages.{sid}.{key}.{k}")
        elif key in _STAGE_SCALAR_KEYS:
            setattr(sc, key, val)
        else:
            raise ConfigError(f"{source}: stages.{sid} has unknown key {key!r}")
    sc.validate(f"{source}: stages.{sid}")
    return sc


def load_config(path: Union[str, Path], *, bind_models: Optional[Mapping[str, Any]] = None,
                model_source: str = "") -> TrainConfig:
    """Load and validate a ``config.yaml``.

    When ``bind_models`` is given the model definitions are attached and the
    returned ``config_hash`` spans both files.
    """
    import yaml

    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - malformed YAML
        raise ConfigError(f"{p}: invalid YAML -- {exc}") from exc
    _require(isinstance(raw, Mapping), f"{p}: top level must be a mapping")

    source = str(p)
    cfg = TrainConfig(
        global_cfg=_validate_global(raw.get("global"), source),
        rollback=_validate_rollback(raw.get("rollback"), source),
        on_capped=str(raw.get("on_capped", "continue")).strip().lower(),
        observation=_parse_observation(raw.get("observation"), source),
        target_alt=raw.get("target_alt", 5.0),
        stage_overrides=dict(raw.get("stages") or {}),
        source_config_path=source,
        raw=dict(raw),
    )
    _require(
        cfg.on_capped in ON_CAPPED_CHOICES,
        f"{source}: on_capped must be one of {list(ON_CAPPED_CHOICES)}, got {cfg.on_capped!r}",
    )
    parse_range(cfg.target_alt, f"{source}: target_alt")  # validate now, fail fast

    for sid, overrides in cfg.stage_overrides.items():
        _require(
            sid in STAGES,
            f"{source}: stages.{sid} is not a known stage "
            f"(known: {sorted(STAGES, key=lambda s: int(s.split('_')[1]))})",
        )

    cfg.stages = {
        sid: _coerce_stage_overrides(STAGES[sid], cfg.stage_overrides.get(sid, {}),
                                     sid=sid, source=source)
        for sid in STAGES
    }
    cfg.recompute_hash()
    if bind_models is not None:
        cfg.bind_models(bind_models, model_source=model_source)
    return cfg