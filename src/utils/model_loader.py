"""model_loader.py -- load model definitions from ``model.yaml`` (plan §12).

A :class:`ModelDef` is the architecture-only description of a policy: algorithm,
policy class, MLP widths, activation, the obs-history stacking it expects, the
SB3 hyperparameters, and the privileged-critic field selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .config_loader import ConfigError, _require

__all__ = [
    "ModelDef",
    "load_models",
    "model_def_from_dict",
    "models_signature",
    "ACTIVATIONS",
]

ACTIVATIONS: Dict[str, str] = {
    "relu": "ReLU", "tanh": "Tanh", "elu": "ELU", "gelu": "GELU",
}

_SUPPORTED_ALGOS = ("PPO",)


@dataclass
class ModelDef:
    """Architecture description of one model (one ``model_id``)."""

    model_id: str
    algo: str = "PPO"
    policy: str = "MlpPolicy"
    net_arch: Dict[str, List[int]] = field(
        default_factory=lambda: {"pi": [256, 256, 128], "vf": [256, 256, 128]})
    activation: str = "ReLU"
    obs_history: Tuple[int, int] = (3, 2)     # (frames, skip)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    privileged_critic: Tuple[str, ...] = ()

    # ------------------------------------------------------------- helpers
    @property
    def uses_privileged_critic(self) -> bool:
        return bool(self.privileged_critic)

    def activation_name(self) -> str:
        return ACTIVATIONS.get(self.activation.lower(), self.activation)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algo": self.algo, "policy": self.policy,
            "net_arch": self.net_arch, "activation": self.activation,
            "obs_history": {"frames": self.obs_history[0],
                            "skip": self.obs_history[1]},
            "hyperparameters": dict(self.hyperparameters),
            "privileged_critic": list(self.privileged_critic),
        }


def _as_net_arch(raw: Any) -> Dict[str, List[int]]:
    if raw is None:
        return {"pi": [256, 256, 128], "vf": [256, 256, 128]}
    if isinstance(raw, Mapping):
        return {
            "pi": list(raw.get("pi", raw.get("actor", [256, 256, 128]))),
            "vf": list(raw.get("vf", raw.get("qf", raw.get("critic", [256, 256, 128])))),
        }
    if isinstance(raw, (list, tuple)):
        width = [int(x) for x in raw]
        return {"pi": width, "vf": list(width)}
    raise ConfigError(f"net_arch: unsupported value {raw!r}")


def _as_history(raw: Any) -> Tuple[int, int]:
    if raw is None:
        return (3, 2)
    if isinstance(raw, Mapping):
        return (int(raw.get("frames", 3)), int(raw.get("skip", 2)))
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return (int(raw[0]), int(raw[1]))
    raise ConfigError(f"obs_history: unsupported value {raw!r}")


def model_def_from_dict(model_id: str, data: Mapping[str, Any]) -> ModelDef:
    _require(isinstance(data, Mapping), f"model {model_id!r}: must be a mapping")
    algo = str(data.get("algo", "PPO")).upper()
    _require(algo in _SUPPORTED_ALGOS,
             f"model {model_id!r}: algo must be one of {_SUPPORTED_ALGOS}, got {algo!r}")
    activation = str(data.get("activation", "ReLU"))
    _require(activation.lower() in ACTIVATIONS,
             f"model {model_id!r}: activation must be one of {sorted(ACTIVATIONS)}")
    privileged = tuple(data.get("privileged_critic", ()) or ())
    if privileged:
        from ..envs.obs_builder import PRIVILEGED_FIELDS
        for name in privileged:
            _require(name in PRIVILEGED_FIELDS,
                     f"model {model_id!r}: unknown privileged field {name!r}; "
                     f"known: {sorted(PRIVILEGED_FIELDS)}")
    return ModelDef(
        model_id=model_id, algo=algo,
        policy=str(data.get("policy", "MlpPolicy")),
        net_arch=_as_net_arch(data.get("net_arch")),
        activation=ACTIVATIONS.get(activation.lower(), activation),
        obs_history=_as_history(data.get("obs_history")),
        hyperparameters=dict(data.get("hyperparameters", {}) or {}),
        privileged_critic=privileged,
    )


def load_models(path: str = "configs/model.yaml") -> Dict[str, ModelDef]:
    """Load every model definition from a ``model.yaml`` file."""
    p = Path(path)
    _require(p.exists(), f"model file not found: {p}")
    try:
        import yaml
    except Exception as exc:  # pragma: no cover
        raise ConfigError("PyYAML is required to load model files") from exc
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    _require(isinstance(data, Mapping), f"{p}: top level must be a mapping")
    _require(len(data) >= 1, f"{p}: no model definitions found")
    models: Dict[str, ModelDef] = {}
    for model_id, body in data.items():
        _require(model_id.isidentifier(),
                 f"model id {model_id!r} must be a valid Python identifier")
        models[model_id] = model_def_from_dict(model_id, body)
    return models


def models_signature(models: Mapping[str, ModelDef]) -> str:
    """A stable string summarising the architecture of every model."""
    import json
    blob = json.dumps({mid: m.to_dict() for mid, m in sorted(models.items())},
                      sort_keys=True)
    return blob
