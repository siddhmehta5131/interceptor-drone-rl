"""model_loader.py -- load model definitions from a `model.yaml` file (plan A-1/A-4).

A model file is a flat mapping of ``model_id -> definition``.  The model_id is
the key used everywhere else in the pipeline: it names the TensorBoard
sub-directory, the checkpoint sub-directory and the attribute name on the
:class:`~src.results.TrainResult` returned by ``train_model()``.

Because ``TrainResult.ppo_baseline`` is attribute access, a model_id must be a
valid Python identifier -- an invalid one is rejected at load time rather than
failing later with an obscure ``AttributeError``.

Nothing here imports torch or stable-baselines3, so ``scripts/smoke_test.py``
can validate model files on a plain Python install.
"""

from __future__ import annotations

import keyword
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from .config_loader import ConfigError, _require

__all__ = [
    "ModelDef",
    "PRIVILEGED_FIELDS",
    "load_models",
    "models_signature",
    "model_def_from_dict",
]

_ALGOS = {"PPO", "SAC", "TD3"}
_POLICIES = {"MlpPolicy"}
_ACTIVATIONS = {"ReLU", "Tanh", "ELU"}
_ACTIVATION_ALIASES = {"relu": "ReLU", "tanh": "Tanh", "elu": "ELU"}

#: Privileged critic fields and how many observation slots each one occupies.
#: Mirrored by :data:`src.envs.obs_builder.PRIVILEGED_FIELDS`; the smoke test
#: asserts the two agree.
PRIVILEGED_FIELDS: Dict[str, int] = {
    "time_remaining": 1,
    "facing_error": 1,
    "target_true_pos": 3,
    "target_true_vel": 3,
}


@dataclass
class ModelDef:
    """One model's architecture + hyperparameters."""

    model_id: str
    algo: str
    policy: str = "MlpPolicy"
    net_arch: Dict[str, List[int]] = field(default_factory=dict)
    activation: str = "ReLU"
    obs_history: Dict[str, int] = field(default_factory=lambda: {"frames": 0, "skip": 1})
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    privileged_critic: List[str] = field(default_factory=list)
    source_path: str = "<memory>"

    # -- convenience -----------------------------------------------------
    @property
    def obs_frames(self) -> int:
        return int(self.obs_history.get("frames", 0))

    @property
    def obs_skip(self) -> int:
        return int(self.obs_history.get("skip", 1))

    @property
    def has_privileged(self) -> bool:
        return bool(self.privileged_critic)

    @property
    def privileged_dim(self) -> int:
        """Total width of the privileged critic block."""
        return sum(PRIVILEGED_FIELDS[n] for n in self.privileged_critic)

    def to_dict(self) -> Dict[str, Any]:
        """Plain-data view -- used for hashing and for `_training_meta.json`."""
        return {
            "model_id": self.model_id,
            "algo": self.algo,
            "policy": self.policy,
            "net_arch": {k: list(v) for k, v in self.net_arch.items()},
            "activation": self.activation,
            "obs_history": dict(self.obs_history),
            "hyperparameters": dict(self.hyperparameters),
            "privileged_critic": list(self.privileged_critic),
        }


def _valid_identifier(name: str) -> bool:
    """D-34: the model_id must be usable as ``TrainResult.<model_id>``.

    That only requires a valid Python identifier, so leading underscores are
    allowed (``result._ppo`` is legal Python) -- only the reserved dunder and
    keyword-shaped names are rejected.
    """
    if not isinstance(name, str) or not name.isidentifier():
        return False
    if keyword.iskeyword(name):
        return False
    return not (name.startswith("__") and name.endswith("__"))


def _canonicalize(raw: Mapping[str, Any], model_id: str) -> Dict[str, Any]:
    """Normalise a raw model block, accepting the older plan schema.

    Plan schema aliases accepted here (they were used by the pre-split
    single-file run config):

    * ``algorithm``            -> ``algo``
    * ``policy_kwargs: {net_arch, activation_fn}`` -> ``net_arch``/``activation``
    * ``obs_history: {m, s}``  -> ``obs_history: {frames, skip}``
    """
    c: Dict[str, Any] = dict(raw)

    if "algo" not in c and "algorithm" in c:
        c["algo"] = c["algorithm"]

    pk = c.pop("policy_kwargs", None)
    if isinstance(pk, Mapping):
        if "net_arch" in pk:
            c.setdefault("net_arch", pk["net_arch"])
        if "activation_fn" in pk:
            c.setdefault("activation", pk["activation_fn"])

    oh = c.get("obs_history")
    if isinstance(oh, Mapping):
        if "frames" not in oh and "m" in oh:
            oh = {"frames": oh["m"], "skip": oh.get("s", oh.get("skip", 1))}
        c["obs_history"] = oh

    # bare net_arch list -> {pi: [...], <head>: [...]}
    na = c.get("net_arch")
    if isinstance(na, (list, tuple)):
        head = "vf" if str(c.get("algo", "")).upper() == "PPO" else "qf"
        c["net_arch"] = {"pi": list(na), head: list(na)}

    act = c.get("activation")
    if isinstance(act, str):
        key = act.strip().lower()
        if key in _ACTIVATION_ALIASES:
            c["activation"] = _ACTIVATION_ALIASES[key]

    return c


def _validate(model_id: str, c: Mapping[str, Any]) -> ModelDef:
    _require(
        _valid_identifier(model_id),
        f"model id {model_id!r} must be a valid Python identifier "
        f"(letters/digits/underscore, not a keyword and not a dunder name)",
    )

    algo = str(c.get("algo", "")).upper()
    _require(algo in _ALGOS, f"{model_id}: algo must be one of {sorted(_ALGOS)}, got {algo!r}")
    algo = algo

    policy = c.get("policy", "MlpPolicy")
    _require(
        policy in _POLICIES,
        f"{model_id}: policy must be one of {sorted(_POLICIES)}, got {policy!r}",
    )

    net_arch = c.get("net_arch")
    _require(
        isinstance(net_arch, Mapping),
        f"{model_id}: net_arch must be a mapping "
        f"{{pi: [...], vf/qf: [...]}} or a bare list",
    )
    _require("pi" in net_arch, f"{model_id}: net_arch must contain a 'pi' entry")
    head = "vf" if algo == "PPO" else "qf"
    _require(head in net_arch, f"{model_id}: {algo} net_arch must contain a {head!r} entry")
    for k, v in net_arch.items():
        _require(
            isinstance(v, (list, tuple)) and len(v) > 0,
            f"{model_id}: net_arch[{k!r}] must be a non-empty list of ints",
        )
        for w in v:
            _require(
                isinstance(w, int) and not isinstance(w, bool) and w > 0,
                f"{model_id}: net_arch[{k!r}] entries must be positive ints, got {w!r}",
            )
    net_arch = {k: list(v) for k, v in net_arch.items()}

    activation = c.get("activation", "ReLU")
    if isinstance(activation, str):
        activation = _ACTIVATION_ALIASES.get(activation.strip().lower(), activation)
    _require(
        activation in _ACTIVATIONS,
        f"{model_id}: activation must be one of {sorted(_ACTIVATIONS)}, got {activation!r}",
    )

    oh = c.get("obs_history") or {}
    _require(isinstance(oh, Mapping), f"{model_id}: obs_history must be a mapping")
    frames = oh.get("frames", 0)
    skip = oh.get("skip", 1)
    _require(
        isinstance(frames, int) and not isinstance(frames, bool) and frames >= 0,
        f"{model_id}: obs_history.frames must be a non-negative int, got {frames!r}",
    )
    _require(
        isinstance(skip, int) and not isinstance(skip, bool) and skip >= 1,
        f"{model_id}: obs_history.skip must be an int >= 1, got {skip!r}",
    )

    hp = c.get("hyperparameters") or {}
    _require(
        isinstance(hp, Mapping),
        f"{model_id}: hyperparameters must be a mapping, got {type(hp).__name__}",
    )
    hp = dict(hp)

    pc = c.get("privileged_critic") or []
    _require(
        isinstance(pc, (list, tuple)),
        f"{model_id}: privileged_critic must be a list, got {type(pc).__name__}",
    )
    pc = [str(n) for n in pc]
    seen = set()
    for n in pc:
        _require(
            n in PRIVILEGED_FIELDS,
            f"{model_id}: unknown privileged_critic field {n!r}; "
            f"supported fields are {sorted(PRIVILEGED_FIELDS)}",
        )
        _require(n not in seen, f"{model_id}: duplicate privileged_critic field {n!r}")
        seen.add(n)

    return ModelDef(
        model_id=model_id,
        algo=algo,
        policy=policy,
        net_arch=net_arch,
        activation=str(activation),
        obs_history={"frames": int(frames), "skip": int(skip)},
        hyperparameters=hp,
        privileged_critic=pc,
    )


def model_def_from_dict(
    model_id: str, raw: Mapping[str, Any], *, source_path: str = "<memory>"
) -> ModelDef:
    """Validate + build a :class:`ModelDef` from an in-memory mapping.

    Used by the continuation path (D-28), where a prior run's saved metadata
    is turned back into a ``ModelDef`` instead of being re-read from YAML.
    """
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{model_id}: model definition must be a mapping, got {type(raw).__name__}")
    md = _validate(model_id, _canonicalize(raw, model_id))
    md.source_path = source_path
    return md


def load_models(path: Union[str, Path]) -> Dict[str, ModelDef]:
    """Load and validate every model definition in ``path``.

    Raises :class:`ConfigError` with a user-facing message on any problem.
    """
    import yaml

    p = Path(path)
    if not p.exists():
        raise ConfigError(f"model file not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - malformed YAML
        raise ConfigError(f"{p}: invalid YAML -- {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{p}: top level must be a mapping of model_id -> definition")

    out: Dict[str, ModelDef] = {}
    for model_id, block in raw.items():
        if not isinstance(block, Mapping):
            raise ConfigError(
                f"{p}: model {model_id!r} must be a mapping, got {type(block).__name__}"
            )
        md = _validate(str(model_id), _canonicalize(block, str(model_id)))
        md.source_path = str(p)
        out[md.model_id] = md
    _require(bool(out), f"{p}: no model definitions found")
    return out


def models_signature(models: Optional[Mapping[str, ModelDef]]) -> Dict[str, Any]:
    """Stable, hashable view of a model mapping (used by ``config_hash``)."""
    if not models:
        return {}
    return {
        mid: {
            "algo": md.algo,
            "policy": md.policy,
            "net_arch": {k: list(v) for k, v in sorted(md.net_arch.items())},
            "activation": md.activation,
            "obs_history": dict(sorted(md.obs_history.items())),
            "hyperparameters": {k: md.hyperparameters[k] for k in sorted(md.hyperparameters)},
            "privileged_critic": list(md.privileged_critic),
        }
        for mid, md in sorted(models.items())
    }