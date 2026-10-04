"""Controllers: fixed instructions, or an uploaded checkpoint's own response.

Both satisfy the same tiny protocol used by the runner::

    ctl.reset()                       # start of a segment
    action = ctl.act(telemetry, dt)   # -> np.ndarray (4,) in [-1, 1]

``ModelController`` imports torch / stable_baselines3 lazily, so the
instruction path works in an image without the RL stack.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from .stage_config import StageDemoConfig


class ControllerError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Checkpoint sidecar
# ---------------------------------------------------------------------------

@dataclass
class CheckpointInfo:
    """What a checkpoint says about the layout it expects."""

    path: Path
    algo: str = "PPO"
    obs_dim: Optional[int] = None
    history_frames: int = 0
    history_skip: int = 1
    privileged_fields: tuple = ()
    net_arch: Dict[str, Any] = field(default_factory=dict)
    from_sidecar: bool = False

    @property
    def privileged_width(self) -> int:
        from .stage_config import privileged_width
        return privileged_width(self.privileged_fields)


def read_checkpoint_info(path: Path) -> CheckpointInfo:
    """Read a checkpoint's sidecar metadata (never loads the weights).

    ``<name>.training_meta.json`` is written next to every checkpoint by the
    training orchestrator, so the demo env can be built to match without
    importing torch.
    """
    info = CheckpointInfo(path=Path(path), algo=path.stem.split("_")[0] or "PPO")
    sidecar = Path(path).with_suffix("").with_suffix(".training_meta.json")
    if not sidecar.is_file():
        sidecar = Path(str(path) + ".training_meta.json")
    if not sidecar.is_file():
        return info
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return info
    mdef = meta.get("model_def") or {}
    info.algo = str(meta.get("algo") or mdef.get("algo") or info.algo)
    hist = mdef.get("obs_history") or {}
    info.history_frames = int(hist.get("frames", 0) or 0)
    info.history_skip = int(hist.get("skip", 1) or 1)
    info.privileged_fields = tuple(mdef.get("privileged_critic") or ())
    net = mdef.get("net_arch") or {}
    info.net_arch = {k: list(v) for k, v in net.items()} if isinstance(net, Mapping) else {}
    info.from_sidecar = True
    return info


# ---------------------------------------------------------------------------
# Model controller
# ---------------------------------------------------------------------------

class ModelController:
    """Drives the drone from a trained policy (SB3), one call per 100 Hz step."""

    def __init__(self, path: Path, algo: str = "PPO", *, deterministic: bool = True,
                 device: str = "auto"):
        self.path = Path(path)
        self.algo = str(algo).upper()
        self.deterministic = bool(deterministic)
        self.device = device
        self._model = None
        self._infer_device = None

    # -- lazy import ----------------------------------------------------

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            import torch  # noqa: F401
            from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
        except ImportError as exc:      # pragma: no cover - depends on image
            raise ControllerError(
                "control == 'model' needs torch + stable-baselines3 in this image. "
                "Build the stage image with the ML extra, or set "
                "\"control\": \"instructions\" in the stage JSON."
            ) from exc
        registry = {"PPO": PPO, "SAC": SAC, "TD3": TD3, "DDPG": DDPG,
                    "A2C": A2C, "DQN": DQN}
        cls = registry.get(self.algo)
        if cls is None:
            raise ControllerError(
                f"unsupported algo {self.algo!r}; known: {sorted(registry)}")
        if not self.path.is_file():
            raise ControllerError(f"checkpoint not found: {self.path}")
        self._model = cls.load(self.path, device=self.device)
        return self._model

    @property
    def obs_dim(self) -> Optional[int]:
        try:
            model = self._load()
        except ControllerError:
            return None
        space = getattr(model, "observation_space", None)
        shape = getattr(space, "shape", None)
        return int(shape[0]) if shape else None

    def reset(self) -> None:
        return None

    def act(self, tel: Mapping[str, Any], dt: float,
            obs: Optional[np.ndarray] = None) -> np.ndarray:
        model = self._load()
        if obs is None:
            raise ControllerError("ModelController.act needs the env observation")
        action, _ = model.predict(np.asarray(obs, dtype=np.float32),
                                  deterministic=self.deterministic)
        return np.clip(np.asarray(action, dtype=np.float64).reshape(4), -1.0, 1.0)

    def describe(self) -> str:
        return f"{self.algo} checkpoint {self.path.name} " \
               f"({'deterministic' if self.deterministic else 'stochastic'})"


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def resolve_observation(cfg: StageDemoConfig, model_path: Optional[Path]
                        ) -> "tuple[StageDemoConfig, CheckpointInfo | None]":
    """Reconcile the JSON observation layout with the checkpoint's own layout.

    The sidecar's ``obs_history``/``privileged_critic`` win over the JSON (the
    checkpoint is the ground truth); ``future_samples`` is then auto-fitted
    from the checkpoint's observation width.
    """
    from dataclasses import replace

    if model_path is None:
        return cfg, None
    info = read_checkpoint_info(model_path)
    obs = cfg.observation
    changes: Dict[str, Any] = {}
    if info.from_sidecar:
        if info.history_frames != obs.history_frames:
            changes["history_frames"] = info.history_frames
        if info.history_skip != obs.history_skip:
            changes["history_skip"] = info.history_skip
        if info.privileged_fields != tuple(obs.privileged_fields):
            changes["privileged_fields"] = info.privileged_fields
    if changes:
        obs = replace(obs, **changes)
    obs = obs.fitted(_checkpoint_obs_dim(model_path))
    if obs is not cfg.observation:
        cfg = replace(cfg, observation=obs)
    return cfg, info


def _checkpoint_obs_dim(path: Path) -> Optional[int]:
    """Observation width recorded in the checkpoint, without loading torch.

    ``save()`` stores ``observation_space`` inside the zip, so the width can be
    read straight out of the pickle-free ``data.json`` entry.
    """
    import zipfile

    try:
        with zipfile.ZipFile(path) as zf:
            if "data" not in zf.namelist():
                return None
            with zf.open("data") as fh:
                payload = json.loads(fh.read().decode("utf-8"))
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    space = payload.get("observation_space") or {}
    shape = ((space.get("spaces") or {}).get("Box", {}) or {}).get("shape")
    if shape:
        return int(shape[0])
    return None


def build_controller(cfg: StageDemoConfig, *, model_path: Optional[Path] = None,
                     data_roots: Sequence[Path] = ()) -> Any:
    """Build the controller for ``cfg`` (instructions or model)."""
    if cfg.control == "model":
        path = model_path or cfg.model.resolve(extra_roots=data_roots)
        return ModelController(path, cfg.model.algo,
                               deterministic=cfg.model.deterministic)
    from .instructions import build_instruction_controller
    return build_instruction_controller(cfg)
