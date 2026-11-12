"""api.py -- the public training / evaluation entry points.

Typical use::

    from interceptor.api import train_model
    result = train_model(source="configs/model.yaml", stages=[1, 2, 3],
                         name="apn_rl_v1", config="configs/config.yaml")
    result.ppo_interceptor.save("data/models/apn_rl_v1_final")
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .results import TrainResult, STATUS_FAILED, STATUS_COMPLETE
from .training.orchestrator import Orchestrator, evaluate_policy, resolve_device
from .utils.config_loader import ConfigError, TrainConfig, load_config
from .utils.model_loader import load_models

__all__ = ["train_model", "load_config_file", "evaluate_model", "TrainConfig"]


def _resolve_stages(cfg: TrainConfig, stages: Optional[Sequence[int]]) -> List[int]:
    available = cfg.stage_numbers()
    if stages is None:
        return available
    out = []
    for s in stages:
        n = int(s)
        if n not in available:
            raise ConfigError(f"stage {n} not configured; available: {available}")
        out.append(n)
    return out


def train_model(
    *,
    source: str = "configs/model.yaml",
    stages: Optional[Sequence[int]] = None,
    name: str = "run",
    config: str = "configs/config.yaml",
    data_dir: Optional[str] = None,
    device: Optional[str] = None,
    max_steps_per_stage: Optional[int] = None,
    verbose: int = 1,
) -> TrainResult:
    """Train every model in ``source`` across the requested curriculum stages."""
    cfg = load_config(config)
    models = load_models(source)
    cfg.bind_models(models)

    run_root = Path(data_dir or cfg.data_dir) / str(name)
    run_root.mkdir(parents=True, exist_ok=True)
    stage_list = _resolve_stages(cfg, stages)

    result = TrainResult(run_dir=str(run_root), config_hash=cfg.config_hash)
    try:
        for model_id, model_def in models.items():
            orch = Orchestrator(cfg, model_def, str(run_root), device=device,
                                verbose=verbose)
            model_result = orch.run(stage_list,
                                    max_steps_per_stage=max_steps_per_stage)
            result.models[model_id] = model_result
            _write_json(run_root / f"{model_id}_result.json",
                        model_result.to_dict())
    except Exception as exc:  # pragma: no cover - defensive
        result.status = STATUS_FAILED
        result.finished_at = time.time()
        _write_json(run_root / "failure.json",
                    {"error": repr(exc), "config_hash": cfg.config_hash})
        raise

    result.finished_at = time.time()
    if all(mr.status == STATUS_COMPLETE for mr in result.models.values()):
        result.status = STATUS_COMPLETE
    else:
        result.status = next(
            (mr.status for mr in result.models.values()
             if mr.status != STATUS_COMPLETE), STATUS_COMPLETE)
    _write_json(run_root / "results.json", result.to_dict())
    _write_json(run_root / "config_snapshot.json",
                {"config": cfg.training_signature_dict(),
                 "config_hash": cfg.config_hash,
                 "stages": {sid: sc.target_type for sid, sc in cfg.stages.items()}})
    return result


def load_config_file(path: str = "configs/config.yaml") -> TrainConfig:
    """Convenience wrapper so callers do not import from ``utils`` directly."""
    return load_config(path)


def evaluate_model(
    model_path: str,
    *,
    config: str = "configs/config.yaml",
    model_yaml: str = "configs/model.yaml",
    stage: int = 1,
    episodes: int = 100,
    deterministic: bool = True,
    collect_trajectory: bool = False,
) -> Dict[str, object]:
    """Load a trained policy and evaluate it on one curriculum stage."""
    from stable_baselines3 import PPO
    from .training.asymmetric_policy import AsymmetricActorCriticPolicy  # noqa: F401

    cfg = load_config(config)
    models = load_models(model_yaml)
    model_def = next(iter(models.values()))
    cfg.bind_models(models)

    model = PPO.load(model_path, device=resolve_device(cfg.device))
    return evaluate_policy(
        model, cfg.stage(int(stage)), cfg, model_def.privileged_critic,
        episodes=int(episodes), deterministic=deterministic,
        collect_trajectory=collect_trajectory)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_default),
                    encoding="utf-8")


def _default(obj):
    try:
        import numpy as np
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    return str(obj)
