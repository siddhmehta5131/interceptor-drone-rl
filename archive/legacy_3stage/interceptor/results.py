"""results.py -- result containers returned by the public training API.

``train_model()`` returns a :class:`TrainResult` that exposes one
:class:`ModelResult` attribute per ``model_id`` (e.g. ``result.ppo_interceptor``)
so callers can inspect per-stage outcomes and ``.save()`` the trained policy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "StageOutcome", "ModelResult", "TrainResult",
    "STATUS_COMPLETE", "STATUS_CAPPED", "STATUS_FAILED", "STATUS_STOPPED",
]

STATUS_COMPLETE = "complete"     # all requested stages cleared
STATUS_CAPPED = "capped"         # a stage exhausted its budget without success
STATUS_FAILED = "failed"         # an error aborted the run
STATUS_STOPPED = "stopped"       # user/config requested an early stop


@dataclass
class StageOutcome:
    """Outcome of training one curriculum stage."""

    stage: int
    status: str = STATUS_COMPLETE
    steps_trained: int = 0
    kill_rate: float = 0.0
    threshold: float = 0.0
    advanced: bool = False
    mean_final_distance: float = 0.0
    mean_episode_reward: float = 0.0
    mean_episode_length: float = 0.0
    checkpoint_path: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.kill_rate >= self.threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage, "status": self.status,
            "steps_trained": self.steps_trained, "kill_rate": self.kill_rate,
            "threshold": self.threshold, "advanced": self.advanced,
            "mean_final_distance": self.mean_final_distance,
            "mean_episode_reward": self.mean_episode_reward,
            "mean_episode_length": self.mean_episode_length,
            "checkpoint_path": self.checkpoint_path, "metrics": self.metrics,
        }


@dataclass
class ModelResult:
    """Result of training one model (one ``model_id``) across all stages."""

    model_id: str
    status: str = STATUS_COMPLETE
    stages: List[StageOutcome] = field(default_factory=list)
    run_dir: str = ""
    final_path: Optional[str] = None
    obs_dim: int = 0
    actor_dim: int = 0
    privileged_dim: int = 0
    algorithm: str = "PPO"
    duration_s: float = 0.0
    model: Any = None            # the underlying SB3 model (not serialised)

    @property
    def stages_attempted(self) -> List[int]:
        return [s.stage for s in self.stages]

    @property
    def cleared_stages(self) -> List[int]:
        return [s.stage for s in self.stages if s.passed]

    @property
    def last_stage(self) -> Optional[StageOutcome]:
        return self.stages[-1] if self.stages else None

    def save(self, path: str) -> str:
        """Persist the underlying SB3 model to ``path`` (``.zip`` appended)."""
        if self.model is None:
            raise RuntimeError("no model to save (training failed?)")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(p))
        self.final_path = str(p)
        return str(p)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id, "status": self.status,
            "stages": [s.to_dict() for s in self.stages],
            "run_dir": self.run_dir, "final_path": self.final_path,
            "obs_dim": self.obs_dim, "actor_dim": self.actor_dim,
            "privileged_dim": self.privileged_dim, "algorithm": self.algorithm,
            "duration_s": self.duration_s,
        }


@dataclass
class TrainResult:
    """Top-level result of :func:`interceptor.api.train_model`."""

    run_dir: str = ""
    config_hash: str = ""
    status: str = STATUS_COMPLETE
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    models: Dict[str, ModelResult] = field(default_factory=dict)

    def __getattr__(self, name: str) -> ModelResult:
        models = self.__dict__.get("models", {})
        if name in models:
            return models[name]
        raise AttributeError(
            f"TrainResult has no attribute {name!r}; "
            f"trained models: {sorted(models)}")

    @property
    def duration_s(self) -> float:
        end = self.finished_at or time.time()
        return end - self.started_at

    def model_ids(self) -> List[str]:
        return list(self.models.keys())

    def summary(self) -> str:
        lines = [f"TrainResult status={self.status} run_dir={self.run_dir}"]
        for mid, mr in self.models.items():
            lines.append(f"  {mid}: status={mr.status} "
                         f"cleared={mr.cleared_stages} obs_dim={mr.obs_dim}")
            for s in mr.stages:
                lines.append(
                    f"    stage {s.stage}: kill_rate={s.kill_rate:.3f} "
                    f"(thr {s.threshold:.2f}) steps={s.steps_trained} "
                    f"{'PASS' if s.passed else 'FAIL'}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_dir": self.run_dir, "config_hash": self.config_hash,
            "status": self.status, "duration_s": self.duration_s,
            "models": {mid: mr.to_dict() for mid, mr in self.models.items()},
        }
