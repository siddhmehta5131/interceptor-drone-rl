"""curriculum.py -- 3-stage curriculum progression logic (plan §13).

The scheduler decides, given the running metrics for the current stage, whether
to advance, roll back, or report a stage as capped.  The core rule (plan §13.4)
is simply::

    advance when recent_kill_rate(window) >= stage.threshold
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..envs.stage_config import StageConfig

__all__ = [
    "ADVANCE", "CAPPED", "ROLLBACK", "CONTINUE",
    "StageMetrics", "should_advance", "is_capped", "CurriculumScheduler",
]

ADVANCE = "advance"
CAPPED = "capped"
ROLLBACK = "rollback"
CONTINUE = "continue"


@dataclass
class StageMetrics:
    """Rolling metrics for the stage currently being trained."""

    stage: int
    steps: int = 0
    episodes: int = 0
    recent_kill_rate: float = 0.0
    window: int = 100
    mean_final_distance: float = 0.0
    mean_episode_reward: float = 0.0
    mean_episode_length: float = 0.0
    extra: Dict[str, float] = field(default_factory=dict)


def should_advance(metrics: StageMetrics, stage_config: StageConfig) -> bool:
    """True once the rolling kill rate clears the stage threshold."""
    return float(metrics.recent_kill_rate) >= float(stage_config.threshold)


def is_capped(metrics: StageMetrics, stage_config: StageConfig) -> bool:
    """True once the stage budget is exhausted without clearing the threshold."""
    return (metrics.steps >= int(stage_config.max_training_steps)
            and not should_advance(metrics, stage_config))


class CurriculumScheduler:
    """Tracks the current stage and the rollback bookkeeping."""

    def __init__(self, stages: List[StageConfig], rollback_cfg: Optional[dict] = None):
        self.stages = list(stages)
        self.rollback_cfg = dict(rollback_cfg or {})
        self.max_attempts = int(self.rollback_cfg.get("max_attempts", 2))
        self.index = 0
        self.attempts: Dict[int, int] = {}

    # ------------------------------------------------------------------ state
    @property
    def current(self) -> StageConfig:
        return self.stages[self.index]

    @property
    def done(self) -> bool:
        return self.index >= len(self.stages)

    def rollback_budget(self, stage_config: StageConfig) -> int:
        scale = float(self.rollback_cfg.get("retry_budget_scale", 0.5))
        return max(1, int(stage_config.max_training_steps * scale))

    def attempt_count(self, stage_number: int) -> int:
        return self.attempts.get(stage_number, 0)

    # --------------------------------------------------------------- actions
    def next_action(self, metrics: StageMetrics) -> str:
        """Return the action to take after a training interval."""
        stage_config = self.current
        if should_advance(metrics, stage_config):
            return ADVANCE
        if is_capped(metrics, stage_config):
            n = self.attempt_count(stage_config.id)
            if n < self.max_attempts:
                return ROLLBACK
            return CAPPED
        return CONTINUE

    def advance(self) -> None:
        self.index += 1

    def note_rollback(self) -> None:
        sid = self.current.id
        self.attempts[sid] = self.attempts.get(sid, 0) + 1

    def reset(self) -> None:
        self.index = 0
        self.attempts = {}
