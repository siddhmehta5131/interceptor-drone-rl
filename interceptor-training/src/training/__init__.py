"""Training package -- curriculum, checkpoints, callbacks, orchestrator.

Submodules are importable without torch/stable-baselines3 (SB3 is only
imported lazily where genuinely needed).  The smoke test therefore exercises
``curriculum`` and ``checkpoint_manager`` on a plain Python install.
"""

from .checkpoint_manager import CheckpointManager
from .curriculum import (
    ADVANCE,
    CAPPED,
    ROLLBACK,
    RUNNING,
    TERMINAL_RESULTS,
    CurriculumScheduler,
    episode_success,
)

__all__ = [
    "CheckpointManager",
    "ADVANCE",
    "CAPPED",
    "ROLLBACK",
    "RUNNING",
    "TERMINAL_RESULTS",
    "CurriculumScheduler",
    "episode_success",
]