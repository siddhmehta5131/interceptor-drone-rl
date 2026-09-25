"""callbacks.py -- SB3 callbacks: reward components, metrics, checkpoints,
curriculum (stable_baselines3 required at import time).

TensorBoard layout (plan §8):

* ``reward_components/{r_alive,r_alt,r_tilt,r_angvel,r_thrust,r_smooth,
  r_velocity_alignment,r_progress_delta,r_time_penalty,r_kill_bonus,
  r_miss_distance,r_crash,r_oob,r_time_bonus}``  -- per-rollout step means
* ``reward/episode_total``                          -- mean episode reward
* ``metrics/{alt_err,tilt,angvel_norm,miss_distance,time_to_intercept}``
* ``curriculum/{current_stage,success_rate,episodes_in_window,required_rate}``

SB3 loggers dump at rollout end for on-policy algos; for off-policy
(SAC/TD3) the metrics callbacks additionally dump at the ``log_interval``
boundary inside ``_on_step`` (always at ``num_timesteps % log_interval == 0``,
which is the same boundary the off-policy ``learn()`` loop uses for its own
dump).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from .curriculum import CurriculumScheduler

__all__ = [
    "REWARD_COMPONENT_KEYS",
    "RewardComponentCallback",
    "MetricsCallback",
    "CurriculumCheckpointCallback",
    "CurriculumCallback",
]

REWARD_COMPONENT_KEYS: tuple = (
    "r_alive",
    "r_alt",
    "r_tilt",
    "r_angvel",
    "r_thrust",
    "r_smooth",
    "r_velocity_alignment",
    "r_progress_delta",
    "r_time_penalty",
    "r_kill_bonus",
    "r_miss_distance",
    "r_crash",
    "r_oob",
    "r_time_bonus",
)

METRIC_KEYS: tuple = (
    "alt_err",
    "tilt",
    "angvel_norm",
    "miss_distance",
    "time_to_intercept",
)

_OFFPOLICY = frozenset({"SAC", "TD3", "DDPG"})


class RewardComponentCallback(BaseCallback):
    """Record per-step reward-component values, mean over rollout (§8)."""

    KEYS = REWARD_COMPONENT_KEYS

    def __init__(self, verbose: int = 0, log_interval: int = 0) -> None:
        super().__init__(verbose)
        # Off-policy (SAC/TD3) never call _on_rollout_end, so the flush
        # cadence is injected explicitly instead of being introspected from
        # the model (SB3 does not reliably store log_interval on it).
        self.log_interval = max(0, int(log_interval))
        self.values: Dict[str, List[float]] = {k: [] for k in self.KEYS}

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if not isinstance(info, dict):
                continue
            for k in self.KEYS:
                if k in info:
                    self.values[k].append(float(info[k]))
        # SAC/TD3 never call _on_rollout_end: flush at the injected
        # log_interval boundary (same one the off-policy learn() loop uses).
        is_offpolicy = self.model.__class__.__name__ in _OFFPOLICY
        if is_offpolicy and self._due_dump():
            self._flush()
            self.logger.dump(self.num_timesteps)
        return True

    def _on_rollout_end(self) -> None:
        # on-policy: record now; the learn() loop dumps right after.
        self._flush()

    def _due_dump(self) -> bool:
        return (
            self.log_interval > 0
            and self.num_timesteps > 0
            and self.num_timesteps % self.log_interval == 0
        )

    def _flush(self) -> None:
        for k in self.KEYS:
            if self.values[k]:
                self.logger.record(
                    f"reward_components/{k}", float(np.mean(self.values[k]))
                )
                self.values[k].clear()


class MetricsCallback(BaseCallback):
    """Per-rollout means of step metrics + mean episode reward (§8)."""

    def __init__(self, verbose: int = 0, log_interval: int = 0) -> None:
        super().__init__(verbose)
        # Off-policy flush cadence is injected explicitly (see
        # RewardComponentCallback): SB3 does not reliably store
        # log_interval on the model object.
        self.log_interval = max(0, int(log_interval))
        self._metrics: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS}
        self._episode_rewards: List[float] = []

    # -- accumulation ---------------------------------------------------
    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if not isinstance(info, dict):
                continue
            for k in METRIC_KEYS:
                if k in info and info[k] is not None:
                    self._metrics[k].append(float(info[k]))
            stats = info.get("episode_stats")
            if isinstance(stats, dict) and "episode_reward" in stats:
                self._episode_rewards.append(float(stats["episode_reward"]))

        # off-policy algos never call _on_rollout_end; dump at the same
        # log_interval boundary the off-policy learn() loop uses.
        is_offpolicy = self.model.__class__.__name__ in _OFFPOLICY
        if is_offpolicy and self._due_dump():
            self._record_and_clear()
            self.logger.dump(self.num_timesteps)
        return True

    def _on_rollout_end(self) -> None:
        # on-policy: record now; the learn() loop dumps right after.
        self._record_and_clear()

    def _due_dump(self) -> bool:
        return (
            self.log_interval > 0
            and self.num_timesteps > 0
            and self.num_timesteps % self.log_interval == 0
        )

    def _record_and_clear(self) -> None:
        for k in METRIC_KEYS:
            if self._metrics[k]:
                self.logger.record(f"metrics/{k}", float(np.mean(self._metrics[k])))
                self._metrics[k].clear()
        if self._episode_rewards:
            self.logger.record(
                "reward/episode_total", float(np.mean(self._episode_rewards))
            )
            self._episode_rewards.clear()


class CurriculumCheckpointCallback(BaseCallback):
    """Periodic model checkpoint via :class:`CheckpointManager`."""

    def __init__(self, manager, stage_id: str, save_freq: int, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.manager = manager
        self.stage_id = stage_id
        self.save_freq = int(save_freq)
        self._last_save = 0

    def _on_step(self) -> bool:
        if (
            self.save_freq > 0
            and self.num_timesteps > 0
            and self.num_timesteps - self._last_save >= self.save_freq
        ):
            self.manager.save_model(self.model, self.stage_id, self.num_timesteps)
            self._last_save = self.num_timesteps
        return True


class CurriculumCallback(BaseCallback):
    """Feed completed episodes to the CurriculumScheduler and stop
    ``learn()`` (``on_step`` -> False) when the stage advances / caps /
    rolls back.  ``stage_finished`` holds the scheduler's terminal result
    after ``learn()`` returns (None if learning ended on the step budget)."""

    def __init__(self, scheduler: CurriculumScheduler, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.scheduler = scheduler
        self.stage_baseline: int = 0
        self.stage_finished: Optional[str] = None

    def _on_training_start(self) -> None:
        # model.num_timesteps at the start of THIS learn() call (0 for a
        # fresh model; >0 when resuming from a mid-stage checkpoint).
        self.stage_baseline = int(self.model.num_timesteps)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if not isinstance(info, dict):
                continue
            stats = info.get("episode_stats")
            if isinstance(stats, dict):
                self.scheduler.on_episode_end(stats)

        # authoritative per-stage step count + hard cap (plan §6.1)
        if self.scheduler.result is None:
            self.scheduler.sync_steps(self.num_timesteps - self.stage_baseline)

        if self.scheduler.result is not None:
            self.stage_finished = self.scheduler.result
            return False
        return True

    def _on_rollout_end(self) -> None:
        sch = self.scheduler
        stage_num = int(self.scheduler.stage.id.split("_")[1])
        self.logger.record("curriculum/current_stage", stage_num)
        self.logger.record("curriculum/success_rate", sch.rate())
        self.logger.record("curriculum/required_rate", sch.required_rate)
        self.logger.record("curriculum/episodes_in_window", sch.episodes_in_window())