"""callbacks.py -- SB3 callbacks for curriculum training (plan §13).

:class:`InterceptMetricsCallback` maintains a rolling window of episode outcomes
for the current stage and stops training early (``on_step`` returns ``False``)
as soon as the stage's kill-rate threshold is met, or when the step budget is
exhausted.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional

from stable_baselines3.common.callbacks import BaseCallback

from ..envs.stage_config import StageConfig
from .curriculum import StageMetrics

__all__ = ["InterceptMetricsCallback"]


class InterceptMetricsCallback(BaseCallback):
    """Track kill rate / distance and stop a stage at the right moment."""

    def __init__(
        self,
        stage_config: StageConfig,
        window: int = 100,
        max_steps: Optional[int] = None,
        log_tensorboard: bool = True,
        log_every: int = 200,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.stage_config = stage_config
        self.window = int(window)
        self.max_steps = int(max_steps or stage_config.max_training_steps)
        self.log_tensorboard = log_tensorboard
        self.log_every = int(log_every)

        self._kill: Deque[float] = deque(maxlen=self.window)
        self._dist: Deque[float] = deque(maxlen=self.window)
        self._rew: Deque[float] = deque(maxlen=self.window)
        self._len: Deque[int] = deque(maxlen=self.window)
        self._ep_rew = 0.0
        self._ep_len = 0
        self._episodes = 0

        self.advanced = False
        self.capped = False

    # ---------------------------------------------------------------- metrics
    @property
    def recent_kill_rate(self) -> float:
        if not self._kill:
            return 0.0
        return float(sum(self._kill) / len(self._kill))

    def metrics(self) -> StageMetrics:
        return StageMetrics(
            stage=_stage_number(self.stage_config),
            steps=int(self.num_timesteps),
            episodes=self._episodes,
            recent_kill_rate=self.recent_kill_rate,
            window=self.window,
            mean_final_distance=_mean(self._dist),
            mean_episode_reward=_mean(self._rew),
            mean_episode_length=_mean(self._len),
        )

    # ------------------------------------------------------------------ hooks
    def _on_step(self) -> bool:
        infos: List[Dict[str, Any]] = self.locals.get("infos", []) or []
        rewards = self.locals.get("rewards", []) or []

        for i, info in enumerate(infos):
            step_reward = float(rewards[i]) if i < len(rewards) else 0.0
            self._ep_rew += step_reward
            self._ep_len += 1
            done = bool(info.get("killed") or info.get("out_of_bounds")
                        or info.get("timeout"))
            if done:
                self._kill.append(1.0 if info.get("killed") else 0.0)
                self._dist.append(float(info.get("distance", 0.0)))
                self._rew.append(self._ep_rew)
                self._len.append(self._ep_len)
                self._ep_rew = 0.0
                self._ep_len = 0
                self._episodes += 1

        if (self.log_tensorboard and self.logger is not None
                and self.n_calls % self.log_every == 0):
            self.logger.record("curriculum/kill_rate", self.recent_kill_rate)
            self.logger.record("curriculum/episodes", self._episodes)
            self.logger.record("curriculum/mean_final_distance", _mean(self._dist))

        if (len(self._kill) >= self.window
                and self.recent_kill_rate >= self.stage_config.threshold):
            self.advanced = True
            if self.verbose:
                print(f"[curriculum] stage {self.stage_config.id} cleared "
                      f"(kill_rate={self.recent_kill_rate:.3f})")
            return False

        if self.num_timesteps >= self.max_steps:
            self.capped = True
            if self.verbose:
                print(f"[curriculum] stage {self.stage_config.id} budget exhausted "
                      f"(kill_rate={self.recent_kill_rate:.3f})")
            return False

        return True


def _mean(values) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _stage_number(stage_config: StageConfig) -> int:
    tail = stage_config.id.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else 0
