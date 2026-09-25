"""curriculum.py -- per-stage CurriculumScheduler (implementation_plan §6).

The scheduler consumes one episode record per completed environment episode
and decides, per plan §6.1/§6.2/§6.3:

* **advance**  -- the rolling success rate over the last ``window`` episodes
  reached the stage's ``success_rate``.  The episode *record* is
  :class:`dict` of the env's ``info['episode_stats']`` (flat float/int keys).
* **capped**   -- ``total_steps >= max_training_steps`` (plan §6.1 hard cap:
  the stage did not converge but training must proceed).
* **rollback** -- rolling success rate stayed below the rollback threshold
  for long enough (plan §6.2: previous stage's skills are insufficient).

Episode success judgement (plan §6, "record[metric] >= threshold"):
* ``kill_rate``        -> ``stats['kill'] == 1.0``
* ``mean_final_distance`` -> ``stats['final_distance'] <= threshold``
* ``episode_reward_mean`` -> ``stats['episode_reward'] >= threshold``
    -- the plan's stage-1 metric name; reward is ~0.1/step so a perfect
    1000-step hover totals ~100, hence the 60 threshold is matched against
    the *episode total* reward (documented interpretation).

Rollback policy: disabled when ``rollback_threshold`` is 0 or negative; the
stage default (``stage.rollback_threshold is None``) applies plan §6.2's
``50% of the stage's success_rate``.  ``rollback_min_steps`` guards against
rolling back before the stage had a fair chance (default ``0.25 *
max_training_steps``).  The scheduler itself is deterministic and
serialisable for resume (``state_dict``/``load_state_dict``).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, List, Optional

from ..envs.stage_config import StageConfig

__all__ = ["EpisodeSuccess", "CurriculumScheduler"]

# result strings returned by CurriculumScheduler.on_episode_end / peak()
RUNNING = "running"
ADVANCE = "advance"
CAPPED = "capped"
ROLLBACK = "rollback"
TERMINAL_RESULTS = (ADVANCE, CAPPED, ROLLBACK)


def episode_success(stage: StageConfig, stats: Dict) -> bool:
    """True when one episode satisfies the stage's success metric."""
    metric = stage.success_metric
    if metric == "kill_rate":
        return bool(stats.get("kill", 0.0))
    if metric == "mean_final_distance":
        return float(stats.get("final_distance", math.inf)) <= float(stage.threshold)
    if metric == "episode_reward_mean":
        return float(stats.get("episode_reward", -math.inf)) >= float(stage.threshold)
    if metric == "episode_reward":
        return float(stats.get("episode_reward", -math.inf)) >= float(stage.threshold)
    raise ValueError(f"unknown success_metric {metric!r}")


class CurriculumScheduler:
    """Decides advance / cap / rollback for one stage.

    Parameters
    ----------
    stage:
        The :class:`StageConfig` being trained.
    rollback_threshold:
        Absolute success-rate under which a rollback is triggered.  ``None``
        uses the stage default (``stage.rollback_threshold`` if set, else
        50% of ``stage.success_rate``); 0 disables rollback.
    rollback_min_steps:
        Minimum cumulative steps the stage must have run before a rollback
        can trigger (default ``0.25 * max_training_steps``).
    budget:
        Per-entry training budget in env steps (defaults to
        ``stage.max_training_steps``).  Used for the hard cap and for the
        rollback re-entry budget (``retry_budget_scale``).
    """

    def __init__(
        self,
        stage: StageConfig,
        *,
        rollback_threshold: Optional[float] = None,
        rollback_min_steps: Optional[int] = None,
        budget: Optional[int] = None,
    ) -> None:
        self.stage = stage
        self.budget = int(budget) if budget is not None else int(stage.max_training_steps)
        self.required_rate = float(stage.success_rate)
        self.rollback_min_steps = (
            int(rollback_min_steps)
            if rollback_min_steps is not None
            else int(0.25 * self.budget)
        )

        if rollback_threshold is not None:
            self.rollback_threshold = float(rollback_threshold)
        elif stage.rollback_threshold is not None:
            self.rollback_threshold = float(stage.rollback_threshold)
        else:
            self.rollback_threshold = 0.5 * self.required_rate

        self._records: Deque[float] = deque(maxlen=stage.window)
        self.total_steps = 0
        self.episodes_seen = 0

        self.advance_triggered = False
        self.capped = False
        self.rollback_triggered = False
        self._finished_result: Optional[str] = None

    # -- queries ------------------------------------------------------------
    @property
    def window(self) -> int:
        return int(self.stage.window)

    @property
    def finished(self) -> bool:
        return self._finished_result is not None

    @property
    def result(self) -> Optional[str]:
        return self._finished_result

    def rate(self) -> float:
        """Rolling success rate over the last ``window`` episodes (0 if none)."""
        if not self._records:
            return 0.0
        return float(sum(self._records)) / float(len(self._records))

    def episodes_in_window(self) -> int:
        return len(self._records)

    # -- event handlers -----------------------------------------------------
    def on_episode_end(self, stats: Dict) -> str:
        """Feed one completed-episode record.

        Returns ``'running'`` while the stage continues, otherwise one of the
        terminal results (the first terminal result is sticky -- subsequent
        calls keep returning it without re-evaluating).
        """
        if self._finished_result is not None:
            return self._finished_result

        self.episodes_seen += 1
        self._records.append(1.0 if episode_success(self.stage, stats) else 0.0)
        self.total_steps += max(0, int(stats.get("steps", 0)))

        # 1) hard cap (plan §6.1) -- highest precedence
        if self.total_steps >= self.budget:
            self.capped = True
            self._finished_result = CAPPED
            return CAPPED

        window_full = len(self._records) >= self.window

        # 2) advance on success-rate (plan §6.1)
        if window_full and self.rate() >= self.required_rate:
            self.advance_triggered = True
            self._finished_result = ADVANCE
            return ADVANCE

        # 3) rollback (plan §6.2)
        if (
            self.rollback_threshold > 0.0
            and window_full
            and self.rate() < self.rollback_threshold
            and self.total_steps >= self.rollback_min_steps
        ):
            self.rollback_triggered = True
            self._finished_result = ROLLBACK
            return ROLLBACK

        return RUNNING

    def on_steps(self, n: int) -> None:
        """Accrue environment steps not covered by episode records
        (e.g. steps of episodes still running when learning stopped)."""
        self.total_steps += max(0, int(n))

    def sync_steps(self, total: int) -> None:
        """Set the authoritative per-stage step count (e.g. derived from the
        model's ``num_timesteps`` delta) and apply the hard cap if reached.

        This closes the gap between episode-driven step accounting and the
        true environment-step counter (a partially-run final episode would
        otherwise leave the scheduler short of its cap).
        """
        if self._finished_result is not None:
            return
        self.total_steps = max(self.total_steps, int(total))
        if self.total_steps >= self.budget:
            self.capped = True
            self._finished_result = CAPPED

    # -- serialisation ------------------------------------------------------
    def state_dict(self) -> Dict:
        return {
            "total_steps": self.total_steps,
            "episodes_seen": self.episodes_seen,
            "records": list(self._records),
            "advance_triggered": self.advance_triggered,
            "capped": self.capped,
            "rollback_triggered": self.rollback_triggered,
            "finished_result": self._finished_result,
        }

    def load_state_dict(self, d: Dict) -> None:
        self.total_steps = int(d["total_steps"])
        self.episodes_seen = int(d.get("episodes_seen", 0))
        recs: List[float] = [float(x) for x in d.get("records", [])]
        self._records = deque(recs, maxlen=self.window)
        self.advance_triggered = bool(d.get("advance_triggered", False))
        self.capped = bool(d.get("capped", False))
        self.rollback_triggered = bool(d.get("rollback_triggered", False))
        self._finished_result = d.get("finished_result")
        if self._finished_result not in (None,) + TERMINAL_RESULTS:
            self._finished_result = None

    def summary(self) -> Dict:
        return {
            "stage": self.stage.id,
            "success_rate": self.rate(),
            "required_rate": self.required_rate,
            "episodes_in_window": self.episodes_in_window(),
            "episodes_seen": self.episodes_seen,
            "total_steps": self.total_steps,
            "max_training_steps": self.budget,
            "rollback_threshold": self.rollback_threshold,
            "result": self.result,
        }