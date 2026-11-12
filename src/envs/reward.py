"""reward.py -- dense + terminal reward for the APN + RL interceptor (plan §10).

Every term is a small, independently-weighted quantity so the overall reward is
easy to reason about and ablate.  The KillBonus is the only large term; the
dense shaping exists to guide the policy toward the intercept geometry that the
APN baseline already solves well.

Terms (plan §10.2 / §10.3)
--------------------------
* ``r_progress``     closing distance (optional normalisation).
* ``r_LOS_rate``     penalise line-of-sight rotation (want a collision course).
* ``r_closing_vel``  reward positive closing velocity.
* ``r_energy``       penalise total commanded acceleration magnitude.
* ``r_rl_residual``  penalise the RL residual (prefer the APN alone).
* ``r_smooth``       penalise action change (jerk).
* ``r_time``         flat per-step time cost.
* ``r_ZEM``          reward low Zero-Effort-Miss.
* ``r_kill``         terminal intercept bonus (miss + time scaled).
* ``r_crash``        terminal out-of-bounds penalty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from .stage_config import RewardConfig

__all__ = ["RewardFunction", "RewardBreakdown"]

_EPS = 1e-6


@dataclass
class RewardBreakdown:
    """Named reward components for logging / debugging."""

    progress: float = 0.0
    los_rate: float = 0.0
    closing_vel: float = 0.0
    energy: float = 0.0
    rl_residual: float = 0.0
    smooth: float = 0.0
    time: float = 0.0
    zem: float = 0.0
    kill: float = 0.0
    crash: float = 0.0
    total: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "progress": self.progress, "los_rate": self.los_rate,
            "closing_vel": self.closing_vel, "energy": self.energy,
            "rl_residual": self.rl_residual, "smooth": self.smooth,
            "time": self.time, "zem": self.zem, "kill": self.kill,
            "crash": self.crash, "total": self.total,
        }


class RewardFunction:
    """Computes the per-step reward and terminal flag for one environment."""

    def __init__(self, cfg: RewardConfig, *, v_max: float, a_max: float,
                 kill_radius: float):
        self.cfg = cfg
        self.v_max = float(max(v_max, _EPS))
        self.a_max = float(max(a_max, _EPS))
        self.kill_radius = float(max(kill_radius, _EPS))

    # ---------------------------------------------------------------- compute
    def compute(
        self,
        *,
        distance: float,
        prev_distance: float,
        closing_vel: float,
        LOS_rate_norm: float,
        a_total_norm: float,
        a_rl_norm: float,
        action: np.ndarray,
        prev_action: np.ndarray,
        ZEM: float,
        step_count: int,
        max_steps: int,
        killed: bool,
        out_of_bounds: bool,
    ) -> Tuple[float, bool, RewardBreakdown]:
        """Return ``(reward, terminated, breakdown)`` for a single step."""
        c = self.cfg
        b = RewardBreakdown()

        # --- r_progress -----------------------------------------------------
        raw = (prev_distance - distance) / max(distance, 1.0)
        if c.k_progress_normalize:
            raw = float(np.clip(raw, -1.0, 1.0))
        b.progress = c.k_progress * raw

        # --- r_LOS_rate -----------------------------------------------------
        b.los_rate = -c.k_LOS_rate * float(LOS_rate_norm)

        # --- r_closing_vel --------------------------------------------------
        b.closing_vel = c.k_closing_vel * max(0.0, float(closing_vel)) / self.v_max

        # --- r_energy -------------------------------------------------------
        b.energy = -c.k_energy * float(a_total_norm) ** 2

        # --- r_rl_residual --------------------------------------------------
        b.rl_residual = -c.k_rl_residual * float(a_rl_norm) ** 2

        # --- r_smooth -------------------------------------------------------
        da = np.asarray(action, dtype=np.float64) - np.asarray(prev_action, dtype=np.float64)
        b.smooth = -c.k_smooth * float(np.sum(da * da))

        # --- r_time ---------------------------------------------------------
        b.time = -c.k_time_penalty

        # --- r_ZEM ----------------------------------------------------------
        # Normalise by the current range so the term is O(1) regardless of
        # engagement distance (raw ZEM in metres would otherwise dominate
        # early in the episode).  Plan §10.3 lists ``-k_ZEM * ZEM``; dividing
        # by range keeps the same "low ZEM is good" semantics at a usable
        # scale (see docs/BUGS.md).
        b.zem = -c.k_ZEM * float(abs(ZEM)) / max(float(distance), 1.0)

        # --- terminal -------------------------------------------------------
        terminated = False
        if killed:
            bonus = c.k_kill_bonus
            if c.k_miss_distance_scale:
                bonus *= max(0.0, 1.0 - float(distance) / self.kill_radius)
            if c.k_time_bonus_scale and max_steps > 0:
                bonus *= max(0.0, 1.0 - float(step_count) / float(max_steps))
            b.kill = max(0.0, bonus)
            terminated = True
        if out_of_bounds and not killed:
            b.crash = c.k_crash_penalty
            terminated = True

        b.total = (b.progress + b.los_rate + b.closing_vel + b.energy
                   + b.rl_residual + b.smooth + b.time + b.zem + b.kill + b.crash)
        return float(b.total), bool(terminated), b
