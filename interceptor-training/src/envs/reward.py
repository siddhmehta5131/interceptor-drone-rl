"""reward.py  --  per-stage reward computation (plan §4 / §5 weights).

Design:

* :class:`RewardComputer` is created once per (stage, env) and computes the
  dense per-step reward, the terminal bonuses, the termination reasons and a
  component breakdown used for TensorBoard and the curriculum metrics.
* Every component is returned in a dict whose keys follow the plan §8
  ``reward_components/`` names so the SB3 callback can record them directly.

Terminal bonuses (plan §5.5 — the kill reward is a single term):

* crash (tilt-flip or hard ground contact) -> ``-k_crash``
* out of bounds                      -> ``-k_oob``
* kill                               -> ``+k_kill_bonus``, optionally scaled
  by miss distance (``* max(0, 1 - d/kill_radius)`` when
  ``k_miss_distance_scale``) and by time-to-kill (``* (1 - t/T_max)`` when
  ``k_time_bonus_scale``).

``k_progress_normalize`` divides the progress delta by ``max(distance, 1.0)``
(plan §5.5), i.e. relative progress, not an absolute meter count.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..physics.constants import PH_C_HOVER
from .stage_config import RewardConfig

__all__ = ["RewardComputer", "TERMINAL_INFO_KEYS"]

# Keys guaranteed present in the info dict of every step (last-step values).
TERMINAL_INFO_KEYS = (
    "r_alive", "r_alt", "r_tilt", "r_angvel", "r_thrust", "r_smooth",
    "r_velocity_alignment", "r_progress_delta", "r_kill_bonus",
    "r_time_penalty", "r_miss_distance",
    "alt_err", "tilt", "angvel_norm", "miss_distance",
    "time_to_intercept", "killed", "crashed", "out_of_bounds",
)


class RewardComputer:
    """Compute per-step reward + termination for one stage config."""

    def __init__(self, cfg: RewardConfig, *, kill_radius: float = 0.5, max_episode_steps: int = 1000):
        self.cfg = cfg
        self.kill_radius = float(kill_radius)
        self.max_episode_steps = int(max_episode_steps)

    # ------------------------------------------------------------------ API

    def compute(
        self,
        *,
        R_WB: np.ndarray,
        v_WB: np.ndarray,
        omega_B: np.ndarray,
        p_WB: np.ndarray,
        action: np.ndarray,
        prev_action: np.ndarray,
        c_cmd: float,
        alt_err: float,
        step_count: int,
        crashed_flip: bool,
        crashed_ground: bool,
        out_of_bounds: bool,
        killed: bool,
        distance: Optional[float] = None,
        prev_distance: Optional[float] = None,
        target_visible: bool = False,
        los_world: Optional[np.ndarray] = None,
    ) -> tuple:
        """Return ``(reward, terminated, components, metrics)``.

        ``components`` : per-step reward breakdown (for TB + info).
        ``metrics``    : episode-level numbers filled at termination.
        """
        cfg = self.cfg
        R = np.asarray(R_WB, dtype=np.float64)
        v = np.asarray(v_WB, dtype=np.float64)

        tilt = float(np.arccos(np.clip(R[2, 2], -1.0, 1.0)))
        angvel_n = float(np.linalg.norm(omega_B))

        comp: Dict[str, float] = {}

        # ---- dense terms ---------------------------------------------------
        comp["r_alive"] = cfg.k_alive
        comp["r_alt"] = -cfg.k_alt * abs(alt_err)
        comp["r_tilt"] = -cfg.k_tilt * tilt
        comp["r_angvel"] = -cfg.k_angvel * max(0.0, angvel_n - cfg.omega_safe) ** 2
        comp["r_thrust"] = -cfg.k_thrust * abs(float(c_cmd) - PH_C_HOVER)
        comp["r_smooth"] = -cfg.k_smooth * float(np.sum((action - prev_action) ** 2))
        comp["r_time_penalty"] = -cfg.k_time_penalty

        # ---- target-relative terms (stages 2+) -----------------------------
        comp["r_velocity_alignment"] = 0.0
        comp["r_progress_delta"] = 0.0
        if distance is not None and target_visible:
            # velocity alignment: max(0, cos angle between v and LOS)
            align = 0.0
            v_norm = float(np.linalg.norm(v))
            if v_norm > 1e-3 and los_world is not None and distance > 1e-6:
                align = float(np.dot(v / v_norm, los_world))
                align = max(0.0, float(np.clip(align, -1.0, 1.0)))
            comp["r_velocity_alignment"] = cfg.k_velocity_alignment * align

            progress = 0.0
            if prev_distance is not None:
                progress = float(prev_distance - distance)
                if cfg.k_progress_normalize:
                    progress = progress / max(float(distance), 1.0)
                    progress = float(np.clip(progress, -1.0, 1.0))
            comp["r_progress_delta"] = cfg.k_progress_delta * progress

        # ---- terminal bonuses ----------------------------------------------
        # Plan §5.5: the kill bonus is a single term, optionally scaled by
        # how close the kill was (miss distance) and how fast it happened.
        comp["r_kill_bonus"] = 0.0
        comp["r_miss_distance"] = 0.0
        terminated = False
        crashed = crashed_flip or crashed_ground

        if crashed:
            comp["r_crash"] = -cfg.k_crash
            terminated = True
        if out_of_bounds:
            comp["r_oob"] = -cfg.k_oob
            terminated = True
        if killed:
            bonus = float(cfg.k_kill_bonus)
            if cfg.k_miss_distance_scale and distance is not None:
                bonus *= max(0.0, 1.0 - float(distance) / max(1e-9, self.kill_radius))
            if cfg.k_time_bonus_scale:
                t_frac = float(np.clip(step_count / max(1, self.max_episode_steps), 0.0, 1.0))
                bonus *= (1.0 - t_frac)
            comp["r_kill_bonus"] = max(0.0, bonus)
            terminated = True

        # NOTE: explicit left fold, NOT builtin sum() -- Python 3.12+ sum()
        # uses compensated summation that can differ by 1 ULP from
        # hover_env's naive `a + b + c + ...` expression (bit-parity).
        reward = 0.0
        for _v in comp.values():
            reward += _v
        reward = float(reward)

        # ---- episode metrics ------------------------------------------------
        metrics = {
            "alt_err": abs(alt_err),
            "tilt": tilt,
            "angvel_norm": angvel_n,
            "miss_distance": float(distance) if distance is not None else float("nan"),
            "time_to_intercept": float(step_count),
            "killed": bool(killed),
            "crashed": bool(crashed),
            "out_of_bounds": bool(out_of_bounds),
        }
        return reward, terminated, comp, metrics