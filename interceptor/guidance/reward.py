"""Guidance reward for the unified intercept environment (P5.9, Q8, F16).

The reward is the sum of four groups:

* a **potential-based progress** term, ``r_shape = gamma*Phi(s') - Phi(s)``
  with ``Phi(s) = -w_progress * d / r_init_max`` (EQ-GUIDE-1, EQ-GUIDE-2).
  Because the shaping is a potential difference, adding it does not change the
  optimal policy (Ng, Harada & Russell, 1999).
* **auxiliary shaping** already present in the notebook: velocity/LOS alignment,
  a LOS-rate penalty and a ZEM penalty (EQ-GUIDE-4).
* **action penalties**: residual authority squared and action change squared
  (EQ-GUIDE-5), plus a constant time cost (EQ-GUIDE-6).
* **terminal** kill bonus (with a speed bonus), crash penalty and divergence
  penalty (EQ-GUIDE-3).  The total is EQ-GUIDE-7.

The discount ``gamma`` is chosen from the episode horizon ``H`` so that a
terminal bonus at the first decision keeps a stated fraction ``rho``:
``gamma = rho**(1/H)`` (EQ-GUIDE-8).  With ``H = 1100`` steps (22 s at
``dt = 0.02``) and ``rho = 0.1``, ``gamma = 0.997 91`` and a 150-point kill
bonus is worth 15.0 at the first step; the old 0.995 left 0.6 (0.4 %).

The v2 notebook reward is kept verbatim in :func:`notebook_reward` for the
golden-fixture replay only; the default :class:`GuidanceReward` is the P5.9
contract.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

__all__ = [
    "GuidanceReward",
    "notebook_reward",
    "horizon_gamma",
    "load_reward",
    "discounted_terminal",
    "HORIZON_STEPS",
    "HORIZON_RETENTION",
]

#: Longest episode in the v2 notebook (22 s at 50 Hz); drives the discount.
HORIZON_STEPS = 1100
#: Fraction of a terminal bonus that should survive to the first decision.
HORIZON_RETENTION = 0.1


def horizon_gamma(horizon_steps: int, retention: float = HORIZON_RETENTION) -> float:
    """Discount that retains ``retention`` of a terminal reward (EQ-GUIDE-8).

    ``gamma = retention**(1/H)``.  For ``H = 1100`` and ``retention = 0.1``
    this gives 0.997 909, i.e. ``gamma**H = 0.1``.
    """
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive")
    if not 0.0 < retention < 1.0:
        raise ValueError("retention must lie in (0, 1)")
    return float(retention ** (1.0 / int(horizon_steps)))


@dataclass(frozen=True)
class GuidanceReward:
    """Weights and terminal values of the P5.9 guidance reward."""

    gamma: float = horizon_gamma(HORIZON_STEPS)
    w_progress: float = 50.0
    w_align: float = 0.35
    w_los: float = 0.15
    w_zem: float = 0.25
    w_res_rl: float = 0.04
    w_res_pure: float = 0.015
    w_smooth: float = 0.03
    c_time: float = 0.005
    kill_bonus: float = 150.0
    kill_speed_bonus: float = 50.0
    crash_penalty: float = 80.0
    diverge_penalty: float = 25.0

    @classmethod
    def from_mapping(cls, data: Mapping) -> "GuidanceReward":
        """Overlay a ``{value, unit, source}`` mapping on the defaults."""
        base = cls()
        values = {}
        for f in fields(cls):
            raw = data.get(f.name)
            if raw is None:
                continue
            if isinstance(raw, Mapping):
                raw = raw.get("value", raw.get("min"))
            values[f.name] = float(raw)
        unknown = set(data) - {f.name for f in fields(cls)} - {"source", "notes"}
        if unknown:
            raise ValueError(f"unknown reward keys: {sorted(unknown)}")
        return cls(**{**{f.name: getattr(base, f.name) for f in fields(cls)},
                      **values})

    def potential(self, distance: float, r_init_max: float) -> float:
        """``Phi(s) = -w_progress * d / r_init_max`` (EQ-GUIDE-1)."""
        return -self.w_progress * float(distance) / max(float(r_init_max), 1e-6)

    def compute(
        self,
        *,
        distance: float,
        prev_distance: float,
        los_vec,
        los_rate_vec,
        zem_norm: float,
        v_hat,
        action,
        prev_action,
        residual_weight: float,
        step_count: int,
        max_steps: int,
        r_init_max: float,
        killed: bool,
        crashed: bool,
        diverged: bool,
        min_miss_distance: float,
    ):
        """Return ``(reward, terms)`` for one transition (EQ-GUIDE-7)."""
        # EQ-GUIDE-2: potential-based progress.
        r_progress = (self.gamma * self.potential(distance, r_init_max)
                      - self.potential(prev_distance, r_init_max))
        # EQ-GUIDE-4: alignment, LOS-rate and ZEM shaping.
        r_align = self.w_align * float(np.dot(v_hat, los_vec))
        r_los = -self.w_los * float(np.linalg.norm(los_rate_vec))
        r_zem = -self.w_zem * float(zem_norm) / max(float(distance), 10.0)
        # EQ-GUIDE-5: residual authority and action change.
        act = np.asarray(action, dtype=float)
        r_residual = -float(residual_weight) * float(np.dot(act, act))
        r_smooth = -self.w_smooth * float(np.sum((act - np.asarray(prev_action, dtype=float)) ** 2))
        # EQ-GUIDE-6: constant time cost.
        r_time = -self.c_time

        reward = r_progress + r_align + r_los + r_zem + r_residual + r_smooth + r_time
        r_terminal = 0.0
        if killed:
            r_terminal = (self.kill_bonus + self.kill_speed_bonus
                          * (1.0 - step_count / max(max_steps, 1)))
        elif crashed:
            r_terminal = -self.crash_penalty
        elif diverged:
            r_terminal = -self.diverge_penalty * float(min_miss_distance) / max(float(r_init_max), 1e-6)
        reward += r_terminal

        terms = {
            "progress": float(r_progress), "align": float(r_align),
            "los": float(r_los), "zem": float(r_zem),
            "residual": float(r_residual), "smooth": float(r_smooth),
            "time": float(r_time), "terminal": float(r_terminal),
        }
        return float(reward), terms


def notebook_reward(
    *,
    distance: float,
    prev_distance: float,
    los_vec,
    los_rate_vec,
    zem_norm: float,
    v_hat,
    action,
    prev_action,
    residual_weight: float,
    step_count: int,
    max_steps: int,
    r_init_max: float,
    v_max: float,
    dt: float,
    killed: bool,
    crashed: bool,
    diverged: bool,
    min_miss_distance: float,
):
    """Exact v2 notebook reward, retained for golden-fixture replay.

    This is *not* the P5.9 reward; it keeps the fixture parity test honest
    about observations and dynamics without pretending the reward is unchanged.
    """
    r_progress = 2.5 * (prev_distance - distance) / max(v_max * dt, 1e-3)
    r_align = 0.35 * float(np.dot(v_hat, los_vec))
    r_los = -0.15 * float(np.linalg.norm(los_rate_vec))
    r_zem = -0.25 * float(zem_norm) / max(float(distance), 10.0)
    r_residual = -float(residual_weight) * float(np.linalg.norm(action)) ** 2
    r_smooth = -0.03 * float(np.sum((np.asarray(action) - np.asarray(prev_action)) ** 2))
    reward = r_progress + r_align + r_los + r_zem + r_residual + r_smooth - 0.005
    r_terminal = 0.0
    if killed:
        r_terminal = 150.0 + 50.0 * (1.0 - step_count / max(max_steps, 1))
    elif crashed:
        r_terminal = -80.0
    elif diverged:
        r_terminal = -25.0 * (min_miss_distance / max(r_init_max, 1e-6))
    reward += r_terminal
    terms = {
        "progress": float(r_progress), "align": float(r_align),
        "los": float(r_los), "zem": float(r_zem),
        "residual": float(r_residual), "smooth": float(r_smooth),
        "time": -0.005, "terminal": float(r_terminal),
    }
    return float(reward), terms


def _unwrap_leaf(value):
    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def load_reward(path: "str | Path | None" = "configs/guidance.yaml") -> GuidanceReward:
    """Load the sourced ``reward`` block; fall back to the P5.9 defaults."""
    base = GuidanceReward()
    if path is None:
        return base
    p = Path(path)
    if not p.exists():
        return base
    import yaml

    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    raw = data.get("reward", {})
    flat = {k: _unwrap_leaf(v) for k, v in raw.items()
            if k not in ("description", "source", "notes")}
    return GuidanceReward.from_mapping(flat)


def discounted_terminal(gamma: float, steps: int, bonus: float = 150.0) -> float:
    """``gamma**steps * bonus`` -- used by the P5.9 note and tests."""
    return float(gamma) ** int(steps) * float(bonus)
