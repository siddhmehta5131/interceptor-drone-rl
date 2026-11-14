"""Asymmetric observation builder for the unified guidance environment (P5.8).

Actor: 85 values = four 16-value frames (m=3, p=2) + three 7-value future
samples (n=3, q=5).  Critic: 16 privileged values.

Normalisation is per platform.  ``ObservationScales`` holds every divisor in
one place and defaults to the :class:`~interceptor.guidance.platform.
InterceptorLimits` envelope (``r_init_max`` for lengths, ``v_max`` for speeds,
``max_time`` for times, ``a_n_max`` / ``a_target_max`` for accelerations), which
is itself loaded from ``configs/platforms.yaml``.  The explicit
``observation_scales`` block in that file overrides the derived defaults, so
the actor's inputs stay O(1) on every vehicle instead of being hard-coded for
one (fix F18).  The privileged critic block is unchanged.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from .platform import InterceptorLimits, PLATFORMS
from .maneuvers import PlatformType
from .seeker import SeekerSensorFilter

__all__ = [
    "AsymmetricObsBuilder",
    "ObservationScales",
    "load_observation_scales",
    "FRAME_DIM",
    "FUTURE_SAMPLE_DIM",
    "PRIVILEGED_DIM",
]


@dataclass(frozen=True)
class ObservationScales:
    """Per-platform divisors for every observation block (P5.8, fix F18)."""

    rel_pos: float
    rel_vel: float
    interceptor_vel: float
    range: float
    closing_vel: float
    los_rate: float
    t_go: float
    future_pos: float
    future_vel: float
    time_remaining: float
    optimal_accel: float
    optimal_vel: float
    target_accel: float

    @classmethod
    def from_limits(cls, limits: InterceptorLimits) -> "ObservationScales":
        """Derive the scales from the platform envelope (notebook defaults).

        This reproduces the pre-P5.8 behaviour exactly, so the golden-fixture
        replay is unchanged; explicit config values override it per platform.
        """
        return cls(
            rel_pos=float(limits.r_init_max),
            rel_vel=float(limits.v_max),
            interceptor_vel=float(limits.v_max),
            range=float(limits.r_init_max),
            closing_vel=float(limits.v_max),
            los_rate=1.0,
            t_go=float(limits.max_time),
            future_pos=float(limits.r_init_max),
            future_vel=float(limits.v_max),
            time_remaining=float(limits.max_time),
            optimal_accel=float(limits.a_n_max),
            optimal_vel=float(limits.v_max),
            target_accel=max(float(limits.a_target_max), 1.0),
        )

    @classmethod
    def from_mapping(cls, limits: InterceptorLimits,
                     data: Mapping) -> "ObservationScales":
        """Overlay a ``{value, unit, source}`` (or bare numeric) mapping."""
        base = cls.from_limits(limits)
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
            raise ValueError(f"unknown observation scale keys: {sorted(unknown)}")
        return cls(**{**{f.name: getattr(base, f.name) for f in fields(cls)},
                      **values})


def _unwrap_leaf(value):
    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def load_observation_scales(
    path: "str | Path | None" = "configs/platforms.yaml",
    limits_by_platform: Optional[Mapping[PlatformType, InterceptorLimits]] = None,
) -> dict:
    """Load per-platform ``observation_scales`` from config (P5.8).

    Falls back to :meth:`ObservationScales.from_limits` when the file or the
    block is missing, so importing the guidance package never depends on the
    working directory.
    """
    limits_by_platform = dict(limits_by_platform or PLATFORMS)
    out = {p: ObservationScales.from_limits(lim)
           for p, lim in limits_by_platform.items()}
    if path is None:
        return out
    p = Path(path)
    if not p.exists():
        return out
    import yaml

    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    root = data.get("observation_scales", {})
    for platform in PlatformType:
        raw = root.get(platform.value)
        if not raw:
            continue
        flat = {k: _unwrap_leaf(v) for k, v in raw.items()
                if k not in ("description", "source", "notes")}
        out[platform] = ObservationScales.from_mapping(
            limits_by_platform[platform], flat)
    return out


class AsymmetricObsBuilder:
    # EQ-OBS-1: one normalised 16-value frame; EQ-OBS-2: history + future.
    FRAME_DIM = 16
    FUTURE_SAMPLE_DIM = 7
    PRIVILEGED_DIM = 16

    def __init__(self, limits: InterceptorLimits, m: int = 3, p: int = 2,
                 n: int = 3, q: int = 5,
                 scales: Optional[ObservationScales] = None,
                 ablate: tuple = ()):
        """``ablate`` (P9.2) may contain ``"history"`` (zero the m older frames)
        and/or ``"prediction"`` (zero the future block), and/or ``"privileged"``
        (zero the 16 critic-only values, i.e. no privileged critic).  Dimensions are
        unchanged so checkpoints and the critic layout stay comparable."""
        bad = set(ablate) - {"history", "prediction", "privileged"}
        if bad:
            raise ValueError(f"unknown ablation(s) {sorted(bad)}")
        self.ablate = tuple(ablate)
        self.limits = limits
        self.scales = scales if scales is not None else ObservationScales.from_limits(limits)
        self.m, self.p, self.n, self.q = int(m), int(p), int(n), int(q)
        self.actor_dim = self.FRAME_DIM * (1 + self.m) + self.FUTURE_SAMPLE_DIM * self.n
        self.total_dim = self.actor_dim + self.PRIVILEGED_DIM
        self._history = deque(maxlen=self.m * self.p + 1)

    def reset(self) -> None:
        self._history.clear()

    def _frame(self, p_M, v_M, est_p_T, est_v_T, visible) -> np.ndarray:
        lim, s = self.limits, self.scales
        rel_pos = np.asarray(est_p_T) - np.asarray(p_M)
        rel_vel = np.asarray(est_v_T) - np.asarray(v_M)
        rng = float(np.linalg.norm(rel_pos))
        rhat = rel_pos / max(rng, 1e-6)
        closing = float(-np.dot(rel_vel, rhat))
        r_eff = max(rng, 15.0)
        los_rate = np.cross(rel_pos, rel_vel) / r_eff ** 2
        t_go = float(np.clip(rng / max(closing, 0.1), 0.0, lim.max_time))
        return np.clip(np.concatenate([
            rel_pos / s.rel_pos,
            rel_vel / s.rel_vel,
            np.asarray(v_M) / s.interceptor_vel,
            [rng / s.range, closing / s.closing_vel],
            los_rate / s.los_rate,
            [t_go / s.t_go, 1.0 if visible else 0.0],
        ]), -10.0, 10.0).astype(np.float32)

    def build(self, *, p_M, v_M, est_p_T, est_v_T, est_a_T, visible: bool,
              dt: float, time_remaining: float, optimal_accel, optimal_vel,
              true_future_pos, true_future_vel, true_acc) -> np.ndarray:
        lim, s = self.limits, self.scales
        self._history.appendleft(self._frame(p_M, v_M, est_p_T, est_v_T, visible))
        frames = list(self._history)
        history = np.concatenate([
            frames[k * self.p] if k * self.p < len(frames) else frames[-1]
            for k in range(self.m + 1)
        ])
        future = []
        for k in range(self.n):
            ahead = float(self.q * (k + 1)) * dt
            fp, fv = SeekerSensorFilter.kalman_predict(est_p_T, est_v_T, est_a_T, ahead)
            rel_p = (fp - (np.asarray(p_M) + np.asarray(v_M) * ahead)) / s.future_pos
            rel_v = (fv - np.asarray(v_M)) / s.future_vel
            future.append(np.concatenate([rel_p, rel_v, [1.0 if visible else 0.0]]))
        if "history" in self.ablate:
            history = history.copy()
            history[self.FRAME_DIM:] = 0.0
        if "prediction" in self.ablate:
            future = [np.zeros_like(f) for f in future]
        actor = np.clip(np.concatenate([history] + future), -10.0, 10.0)
        priv = np.clip(np.concatenate([
            [time_remaining / s.time_remaining],
            np.asarray(optimal_accel) / s.optimal_accel,
            np.asarray(optimal_vel) / s.optimal_vel,
            (np.asarray(true_future_pos) - np.asarray(p_M)) / s.future_pos,
            (np.asarray(true_future_vel) - np.asarray(v_M)) / s.future_vel,
            np.asarray(true_acc) / s.target_accel,
        ]), -10.0, 10.0)
        if "privileged" in self.ablate:
            priv = np.zeros_like(priv)
        return np.concatenate([actor, priv]).astype(np.float32)
