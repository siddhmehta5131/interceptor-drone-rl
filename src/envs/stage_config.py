"""stage_config.py -- 3-stage curriculum for the APN + RL interceptor.

This module owns the dataclasses that describe *what a stage is*:

* :class:`RewardConfig`   -- dense + terminal reward weights (plan Phase 6)
* :class:`SpawnConfig`    -- target/interceptor spawn geometry and target limits
* :class:`InterceptConfig`-- kill radius, world bounds, episode step budget
* :class:`ObsStageConfig` -- per-stage observation layout + RL action scaling
* :class:`StageConfig`    -- the bundle for one curriculum stage

The authoritative stage definitions live in ``configs/stages.yaml``; when that
file is absent the built-in :data:`DEFAULT_STAGES` table (identical to the YAML)
is used, so importing this module never fails.

The three stages are:

=========  ===============  ===============================  =============
Stage      Target type      RL action scale                  Success
=========  ===============  ===============================  =============
stage_1    static           0.1 * a_max                       kill > 0.95
stage_2    order_1..3       0.3 * a_max                       kill > 0.90
stage_3    evasive          0.5 * a_max                       kill > 0.75
=========  ===============  ===============================  =============
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

__all__ = [
    "RewardConfig",
    "SpawnConfig",
    "InterceptConfig",
    "ObsStageConfig",
    "StageConfig",
    "STAGES",
    "DEFAULT_STAGES",
    "POLYNOMIAL_TARGET_TYPES",
    "load_stages",
    "stages_to_dict",
    "stages_from_dict",
]

#: Target types that follow a closed-form polynomial path.
POLYNOMIAL_TARGET_TYPES = ("order_1", "order_2", "order_3")

Range = Tuple[float, float]


def _as_range(value: Any, default: Range) -> Range:
    """Coerce a scalar / [lo, hi] / {min:, max:} into a ``(lo, hi)`` tuple."""
    if value is None:
        return default
    if isinstance(value, Mapping) and "min" in value and "max" in value:
        return (float(value["min"]), float(value["max"]))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (float(value), float(value))
    return default


def _sample(rng: np.random.Generator, r: Range) -> float:
    if r[0] == r[1]:
        return float(r[0])
    return float(rng.uniform(r[0], r[1]))


# ---------------------------------------------------------------------------
# Reward (plan Phase 6)
# ---------------------------------------------------------------------------

@dataclass
class RewardConfig:
    """Weights for every reward term (plan §10.2)."""

    # --- Dense per-step terms ---
    k_progress: float = 1.0             # reward for closing distance
    k_progress_normalize: bool = True   # divide by max(distance, 1.0)
    k_LOS_rate: float = 0.1             # penalise high LOS rate (want -> 0)
    k_energy: float = 0.02              # penalise |a_total|^2 (energy cost)
    k_rl_residual: float = 0.05         # penalise |da_RL|^2 (prefer APN alone)
    k_smooth: float = 0.03              # penalise |a(t) - a(t-1)|^2 (jerk)
    k_time_penalty: float = 0.001       # flat per-step time cost
    k_closing_vel: float = 0.1          # reward for positive closing velocity

    # --- Terminal bonuses ---
    k_kill_bonus: float = 100.0         # intercept reward
    k_miss_distance_scale: bool = True  # scale kill bonus by miss accuracy
    k_time_bonus_scale: bool = True     # scale kill bonus by speed of kill
    k_crash_penalty: float = -50.0      # out of bounds penalty
    k_ZEM: float = 0.05                 # reward for low Zero Effort Miss


# ---------------------------------------------------------------------------
# Spawn / target limits
# ---------------------------------------------------------------------------

@dataclass
class SpawnConfig:
    """Spawn geometry and target kinematic limits."""

    target_distance_range: Range = (20.0, 60.0)   # interceptor -> target (m)
    altitude_range: Range = (5.0, 50.0)           # world z band (m)
    interceptor_speed_range: Range = (15.0, 25.0) # initial closing speed (m/s)
    target_speed_range: Range = (0.0, 0.0)        # target speed (m/s)
    target_accel_range: Range = (0.0, 0.0)        # polynomial target accel (m/s^2)
    target_accel_g_limit: float = 3.0             # reactive accel cap (g)
    target_jerk_limit: float = 20.0               # reactive jerk cap (m/s^3)
    evasive_probability: float = 0.0              # P(flee) per segment
    path_end_distance: Range = (30.0, 80.0)       # polynomial endpoint distance (m)
    path_speed_cap: Range = (8.0, 20.0)           # polynomial peak speed (m/s)
    path_shape: str = "random"                    # "random" | "linear"
    cone_half_angle_deg: float = 45.0             # LOS spawn cone half-angle
    lateral_offset_max: float = 0.0               # extra lateral jitter on spawn


# ---------------------------------------------------------------------------
# Intercept / world
# ---------------------------------------------------------------------------

@dataclass
class InterceptConfig:
    """Engagement-level parameters."""

    kill_radius: float = 1.0        # intercept distance threshold (m)
    world_radius: float = 150.0     # out-of-bounds radius (m)
    max_episode_steps: int = 0      # 0 => derive from episode length and dt


# ---------------------------------------------------------------------------
# Observation layout
# ---------------------------------------------------------------------------

@dataclass
class ObsStageConfig:
    """Per-stage observation stacking + RL action scaling."""

    frame_dim: int = 16
    history_frames: int = 3
    history_skip: int = 2
    future_samples: int = 3
    future_skip: int = 5
    future_source: str = "pred"     # 'true' (oracle) or 'pred' (predictor)
    predictor: str = "const_vel"    # const_vel | kalman
    rl_action_scale: float = 1.0    # absolute m/s^2 added on top of APN

    @property
    def actor_dim(self) -> int:
        """Size of the actor observation (history + future blocks)."""
        return self.frame_dim * (1 + self.history_frames) + 7 * self.future_samples


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------

@dataclass
class StageConfig:
    """One curriculum stage."""

    id: str = "stage_1"
    name: str = "Stage 1"
    target_type: str = "static"
    target_visible: bool = True
    world_radius: float = 150.0
    episode_length_s: Range = (3.0, 5.0)
    max_training_steps: int = 2_000_000
    success_metric: str = "kill_rate"
    threshold: float = 0.95
    window: int = 100
    success_rate: float = 0.95
    rollback_threshold: float = 0.5
    apn_N_prime: float = 4.0
    apn_t_go_min: float = 0.05

    reward: RewardConfig = field(default_factory=RewardConfig)
    spawn: SpawnConfig = field(default_factory=SpawnConfig)
    intercept: InterceptConfig = field(default_factory=InterceptConfig)
    obs: ObsStageConfig = field(default_factory=ObsStageConfig)

    # -- helpers -------------------------------------------------------
    @property
    def is_polynomial(self) -> bool:
        return self.target_type in POLYNOMIAL_TARGET_TYPES

    @property
    def is_evasive(self) -> bool:
        return self.target_type == "evasive"

    def sample_episode_length(self, rng: np.random.Generator) -> float:
        return _sample(rng, self.episode_length_s)

    def validate(self, label: Optional[str] = None) -> "StageConfig":
        where = label or self.id
        valid = ("static",) + POLYNOMIAL_TARGET_TYPES + ("evasive",)
        if self.target_type not in valid:
            raise ValueError(f"{where}: target_type must be one of {valid}, "
                             f"got {self.target_type!r}")
        if self.intercept.kill_radius <= 0:
            raise ValueError(f"{where}: kill_radius must be > 0")
        if self.episode_length_s[0] <= 0 or self.episode_length_s[1] < self.episode_length_s[0]:
            raise ValueError(f"{where}: invalid episode_length_s range {self.episode_length_s}")
        if self.obs.frame_dim != 16:
            raise ValueError(f"{where}: obs.frame_dim must be 16, got {self.obs.frame_dim}")
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError(f"{where}: threshold must be in (0, 1]")
        return self


# ---------------------------------------------------------------------------
# Built-in default table (mirrors configs/stages.yaml)
# ---------------------------------------------------------------------------

def _make_default_stages() -> Dict[str, StageConfig]:
    s1 = StageConfig(
        id="stage_1", name="Static Target", target_type="static",
        world_radius=150.0, episode_length_s=(3.0, 5.0),
        max_training_steps=2_000_000, threshold=0.95, success_rate=0.95,
        apn_N_prime=4.0, apn_t_go_min=0.05,
        spawn=SpawnConfig(
            target_distance_range=(20.0, 60.0), altitude_range=(5.0, 50.0),
            interceptor_speed_range=(15.0, 25.0),
            target_speed_range=(0.0, 0.0), target_accel_range=(0.0, 0.0)),
        intercept=InterceptConfig(kill_radius=1.0, world_radius=150.0),
        obs=ObsStageConfig(rl_action_scale=1.5),   # 0.1 * a_max (a_max=15)
    )
    s2 = StageConfig(
        id="stage_2", name="Dynamic Target", target_type="order_1",
        world_radius=150.0, episode_length_s=(5.0, 10.0),
        max_training_steps=5_000_000, threshold=0.90, success_rate=0.90,
        apn_N_prime=4.0, apn_t_go_min=0.05,
        spawn=SpawnConfig(
            target_distance_range=(30.0, 80.0), altitude_range=(5.0, 60.0),
            interceptor_speed_range=(20.0, 30.0),
            target_speed_range=(5.0, 20.0), target_accel_range=(0.0, 5.0),
            path_end_distance=(40.0, 100.0), path_speed_cap=(10.0, 25.0)),
        intercept=InterceptConfig(kill_radius=1.0, world_radius=200.0),
        obs=ObsStageConfig(rl_action_scale=4.5),   # 0.3 * a_max
    )
    s3 = StageConfig(
        id="stage_3", name="Evasive Target", target_type="evasive",
        world_radius=250.0, episode_length_s=(8.0, 15.0),
        max_training_steps=10_000_000, threshold=0.75, success_rate=0.75,
        apn_N_prime=4.5, apn_t_go_min=0.05,
        spawn=SpawnConfig(
            target_distance_range=(40.0, 100.0), altitude_range=(5.0, 80.0),
            interceptor_speed_range=(25.0, 35.0),
            target_speed_range=(10.0, 30.0), target_accel_range=(0.0, 0.0),
            target_accel_g_limit=3.0, target_jerk_limit=20.0,
            evasive_probability=0.6),
        intercept=InterceptConfig(kill_radius=1.5, world_radius=250.0),
        obs=ObsStageConfig(rl_action_scale=7.5),   # 0.5 * a_max
    )
    return {s.id: s.validate() for s in (s1, s2, s3)}


DEFAULT_STAGES: Dict[str, StageConfig] = _make_default_stages()
# ``STAGES`` is what the rest of the codebase imports.  It defaults to the
# built-in table and is replaced by ``configs/stages.yaml`` if that file exists.
STAGES: Dict[str, StageConfig] = dict(DEFAULT_STAGES)


# ---------------------------------------------------------------------------
# YAML round-trip
# ---------------------------------------------------------------------------

def stages_to_dict(stages: Optional[Mapping[str, StageConfig]] = None) -> Dict[str, Any]:
    """Serialise a stage table to plain data (JSON/YAML friendly)."""
    stages = stages or STAGES
    return {"stages": {sid: _stage_plain(sc) for sid, sc in stages.items()}}


def _stage_plain(sc: StageConfig) -> Dict[str, Any]:
    out = asdict(sc)
    return out


def stages_from_dict(data: Mapping[str, Any]) -> Dict[str, StageConfig]:
    """Build a stage table from plain data (the inverse of ``stages_to_dict``)."""
    raw_stages = data.get("stages", data)
    out: Dict[str, StageConfig] = {}
    for sid, raw in raw_stages.items():
        out[sid] = _stage_from_mapping(sid, raw)
    return out


def _stage_from_mapping(sid: str, raw: Mapping[str, Any]) -> StageConfig:
    raw = dict(raw)
    raw.setdefault("id", sid)
    reward = RewardConfig(**{k: v for k, v in raw.pop("reward", {}).items()
                             if k in {f.name for f in fields(RewardConfig)}})
    spawn_raw = raw.pop("spawn", {})
    spawn = SpawnConfig(**{k: v for k, v in spawn_raw.items()
                           if k in {f.name for f in fields(SpawnConfig)}})
    for key in ("target_distance_range", "altitude_range", "interceptor_speed_range",
                "target_speed_range", "target_accel_range", "path_end_distance",
                "path_speed_cap"):
        if key in spawn_raw:
            setattr(spawn, key, _as_range(spawn_raw[key], getattr(spawn, key)))
    intercept_raw = raw.pop("intercept", {})
    intercept = InterceptConfig(**{k: v for k, v in intercept_raw.items()
                                   if k in {f.name for f in fields(InterceptConfig)}})
    obs_raw = raw.pop("obs", {})
    obs = ObsStageConfig(**{k: v for k, v in obs_raw.items()
                            if k in {f.name for f in fields(ObsStageConfig)}})

    scalar_names = {f.name for f in fields(StageConfig)} - {
        "reward", "spawn", "intercept", "obs"}
    scalars = {k: v for k, v in raw.items() if k in scalar_names}
    sc = StageConfig(**scalars, reward=reward, spawn=spawn,
                     intercept=intercept, obs=obs)
    sc.episode_length_s = _as_range(sc.episode_length_s, (3.0, 5.0))
    return sc.validate(sid)


def load_stages(path: "str | Path | None" = "configs/stages.yaml",
                *, use_builtin_fallback: bool = True) -> Dict[str, StageConfig]:
    """Load the stage table from YAML, falling back to the built-ins."""
    if path is not None:
        p = Path(path)
        if p.exists():
            try:
                import yaml
                data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                stages = stages_from_dict(data)
                if stages:
                    return stages
            except Exception:
                if not use_builtin_fallback:
                    raise
    return copy.deepcopy(DEFAULT_STAGES)


# Populate STAGES from the YAML if present.
STAGES = load_stages()
