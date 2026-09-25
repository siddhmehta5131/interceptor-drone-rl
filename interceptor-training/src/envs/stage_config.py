"""stage_config.py  --  dataclasses and the 8-stage curriculum table.

Every numeric default below comes verbatim from ``implementation_plan.md``
v1.0 (the stage YAML + the tuned reward weights).  ``k_crash``/``k_oob``
(200), ``k_angvel`` (0.02) and ``k_smooth`` (0.05) are constant across all
stages per the plan.

Interpretation decisions (plan §5.5 formulas used verbatim):
  * ``RewardConfig.k_miss_distance_scale`` (True in stages 3+):
      the KILL bonus is scaled by miss distance: on an intercept the bonus
      becomes ``k_kill_bonus * max(0, 1 - distance/kill_radius)``, so kills
      at the edge of the capture radius pay ~0 and dead-centre hits pay the
      full bonus.
  * ``RewardConfig.k_time_bonus_scale`` (True in stages 4+):
      the KILL bonus is additionally scaled by time-to-kill:
      ``k_kill_bonus * (1 - steps/max_episode_steps)`` (full bonus for an
      instantaneous kill, 0 at timeout).  Both gates multiply together.
  * ``RewardConfig.k_progress_normalize`` (True in stages 6+):
      progress ``prev_distance - distance`` is divided by
      ``max(distance, 1.0)`` and clipped to [-1, 1] (relative progress).
  * ``k_time_penalty`` is applied flat per step: ``r_time = -k_time_penalty``.
  * ``k_velocity_alignment`` uses ``max(0, dot(v_hat, los_hat))`` so the
    term is zero when stationary/driving away (progress handles direction).
  * Vertical world bound for target stages: ``z_hi = 40.0`` m (docs §bound).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from ..physics.constants import (
    PH_GROUND_CONTACT_V_THRESHOLD,
    PH_TILT_CRASH_THRESHOLD,
)

__all__ = [
    "STAGES",
    "RewardConfig",
    "SpawnConfig",
    "InterceptConfig",
    "StageConfig",
    "make_obs_config",
]

# Vertical world boundary for target stages (m).  Stage 1 keeps the
# hover_env convention ``_z_hi = target_alt + 10``.
TARGET_STAGE_Z_HI = 40.0

# Look-ahead prediction horizon used when ``InterceptConfig.lookahead`` is
# enabled (seconds of target motion predicted into the observation).
LOOKAHEAD_HORIZON_S = 0.5


# ============================================================================
# Reward configuration
# ============================================================================

@dataclass
class RewardConfig:
    """Per-stage reward weights (plan defaults; crash/oob/angvel/smooth const).

    ``k_miss_distance_scale`` and ``k_time_bonus_scale`` are boolean gates
    that enable the scaled terminal bonuses described in the module docstring.
    """

    # -- dense shaping --------------------------------------------------------
    k_alive: float = 0.0
    k_alt: float = 0.0
    k_tilt: float = 0.0
    k_angvel: float = 0.02
    omega_safe: float = 3.0
    k_thrust: float = 0.0
    k_smooth: float = 0.05
    k_velocity_alignment: float = 0.0
    k_progress_delta: float = 0.0
    k_time_penalty: float = 0.0

    # -- terminal bonuses ------------------------------------------------------
    k_kill_bonus: float = 0.0
    k_miss_distance_scale: bool = False
    k_time_bonus_scale: bool = False
    k_progress_normalize: bool = False

    # -- safety ----------------------------------------------------------------
    k_crash: float = 200.0
    k_oob: float = 200.0
    tilt_crash_threshold: float = PH_TILT_CRASH_THRESHOLD
    ground_contact_v_threshold: float = PH_GROUND_CONTACT_V_THRESHOLD


# ============================================================================
# Spawn / intercept / observation configuration
# ============================================================================

@dataclass
class SpawnConfig:
    """Initial placement/kinematics of the target (or waypoint)."""

    target_distance_range: Tuple[float, float] = (10.0, 30.0)
    waypoint_distance_range: Tuple[float, float] = (8.0, 20.0)
    hemisphere: str = "forward"          # "forward" = +x cone around spawn
    cone_half_angle_deg: float = 45.0
    lateral_offset_max: float = 0.0      # stage 3 only
    speed_range: Tuple[float, float] = (2.0, 8.0)
    target_accel_g_limit: float = 0.0    # G units (x9.81 => m/s^2)
    target_jerk_limit: float = 0.0       # m/s^3
    evasive_probability: float = 0.0

    # target altitude band (m) -- target spawned inside this vertical window
    altitude_range: Tuple[float, float] = (3.0, 8.0)


@dataclass
class InterceptConfig:
    """Interception geometry / behaviour."""

    kill_radius: float = 0.5
    lookahead_enabled: bool = True       # True -> obs uses predicted target pos


@dataclass
class ObsStageConfig:
    """Observation settings fixed per stage (history is per-algo config)."""

    include_target: bool = True
    obs_noise_std: float = 0.01          # plan decision: sigma=0.01 everywhere


# ============================================================================
# Stage configuration
# ============================================================================

@dataclass
class StageConfig:
    """One curriculum stage.  ``id`` matches the plan's ``stage_N`` keys."""

    id: str                              # 'stage_1' ... 'stage_8'
    name: str
    target_type: str                     # none|static_waypoint|static|order_1..3|evasive
    target_visible: bool
    world_radius: float                  # square XY bound (+-radius)
    max_episode_steps: int
    max_training_steps: int
    success_metric: str                  # episode_reward_mean|mean_final_distance|kill_rate
    threshold: float
    window: int
    success_rate: float
    reward: RewardConfig = field(default_factory=RewardConfig)
    spawn: SpawnConfig = field(default_factory=SpawnConfig)
    intercept: InterceptConfig = field(default_factory=InterceptConfig)
    obs: ObsStageConfig = field(default_factory=ObsStageConfig)

    # optional features (default OFF for all stages -- Stage-1 parity)
    domain_randomize: bool = False
    wind_enabled: bool = False
    ground_effect: bool = False

    # rollback policy (plan §6.2): None -> 50% of this stage's success_rate
    rollback_threshold: Optional[float] = None


def make_obs_config(
    stage: StageConfig,
    history_frames: int = 0,
    history_skip: int = 1,
) -> "tuple":
    """Combine per-stage + per-algorithm obs settings.

    Returns ``(include_target, history_frames, history_skip, obs_noise_std)``.
    Stage 1 (no target) never stacks history (14-dim stays 14-dim).
    """
    if not stage.obs.include_target:
        return (False, 0, 1, stage.obs.obs_noise_std)
    return (
        True,
        int(history_frames),
        max(1, int(history_skip)),
        stage.obs.obs_noise_std,
    )


# ============================================================================
# The 8-stage curriculum table (implementation_plan.md v1.0)
# ============================================================================

def _build_stages() -> Dict[str, StageConfig]:
    stages: Dict[str, StageConfig] = {}

    # -- Stage 1: Hover/Attitude ---------------------------------------------
    stages["stage_1"] = StageConfig(
        id="stage_1",
        name="Hover/Attitude",
        target_type="none",
        target_visible=False,
        world_radius=15.0,
        max_episode_steps=1000,
        max_training_steps=3_000_000,
        success_metric="episode_reward_mean",
        threshold=60.0,
        window=100,
        success_rate=0.85,
        reward=RewardConfig(
            k_alive=0.10, k_alt=0.15, k_tilt=0.40,
            k_angvel=0.02, omega_safe=3.0, k_thrust=0.15, k_smooth=0.05,
            k_crash=200.0, k_oob=200.0,
        ),
        obs=ObsStageConfig(include_target=False, obs_noise_std=0.01),
    )

    # -- Stage 2: Directional Flight / Waypoint ------------------------------
    stages["stage_2"] = StageConfig(
        id="stage_2",
        name="Directional Flight/Waypoint",
        target_type="static_waypoint",
        target_visible=True,
        world_radius=30.0,
        max_episode_steps=1500,
        max_training_steps=5_000_000,
        success_metric="mean_final_distance",
        threshold=2.0,
        window=100,
        success_rate=0.80,
        spawn=SpawnConfig(
            waypoint_distance_range=(8.0, 20.0),
            hemisphere="forward",
            cone_half_angle_deg=60.0,
        ),
        reward=RewardConfig(
            k_alive=0.05, k_tilt=0.20, k_velocity_alignment=0.30,
            k_progress_delta=0.15, k_angvel=0.02, k_smooth=0.05,
            k_crash=200.0, k_oob=200.0,
        ),
        obs=ObsStageConfig(include_target=True, obs_noise_std=0.01),
    )

    # -- Stage 3: Static Target Intercept (no time penalty) ------------------
    stages["stage_3"] = StageConfig(
        id="stage_3",
        name="Static Target Intercept",
        target_type="static",
        target_visible=True,
        world_radius=40.0,
        max_episode_steps=2000,
        max_training_steps=5_000_000,
        success_metric="kill_rate",
        threshold=0.70,
        window=100,
        success_rate=0.70,
        spawn=SpawnConfig(
            target_distance_range=(10.0, 30.0),
            hemisphere="forward",
            cone_half_angle_deg=45.0,
            lateral_offset_max=5.0,
        ),
        intercept=InterceptConfig(kill_radius=0.5, lookahead_enabled=True),
        reward=RewardConfig(
            k_velocity_alignment=0.30, k_progress_delta=0.15,
            k_kill_bonus=500.0, k_miss_distance_scale=True,
            k_angvel=0.02, k_smooth=0.05, k_crash=200.0, k_oob=200.0,
        ),
        obs=ObsStageConfig(include_target=True, obs_noise_std=0.01),
    )

    # -- Stage 4: Static Target (with time penalty) ---------------------------
    stages["stage_4"] = StageConfig(
        id="stage_4",
        name="Static Target Intercept (time)",
        target_type="static",
        target_visible=True,
        world_radius=40.0,
        max_episode_steps=2000,
        max_training_steps=5_000_000,
        success_metric="kill_rate",
        threshold=0.75,
        window=100,
        success_rate=0.75,
        spawn=SpawnConfig(
            target_distance_range=(10.0, 30.0),
            hemisphere="forward",
            cone_half_angle_deg=45.0,
        ),
        intercept=InterceptConfig(kill_radius=0.5, lookahead_enabled=True),
        reward=RewardConfig(
            k_velocity_alignment=0.30, k_progress_delta=0.15,
            k_kill_bonus=500.0, k_miss_distance_scale=True,
            k_time_penalty=0.02, k_time_bonus_scale=True,
            k_angvel=0.02, k_smooth=0.05, k_crash=200.0, k_oob=200.0,
        ),
        obs=ObsStageConfig(include_target=True, obs_noise_std=0.01),
    )

    # -- Stage 5: Order-1 moving target (constant velocity) -------------------
    stages["stage_5"] = StageConfig(
        id="stage_5",
        name="Order-1 Moving Target",
        target_type="order_1",
        target_visible=True,
        world_radius=60.0,
        max_episode_steps=2500,
        max_training_steps=8_000_000,
        success_metric="kill_rate",
        threshold=0.65,
        window=100,
        success_rate=0.65,
        spawn=SpawnConfig(
            target_distance_range=(15.0, 40.0),
            speed_range=(2.0, 8.0),
            hemisphere="forward",
            cone_half_angle_deg=45.0,
            target_accel_g_limit=0.0,
        ),
        intercept=InterceptConfig(kill_radius=0.5, lookahead_enabled=True),
        reward=RewardConfig(
            k_velocity_alignment=0.30, k_progress_delta=0.15,
            k_kill_bonus=500.0, k_miss_distance_scale=True,
            k_time_penalty=0.02, k_time_bonus_scale=True,
            k_angvel=0.02, k_smooth=0.05, k_crash=200.0, k_oob=200.0,
        ),
        obs=ObsStageConfig(include_target=True, obs_noise_std=0.01),
    )

    # -- Stage 6: Order-2 parabolic target (accelerating) ---------------------
    stages["stage_6"] = StageConfig(
        id="stage_6",
        name="Order-2 Parabolic Target",
        target_type="order_2",
        target_visible=True,
        world_radius=80.0,
        max_episode_steps=3000,
        max_training_steps=10_000_000,
        success_metric="kill_rate",
        threshold=0.60,
        window=100,
        success_rate=0.60,
        spawn=SpawnConfig(
            target_distance_range=(15.0, 50.0),
            speed_range=(2.0, 8.0),
            hemisphere="forward",
            cone_half_angle_deg=45.0,
            target_accel_g_limit=2.0,
        ),
        intercept=InterceptConfig(kill_radius=0.5, lookahead_enabled=True),
        reward=RewardConfig(
            k_velocity_alignment=0.30, k_progress_delta=0.15,
            k_kill_bonus=500.0, k_miss_distance_scale=True,
            k_time_penalty=0.02, k_time_bonus_scale=True,
            k_progress_normalize=True,
            k_angvel=0.02, k_smooth=0.05, k_crash=200.0, k_oob=200.0,
        ),
        obs=ObsStageConfig(include_target=True, obs_noise_std=0.01),
    )

    # -- Stage 7: Order-3 cubic target (jerk-limited) --------------------------
    stages["stage_7"] = StageConfig(
        id="stage_7",
        name="Order-3 Cubic Target",
        target_type="order_3",
        target_visible=True,
        world_radius=100.0,
        max_episode_steps=3000,
        max_training_steps=12_000_000,
        success_metric="kill_rate",
        threshold=0.55,
        window=100,
        success_rate=0.55,
        spawn=SpawnConfig(
            target_distance_range=(15.0, 60.0),
            speed_range=(2.0, 10.0),
            hemisphere="forward",
            cone_half_angle_deg=45.0,
            target_accel_g_limit=2.0,
            target_jerk_limit=5.0,
        ),
        intercept=InterceptConfig(kill_radius=0.5, lookahead_enabled=True),
        reward=RewardConfig(
            k_velocity_alignment=0.30, k_progress_delta=0.15,
            k_kill_bonus=500.0, k_miss_distance_scale=True,
            k_time_penalty=0.02, k_time_bonus_scale=True,
            k_progress_normalize=True,
            k_angvel=0.02, k_smooth=0.05, k_crash=200.0, k_oob=200.0,
        ),
        obs=ObsStageConfig(include_target=True, obs_noise_std=0.01),
    )

    # -- Stage 8: Complex / Evasive (stretch goal; configs skip by default) ---
    stages["stage_8"] = StageConfig(
        id="stage_8",
        name="Complex/Evasive Target",
        target_type="evasive",
        target_visible=True,
        world_radius=120.0,
        max_episode_steps=3000,
        max_training_steps=15_000_000,
        success_metric="kill_rate",
        threshold=0.50,
        window=100,
        success_rate=0.50,
        spawn=SpawnConfig(
            target_distance_range=(20.0, 70.0),
            speed_range=(3.0, 12.0),
            hemisphere="forward",
            cone_half_angle_deg=45.0,
            target_accel_g_limit=3.0,
            target_jerk_limit=8.0,
            evasive_probability=0.5,
        ),
        intercept=InterceptConfig(kill_radius=0.5, lookahead_enabled=True),
        reward=RewardConfig(
            k_velocity_alignment=0.30, k_progress_delta=0.15,
            k_kill_bonus=500.0, k_miss_distance_scale=True,
            k_time_penalty=0.02, k_time_bonus_scale=True,
            k_progress_normalize=True,
            k_angvel=0.02, k_smooth=0.05, k_crash=200.0, k_oob=200.0,
        ),
        obs=ObsStageConfig(include_target=True, obs_noise_std=0.01),
    )

    return stages


STAGES: Dict[str, StageConfig] = _build_stages()