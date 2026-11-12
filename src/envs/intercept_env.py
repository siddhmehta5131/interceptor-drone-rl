"""intercept_env.py -- Gymnasium environment for the APN + RL interceptor.

The environment implements the plan §8.1 step order exactly:

1. decode the RL action into a residual acceleration,
2. advance the target,
3. compute the APN command,
4. sum APN + RL, clamp to ``a_max``,
5. convert the total acceleration into a heading-rate command,
6. integrate the interceptor kinematics,
7. run the kill / out-of-bounds / timeout checks,
8. compute the reward,
9. build the observation,
10. append to the trajectory log.

The returned observation is ``concat(actor_obs, privileged_obs)`` so the
retained :class:`~src.training.asymmetric_policy.AsymmetricActorCriticPolicy`
can slice the actor block while the critic sees the privileged columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # gymnasium is required by SB3; keep the import lazy-friendly.
    import gymnasium as gym
    from gymnasium import spaces
    _HAVE_GYM = True
except Exception:  # pragma: no cover
    gym = None
    spaces = None
    _HAVE_GYM = False

from ..guidance.apn import compute_apn_command
from ..guidance.kinematic_model import (
    InterceptorState, KinematicModel, quat_from_vectors,
)
from ..guidance.optimal_trajectory import compute_optimal_intercept
from ..prediction.base import make_predictor
from .obs_builder import PRIVILEGED_FIELDS, ObsBuilder
from .reward import RewardFunction
from .stage_config import ObsStageConfig, StageConfig
from .target_generator import TargetGenerator

__all__ = ["InterceptEnv", "InterceptorConfig", "APNConfig",
           "DEFAULT_PRIVILEGED_FIELDS"]

_EPS = 1e-9

#: Critic-only fields used by default (plan §7.4).
DEFAULT_PRIVILEGED_FIELDS: Tuple[str, ...] = (
    "time_remaining",
    "optimal_accel",
    "optimal_vel",
    "target_true_future_pos",
    "target_true_future_vel",
    "target_true_accel",
)


@dataclass
class InterceptorConfig:
    """Interceptor kinematic limits + simulation timestep (plan §6)."""

    v_max: float = 50.0
    a_max: float = 15.0
    omega_max: float = 3.0
    speed_ratio: float = 1.5
    dt: float = 0.02


@dataclass
class APNConfig:
    """Augmented Proportional Navigation parameters (plan §2.3)."""

    N_prime: float = 4.0
    t_go_min: float = 0.05
    t_go_max: float = 20.0


class _StateAdapter:
    """Adapt ``target.state_at`` (a callable) to the oracle path protocol."""

    def __init__(self, fn):
        self._fn = fn

    def position_at(self, t: float) -> np.ndarray:
        return self._fn(t)[0]

    def velocity_at(self, t: float) -> np.ndarray:
        return self._fn(t)[1]

    def acceleration_at(self, t: float) -> np.ndarray:
        return self._fn(t)[2]

    def __call__(self, t: float) -> np.ndarray:  # pragma: no cover
        return self._fn(t)[0]


_Base = gym.Env if _HAVE_GYM else object


class InterceptEnv(_Base):
    """Single-environment APN + RL interceptor (plan §8)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        stage_config: StageConfig,
        obs_config: Optional[ObsStageConfig] = None,
        apn_config: Optional[APNConfig] = None,
        interceptor_config: Optional[InterceptorConfig] = None,
        privileged_fields: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
    ):
        if not _HAVE_GYM:
            raise ImportError("gymnasium is required to use InterceptEnv")

        self.stage = stage_config
        self.obs_cfg = obs_config or stage_config.obs
        self.int_cfg = interceptor_config or InterceptorConfig()
        self.apn_cfg = apn_config or APNConfig(
            N_prime=stage_config.apn_N_prime, t_go_min=stage_config.apn_t_go_min)
        self.dt = float(self.int_cfg.dt)
        self.kill_radius = float(stage_config.intercept.kill_radius)
        self.world_radius = float(stage_config.intercept.world_radius)
        self.rl_action_scale = float(self.obs_cfg.rl_action_scale)

        self.privileged_fields = tuple(
            privileged_fields if privileged_fields is not None
            else DEFAULT_PRIVILEGED_FIELDS)
        for name in self.privileged_fields:
            if name not in PRIVILEGED_FIELDS:
                raise ValueError(f"unknown privileged field {name!r}")

        self._rng = np.random.default_rng(seed)
        self.target = TargetGenerator(stage_config, self._rng)
        self.kinematic = KinematicModel(
            v_max=self.int_cfg.v_max, a_max=self.int_cfg.a_max,
            omega_max=self.int_cfg.omega_max, dt=self.dt)
        self.reward_fn = RewardFunction(
            stage_config.reward, v_max=self.int_cfg.v_max,
            a_max=self.int_cfg.a_max, kill_radius=self.kill_radius)
        self.predictor = make_predictor(self.obs_cfg.predictor)
        self.obs_builder = ObsBuilder(self.obs_cfg, self.predictor)

        # spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        priv_dim = sum(PRIVILEGED_FIELDS[n] for n in self.privileged_fields)
        self.actor_dim = self.obs_builder.actor_dim
        self.privileged_dim = int(priv_dim)
        self.observation_dim = self.actor_dim + self.privileged_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_dim,),
            dtype=np.float32)

        # runtime state
        self.state: Optional[InterceptorState] = None
        self._prev_distance = 0.0
        self._prev_action = np.zeros(3, dtype=np.float64)
        self._step_count = 0
        self._max_steps = 1
        self._episode_length_s = stage_config.episode_length_s[1]
        self._obs: Optional[np.ndarray] = None
        self.trajectory_log: List[Dict[str, Any]] = []
        self.last_info: Dict[str, Any] = {}

    # ------------------------------------------------------------------ helpers
    def _oracle_path(self):
        """Target path object the oracle can sample (exact or extrapolated)."""
        if self.target.path is not None:
            return self.target.path
        return _StateAdapter(self.target.state_at)

    def _compute_heading_rate(self, a_cmd: np.ndarray) -> np.ndarray:
        """Convert a total acceleration into a body heading-rate command."""
        v = self.state.velocity
        vn = float(np.linalg.norm(v))
        if vn < _EPS:
            return np.zeros(3, dtype=np.float64)
        v_hat = v / vn
        omega = np.cross(v_hat, a_cmd) / max(vn, 1.0)
        n = float(np.linalg.norm(omega))
        if n > self.int_cfg.omega_max:
            omega = omega * (self.int_cfg.omega_max / n)
        return omega

    def _check_bounds(self) -> bool:
        p = self.state.position
        if not np.all(np.isfinite(p)):
            return True
        if p[2] < 0.0:                      # ground impact
            return True
        return bool(np.linalg.norm(p) > self.world_radius)

    def _spawn_interceptor(self) -> InterceptorState:
        sp = self.stage.spawn
        az = sp.altitude_range
        z0 = float(self._rng.uniform(az[0], az[1]))
        pos = np.array([0.0, 0.0, z0])
        return InterceptorState(
            position=pos, velocity=np.zeros(3),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3))

    def _build_observation(self, oracle: Dict[str, Any]) -> np.ndarray:
        ahead = float(self.obs_cfg.future_skip * self.obs_cfg.future_samples) * self.dt
        tp, tv, ta = self.target.state_at(self.target.elapsed + ahead)
        time_remaining = max(0.0, (self._max_steps - self._step_count) * self.dt)
        if oracle:
            opt_a = oracle.get("optimal_accel", np.zeros(3))
            opt_v = oracle.get("optimal_vel", np.zeros(3))
        else:
            opt_a = np.zeros(3)
            opt_v = np.zeros(3)
        actor, priv = self.obs_builder.build(
            interceptor_pos=self.state.position,
            interceptor_vel=self.state.velocity,
            target_pos=self.target.pos,
            target_vel=self.target.vel,
            target_acc=self.target.acc,
            target_visible=self.target.visible,
            dt=self.dt,
            time_remaining=time_remaining,
            optimal_accel=opt_a,
            optimal_vel=opt_v,
            target_true_future_pos=tp,
            target_true_future_vel=tv,
            target_true_accel=ta,
            target_state_fn=self.target.state_at,
            privileged_fields=self.privileged_fields,
        )
        return np.concatenate([actor, priv]).astype(np.float32)

    def _compute_oracle(self) -> Dict[str, Any]:
        try:
            return compute_optimal_intercept(
                self.state.position, self.state.velocity, self._oracle_path(),
                self.int_cfg.v_max, self.int_cfg.a_max, self.dt)
        except Exception:
            return {}

    # -------------------------------------------------------------------- reset
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.target.rng = self._rng

        ep_len = self.stage.sample_episode_length(self._rng)
        self._episode_length_s = float(ep_len)
        if self.stage.intercept.max_episode_steps > 0:
            self._max_steps = int(self.stage.intercept.max_episode_steps)
        else:
            self._max_steps = max(1, int(round(ep_len / self.dt)))

        self.state = self._spawn_interceptor()
        self.target.reset(self.state.position, episode_length_s=ep_len)

        # interceptor spawns already closing on the target.
        rel = self.target.pos - self.state.position
        rn = float(np.linalg.norm(rel))
        dirn = rel / rn if rn > _EPS else np.array([1.0, 0.0, 0.0])
        lo, hi = self.stage.spawn.interceptor_speed_range
        speed = float(self._rng.uniform(lo, hi))
        self.state.velocity = dirn * speed
        self.state.orientation = quat_from_vectors(
            np.array([1.0, 0.0, 0.0]), dirn)

        self._prev_distance = float(rn)
        self._prev_action = np.zeros(3, dtype=np.float64)
        self._step_count = 0
        self.trajectory_log = []
        self.predictor.reset()
        self.obs_builder.reset()

        oracle = self._compute_oracle()
        self._obs = self._build_observation(oracle)
        self.last_info = {"distance": self._prev_distance, "step_count": 0}
        return self._obs, dict(self.last_info)

    # --------------------------------------------------------------------- step
    def step(self, action):
        assert self.state is not None, "reset() must be called before step()"
        a = np.asarray(action, dtype=np.float64).reshape(3)
        a = np.clip(a, -1.0, 1.0)

        # 1. decode RL residual
        delta_a_rl = a * self.rl_action_scale

        # 2. advance target
        self.target.step(self.dt, self.state.position)

        # 3. APN command
        a_apn, apn_info = compute_apn_command(
            self.state.position, self.state.velocity,
            self.target.pos, self.target.vel, self.target.acc,
            N_prime=self.apn_cfg.N_prime, t_go_min=self.apn_cfg.t_go_min,
            t_go_max=self.apn_cfg.t_go_max)

        # 4. total + clamp
        a_total_raw = a_apn + delta_a_rl
        a_total = self.kinematic.clamp_acceleration(a_total_raw)

        # 5/6. heading rate + integrate
        omega_cmd = self._compute_heading_rate(a_total)
        self.state = self.kinematic.step(self.state, a_total, omega_cmd)

        # 7. checks
        distance = float(np.linalg.norm(self.target.pos - self.state.position))
        killed = distance < self.kill_radius
        out_of_bounds = self._check_bounds()
        self._step_count += 1
        timeout = self._step_count >= self._max_steps
        self._obs = None  # invalidated

        velocity_output = np.concatenate(
            [self.state.velocity, self.state.angular_velocity])

        # 8. reward
        zem = float(apn_info.get("ZEM", 0.0))
        los_rate_norm = float(apn_info.get("LOS_rate", 0.0))
        a_total_norm = float(np.linalg.norm(a_total)) / max(self.int_cfg.a_max, _EPS)
        a_rl_norm = float(np.linalg.norm(delta_a_rl)) / max(self.int_cfg.a_max, _EPS)
        reward, terminated, breakdown = self.reward_fn.compute(
            distance=distance, prev_distance=self._prev_distance,
            closing_vel=float(apn_info.get("closing_vel", 0.0)),
            LOS_rate_norm=los_rate_norm, a_total_norm=a_total_norm,
            a_rl_norm=a_rl_norm, action=a, prev_action=self._prev_action,
            ZEM=zem, step_count=self._step_count, max_steps=self._max_steps,
            killed=killed, out_of_bounds=out_of_bounds)

        # 9. observation
        oracle = self._compute_oracle()
        self._obs = self._build_observation(oracle)

        # 10. trajectory log
        self.trajectory_log.append({
            "step": self._step_count,
            "interceptor_pos": self.state.position.copy(),
            "interceptor_vel": self.state.velocity.copy(),
            "interceptor_quat": self.state.orientation.copy(),
            "target_pos": self.target.pos.copy(),
            "target_vel": self.target.vel.copy(),
            "a_apn": np.asarray(a_apn, dtype=np.float64).copy(),
            "a_rl": delta_a_rl.copy(),
            "a_total": a_total.copy(),
            "velocity_output": velocity_output.copy(),
            "distance": distance,
            "LOS_rate": los_rate_norm,
            "closing_vel": float(apn_info.get("closing_vel", 0.0)),
            "t_go": float(apn_info.get("t_go", 0.0)),
        })

        self._prev_distance = distance
        self._prev_action = a.copy()

        info = {
            "step_count": self._step_count,
            "distance": distance,
            "killed": bool(killed),
            "out_of_bounds": bool(out_of_bounds),
            "timeout": bool(timeout),
            "closing_vel": float(apn_info.get("closing_vel", 0.0)),
            "t_go": float(apn_info.get("t_go", 0.0)),
            "ZEM": zem,
            "LOS_rate": los_rate_norm,
            "a_apn": np.asarray(a_apn, dtype=np.float64).copy(),
            "a_rl": delta_a_rl.copy(),
            "a_total": a_total.copy(),
            "velocity_output": velocity_output.copy(),
            "reward_components": breakdown.to_dict(),
            "apn_info": apn_info,
        }
        self.last_info = info
        terminated = bool(terminated)
        truncated = bool(timeout and not terminated)
        return self._obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------- misc
    def get_trajectory_log(self) -> List[Dict[str, Any]]:
        return self.trajectory_log

    def render(self):  # pragma: no cover - rendering handled by the visualiser
        return None

    def close(self):
        return None
