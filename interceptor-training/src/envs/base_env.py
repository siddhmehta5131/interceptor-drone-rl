"""base_env.py  --  shared gymnasium environment for all 8 curriculum stages.

* Stage 1 (``target_type == "none"``) reproduces ``hover_env.AltitudeHoldEnv``
  bit-for-bit (physics, RNG draw order, reward, termination, 14-dim obs) so a
  parity test can lock the two implementations together.
* Stages 2+ add the target generator, the 19-dim target frame, history
  stacking, and the plan's interception rewards (see ``reward.py``).

Optional extensions (plan default OFF -- keeps Stage-1 parity meaningful):
``domain_randomize``, ``wind_enabled``, ``ground_effect``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..physics import aero as _aero
from ..physics.constants import (
    PH_C_HOVER,
    PH_C_SCALE,
    PH_CUT_THR,
    PH_DT,
    PH_OMEGA_HOVER,
    PH_W_SCALE,
)
from ..physics.pipeline import (
    PhysicsParams,
    command_from_action,
    esc,
    mixer,
    pid,
    rk4,
    sample_physics_params,
)
from ..physics.quaternion import quat2rot
from .obs_builder import (
    STAGE1_FRAME_DIM,
    TARGET_FRAME_DIM,
    ObsBuilder,
    add_target_to_frame,
    make_base_frame,
)
from .reward import RewardComputer
from .stage_config import (
    STAGES,
    TARGET_STAGE_Z_HI,
    InterceptConfig,
    ObsStageConfig,
    RewardConfig,
    SpawnConfig,
    StageConfig,
)
from .target_generator import TargetGenerator

__all__ = ["InterceptorBaseEnv", "make_env_factory"]

# ----------------------------------------------------------------------------
# Optional-gymnasium stub pattern (mirrors hover_env.py)
# ----------------------------------------------------------------------------

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_OK = True
except ImportError:  # pragma: no cover - exercised when gymnasium absent
    _GYM_OK = False

    class _SpacesStub:
        class Box:
            def __init__(self, low, high, shape, dtype):
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype

    spaces = _SpacesStub()

if _GYM_OK:
    _BASE_CLS = gym.Env
else:  # pragma: no cover
    class _BaseStub:
        metadata = {}

        def reset(self, *a, **kw):
            raise NotImplementedError

        def step(self, action):
            raise NotImplementedError

    _BASE_CLS = _BaseStub

# Stage-1 default altitude error clip (hover_env max_z_error)
_MAX_Z_ERROR = 20.0


class InterceptorBaseEnv(_BASE_CLS):
    """Multi-stage interceptor drone environment (100 Hz physics)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        stage_config: StageConfig,
        *,
        history_frames: int = 0,
        history_skip: int = 1,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        params: Optional[PhysicsParams] = None,
        override: Optional[dict] = None,
    ):
        if _GYM_OK:
            super().__init__()

        self.stage = stage_config
        self.cfg: RewardConfig = stage_config.reward
        self.spawn: SpawnConfig = stage_config.spawn
        self.intercept: InterceptConfig = stage_config.intercept
        self.obs_cfg: ObsStageConfig = stage_config.obs
        self._include_target = stage_config.obs.include_target

        self.max_episode_steps = int(stage_config.max_episode_steps)
        self.obs_noise_std = stage_config.obs.obs_noise_std

        # optional feature toggles + overrides (used by smoke tests / ablations)
        override = override or {}
        self.domain_randomize = bool(override.get("domain_randomize", stage_config.domain_randomize))
        self.wind_enabled = bool(override.get("wind_enabled", stage_config.wind_enabled))
        self.ground_effect = bool(override.get("ground_effect", stage_config.ground_effect))
        lookahead = override.get("lookahead_enabled", self.intercept.lookahead_enabled)
        self._lookahead = bool(lookahead)

        self._rng = rng if rng is not None else np.random.default_rng(seed)
        self._params = params if params is not None else PhysicsParams()
        self._wind: Optional[np.ndarray] = None
        self._wind_mean = np.zeros(3)
        self._wind_theta = 1.0
        self._wind_sigma = 1.0

        # observation / action spaces
        self._history_frames = int(history_frames) if self._include_target else 0
        self._history_skip = max(1, int(history_skip)) if self._include_target else 1
        obs_dim = (
            STAGE1_FRAME_DIM
            if not self._include_target
            else TARGET_FRAME_DIM * (self._history_frames + 1)
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self._obs_builder = ObsBuilder(
            self.obs_cfg, self._history_frames, self._history_skip, self._rng
        )
        self._reward_comp = RewardComputer(
            self.cfg, kill_radius=self.intercept.kill_radius,
            max_episode_steps=self.max_episode_steps,
        )
        self._target: Optional[TargetGenerator] = None
        self._ep_comp_sums: Dict[str, float] = {}
        self._ep_metric_sums: Dict[str, float] = {}
        self._ep_reward_total = 0.0
        self._reset_state()
        if self._include_target:
            self._target = TargetGenerator(self.stage, rng=np.random.default_rng(
                (seed if seed is not None else 0) + self._stage_seed()))

    def _stage_seed(self) -> int:
        """Stable per-stage seed offset (no str hash -- process randomized)."""
        try:
            idx = int(self.stage.id.split("_")[1])
        except (IndexError, ValueError):
            idx = 1
        return 1_999_937 * idx

    # ------------------------------------------------------------------ API

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            # reseed target generator deterministically as well
            if self._include_target:
                self._target = TargetGenerator(self.stage, rng=np.random.default_rng(
                    seed + self._stage_seed()))
        self._reset_state()
        self._obs_builder.rng = self._rng
        self._obs_builder.reset()
        obs = self._get_obs()
        info = {"target_alt": float(self._target_alt)}
        if _GYM_OK:
            return obs, info
        return obs

    def step(self, action):
        """Advance one 100 Hz step.  ``[thrust, roll, pitch, yaw]`` in [-1,1]."""
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        c_cmd, crashed_ground = self._physics_step(action)

        # ---- target advance + geometry -------------------------------------
        prev_distance = self._distance
        killed = False
        los_world = None
        if self._include_target:
            tdict = self._target.step(PH_DT, self._p_WB)
            self._distance = self._target.distance_to(self._p_WB)
            delta = self._target.pos - self._p_WB
            dn = float(np.linalg.norm(delta))
            if dn > 1e-6:
                los_world = delta / dn
            if self._target.kills_enabled and self._distance < self.intercept.kill_radius:
                killed = True

        # ---- reward / termination -------------------------------------------
        R_WB = quat2rot(self._q_WB)
        tilt = float(np.arccos(np.clip(R_WB[2, 2], -1.0, 1.0)))
        crashed_flip = tilt > self.cfg.tilt_crash_threshold
        oob = self._out_of_bounds()

        alt_err_abs = float(abs(self._p_WB[2] - self._target_alt))
        reward, terminated, comp, metrics = self._reward_comp.compute(
            R_WB=R_WB, v_WB=self._v_WB, omega_B=self._omega_B, p_WB=self._p_WB,
            action=action, prev_action=self._prev_action, c_cmd=c_cmd,
            alt_err=alt_err_abs, step_count=self._step_count,
            crashed_flip=crashed_flip, crashed_ground=crashed_ground,
            out_of_bounds=oob, killed=killed,
            distance=self._distance, prev_distance=prev_distance,
            target_visible=(self._target.visible if self._include_target else False),
            los_world=los_world,
        )

        # ---- bookkeeping -----------------------------------------------------
        self._prev_action = action.copy()
        self._step_count += 1
        truncated = bool(self._step_count >= self.max_episode_steps)

        self._ep_reward_total += reward
        for k, v in comp.items():
            self._ep_comp_sums[k] = self._ep_comp_sums.get(k, 0.0) + v
        for k in ("alt_err", "tilt", "angvel_norm"):
            self._ep_metric_sums[k] = self._ep_metric_sums.get(k, 0.0) + metrics[k]

        obs = self._get_obs()
        info = self._build_info(comp, metrics, killed, c_cmd, terminated or truncated)
        if _GYM_OK:
            return obs, float(reward), bool(terminated), truncated, info
        return obs, float(reward), bool(terminated or truncated), info

    def render(self):
        return None

    def close(self):
        pass

    # ------------------------------------------------------------- internals

    def _reset_state(self):
        """(Re)initialise all simulation state for a fresh episode."""
        # per-episode bounds
        self._z_lo = 0.0
        if self._include_target:
            self._z_hi = TARGET_STAGE_Z_HI
            self._xy_limit = float(self.stage.world_radius)
        else:
            self._z_margin = 10.0
            self._z_hi = self._target_alt + self._z_margin
            self._xy_limit = float(self.stage.world_radius)

        # domain randomization / wind / ground effect (optional, OFF default)
        if self.domain_randomize:
            self._params = sample_physics_params(self._rng)
        else:
            self._params = PhysicsParams()
        if self.wind_enabled:
            self._wind_mean, self._wind_theta, self._wind_sigma = _aero.sample_wind_params(self._rng)
            self._wind = self._wind_mean.copy()
        else:
            self._wind = None

        # spawn (identical draw order to hover_env._reset_state)
        xy = self._rng.uniform(-0.3, 0.3, size=2)
        tilt_rad = self._rng.uniform(0.0, np.deg2rad(5.0))
        axis = self._rng.standard_normal(3)
        axis /= np.linalg.norm(axis) + 1e-12
        q_init = np.r_[np.cos(tilt_rad / 2.0), np.sin(tilt_rad / 2.0) * axis]
        q_init /= np.linalg.norm(q_init)
        spawn_z = float(self._target_alt)

        self._p_WB = np.array([xy[0], xy[1], spawn_z], dtype=np.float64)
        self._q_WB = q_init.astype(np.float64)
        self._v_WB = np.zeros(3, dtype=np.float64)
        self._omega_B = np.zeros(3, dtype=np.float64)
        self._omega_B_2 = np.zeros(3, dtype=np.float64)
        self._Omega = np.full(4, PH_OMEGA_HOVER, dtype=np.float64)
        self._I_pid = np.zeros(3, dtype=np.float64)
        self._U_bat = 4.2
        self._prev_action = np.zeros(4, dtype=np.float64)
        self._step_count = 0

        # target state + geometry bookkeeping
        self._ep_reward_total = 0.0
        self._ep_comp_sums = {}
        self._ep_metric_sums = {}
        if self._include_target:
            if self._target is None:
                self._target = TargetGenerator(self.stage, rng=np.random.default_rng(0))
            self._target.reset(self._p_WB)
            self._distance = self._target.distance_to(self._p_WB)
        else:
            self._distance = float("inf")

    def _physics_step(self, action: np.ndarray):
        """One RK4 step (verbatim hover_env order + optional wind/GE/DR)."""
        c_cmd, omega_cmd, throttle_cut = command_from_action(action)

        u_torque, self._I_pid = pid(
            omega_cmd, self._omega_B, self._omega_B_2, self._I_pid, throttle_cut, PH_DT)
        cmd = mixer(u_torque, c_cmd)
        if throttle_cut:
            cmd = np.zeros(4)

        ge_gain = _aero.ground_effect_gain(self._p_WB[2]) if self.ground_effect else 1.0

        p_n, q_n, v_n, om_n, Om_n = rk4(
            self._p_WB, self._q_WB, self._v_WB, self._omega_B, self._Omega,
            cmd, self._U_bat, self._params,
            wind_W=self._wind, ge_gain=ge_gain,
        )

        if not throttle_cut:
            _, self._U_bat = esc(cmd, self._U_bat)

        self._omega_B_2 = self._omega_B.copy()
        self._p_WB = p_n
        self._q_WB = q_n
        self._v_WB = v_n
        self._omega_B = om_n
        self._Omega = Om_n

        # advance wind (OU) after the step -- extension only
        if self._wind is not None:
            self._wind = _aero.wind_ou_step(
                self._wind, self._wind_mean, self._wind_theta, self._wind_sigma, PH_DT, self._rng)

        crashed_ground = False
        if self._p_WB[2] <= 0.0:
            if self._v_WB[2] < -self.cfg.ground_contact_v_threshold:
                crashed_ground = True
            self._p_WB[2] = 0.0
            self._v_WB[2] = max(float(self._v_WB[2]), 0.0)

        return c_cmd, crashed_ground

    def _out_of_bounds(self) -> bool:
        return bool(
            abs(self._p_WB[0]) > self._xy_limit
            or abs(self._p_WB[1]) > self._xy_limit
            or self._p_WB[2] < self._z_lo
            or self._p_WB[2] > self._z_hi
        )

    def _get_obs(self) -> np.ndarray:
        R_WB = quat2rot(self._q_WB)
        if not self._include_target:
            alt_err = float(np.clip(self._p_WB[2] - self._target_alt, -_MAX_Z_ERROR, _MAX_Z_ERROR))
            frame = make_base_frame(
                R_WB, self._v_WB, self._omega_B, alt_err, include_target=False)
            return self._obs_builder.update(frame)

        # target stages: 19-dim frame with body-frame target fields
        base = make_base_frame(R_WB, self._v_WB, self._omega_B, 0.0, include_target=True)
        target_pos = self._target.predicted_pos() if self._lookahead else self._target.pos
        rel_world = target_pos - self._p_WB
        rel_vel_world = self._target.vel - self._v_WB
        rel_pos_body = R_WB.T @ rel_world
        rel_vel_body = R_WB.T @ rel_vel_world
        frame = add_target_to_frame(base, rel_pos_body, rel_vel_body, self._target.visible)
        return self._obs_builder.update(frame)

    def _build_info(self, comp: Dict[str, float], metrics: Dict[str, float],
                    killed: bool, c_cmd: float, done: bool) -> Dict:
        info: Dict = {}
        if not self._include_target:
            info["target_alt"] = float(self._target_alt)
            info["thrust_dev"] = float(abs(c_cmd - PH_C_HOVER))
        for k, v in comp.items():
            info[k] = float(v)
        info.update({
            "alt_err": float(metrics["alt_err"]),
            "tilt": float(metrics["tilt"]),
            "angvel_norm": float(metrics["angvel_norm"]),
            "miss_distance": float(metrics["miss_distance"]),
            "time_to_intercept": float(metrics["time_to_intercept"]),
            "killed": bool(killed),
            "crashed": bool(metrics["crashed"]),
            "out_of_bounds": bool(metrics["out_of_bounds"]),
        })
        if self._include_target:
            info["distance"] = float(self._distance)
            info["target_visible"] = bool(self._target.visible)

        if done:
            steps = max(1, self._step_count)
            stats: Dict = {
                "episode_reward": float(self._ep_reward_total),
                "reward_total": float(self._ep_reward_total),
                "reward_mean": float(self._ep_reward_total / steps),
                "final_distance": float(self._distance),
                "kill": 1.0 if killed else 0.0,
                "steps": float(self._step_count),
                "time_to_intercept": float(self._step_count),
                "miss_distance": float(metrics["miss_distance"]),
            }
            for k in ("alt_err", "tilt", "angvel_norm"):
                stats[f"mean_{k}"] = float(self._ep_metric_sums.get(k, 0.0) / steps)
            for k, v in self._ep_comp_sums.items():
                stats[f"mean_{k}"] = float(v / steps)
            info["episode_stats"] = stats
        return info

    # convenience for stage-1 parity of hover attribute names
    @property
    def _target_alt(self) -> float:
        return 5.0


def make_env_factory(stage, history_frames: int = 0, history_skip: int = 1,
                     seed: Optional[int] = None, monitor_dir: Optional[str] = None,
                     override: Optional[dict] = None):
    """Return a zero-arg callable creating one env (for VecEnv factories).

    ``stage`` is the EFFECTIVE :class:`StageConfig` (after YAML overrides are
    merged by the config loader) so per-stage overrides reach the env; a bare
    ``stage_id`` string is accepted for backward compatibility.
    When ``monitor_dir`` is given the env is wrapped in ``Monitor`` (the SB3
    training entrypoint uses this to record per-episode rewards).
    """
    from .stage_config import STAGES

    if isinstance(stage, str):
        stage = STAGES[stage]

    def _factory():
        import sys
        sys.path.insert(0, ".")
        env = InterceptorBaseEnv(
            stage, history_frames=history_frames, history_skip=history_skip,
            seed=seed, override=override,
        )
        if monitor_dir is not None:
            from stable_baselines3.common.monitor import Monitor
            import os
            env = Monitor(env, filename=os.path.join(monitor_dir, f"env_{seed}"))
        return env

    return _factory