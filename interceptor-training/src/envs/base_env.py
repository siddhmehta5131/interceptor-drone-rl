"""base_env.py  --  shared gymnasium environment for all 8 curriculum stages.

* Stage 1 (``target_type == "none"``) reproduces ``hover_env.AltitudeHoldEnv``
  bit-for-bit (physics, RNG draw order, reward, termination, 14-dim obs) so a
  parity test can lock the two implementations together.
* Stages 2+ add the target generator, the 19-dim target frame, history
  stacking, a predictor-driven future block, and the plan's interception
  rewards (see ``reward.py``).

Optional extensions (plan default OFF -- keeps Stage-1 parity meaningful):
``domain_randomize``, ``wind_enabled``, ``ground_effect``.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

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
from ..prediction import get_predictor
from .obs_builder import (
    FUTURE_SAMPLE_DIM,
    STAGE1_FRAME_DIM,
    TARGET_FRAME_DIM,
    ObsBuilder,
    add_target_to_frame,
    make_base_frame,
    make_future_block,
    make_privileged_block,
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

# How many past target positions the ridge predictor may look at.
_PREDICTOR_HISTORY = 32

# Per-step metrics accumulated over an episode and reported in episode_stats.
_EPISODE_METRIC_KEYS = ("alt_err", "tilt", "angvel_norm", "facing_error")

# D-16: where the ``n`` future samples come from.  ``true`` reads the
# simulator's exact polynomial path (stages 5-7); ``pred`` uses the predictor
# module.  ``true`` degrades to ``pred`` for targets without a closed-form
# path (static and the evasive stage-8 target).
_FUTURE_SOURCES = ("true", "pred")


class InterceptorBaseEnv(_BASE_CLS):
    """Multi-stage interceptor drone environment (100 Hz physics)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        stage_config: StageConfig,
        *,
        history_frames: int = 0,
        history_skip: int = 1,
        target_alt: float | Tuple[float, float] = 5.0,
        future_samples: int = 0,
        future_skip: int = 1,
        predictor: str = "const_vel",
        future_source: str = "pred",
        privileged_fields: Sequence[str] = (),
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        params: Optional[PhysicsParams] = None,
        override: Optional[dict] = None,
    ):
        if _GYM_OK:
            super().__init__()

        if future_source not in _FUTURE_SOURCES:
            raise ValueError(
                f"future_source must be one of {_FUTURE_SOURCES}, got {future_source!r}"
            )

        self.stage = stage_config
        self.cfg: RewardConfig = stage_config.reward
        self.spawn: SpawnConfig = stage_config.spawn
        self.intercept: InterceptConfig = stage_config.intercept
        self.obs_cfg: ObsStageConfig = stage_config.obs
        self._include_target = stage_config.obs.include_target

        # Episode length is a per-episode draw; this is the long-case ceiling
        # used by the time gates and the reward computer (D-13).
        self.max_episode_steps = int(stage_config.max_episode_steps)
        self._max_episode_steps = self.max_episode_steps
        self._episode_length_s = float(stage_config.episode_length_range[1])
        self.obs_noise_std = stage_config.obs.obs_noise_std

        # optional feature toggles + overrides (used by smoke tests / ablations)
        override = override or {}
        self.domain_randomize = bool(override.get("domain_randomize", stage_config.domain_randomize))
        self.wind_enabled = bool(override.get("wind_enabled", stage_config.wind_enabled))
        self.ground_effect = bool(override.get("ground_effect", stage_config.ground_effect))

        # Target altitude: a fixed value or a (lo, hi) range drawn per episode
        # (D-1).  A degenerate range draws NOTHING so Stage-1 parity holds.
        self._target_alt_range = self._coerce_alt_range(target_alt)

        self._rng = rng if rng is not None else np.random.default_rng(seed)
        self._params = params if params is not None else PhysicsParams()
        self._wind: Optional[np.ndarray] = None
        self._wind_mean = np.zeros(3)
        self._wind_theta = 1.0
        self._wind_sigma = 1.0

        # observation / action spaces
        self._history_frames = int(history_frames) if self._include_target else 0
        self._history_skip = max(1, int(history_skip)) if self._include_target else 1
        self._future_samples = int(future_samples) if self._include_target else 0
        self._future_skip = max(1, int(future_skip))
        self._privileged_fields = tuple(privileged_fields or ()) if self._include_target else ()
        self._future_source = str(future_source)
        self._predictor = get_predictor(predictor)
        self._predictor_history: deque = deque(maxlen=_PREDICTOR_HISTORY)

        self._obs_builder = ObsBuilder(
            self.obs_cfg, self._history_frames, self._history_skip, self._rng,
            future_samples=self._future_samples,
            future_skip=self._future_skip,
            privileged_fields=self._privileged_fields,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_builder.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self._reward_comp = RewardComputer(
            self.cfg, kill_radius=self.intercept.kill_radius,
            max_episode_steps=self.max_episode_steps,
        )
        self._target: Optional[TargetGenerator] = None
        self._current_target_alt = float(self._target_alt_range[0])
        self._ep_comp_sums: Dict[str, float] = {}
        self._ep_metric_sums: Dict[str, float] = {}
        self._ep_reward_total = 0.0
        self._reset_state()
        if self._include_target:
            self._target = TargetGenerator(self.stage, rng=np.random.default_rng(
                (seed if seed is not None else 0) + self._stage_seed()))

    # ------------------------------------------------------------------ setup

    @staticmethod
    def _coerce_alt_range(target_alt) -> Tuple[float, float]:
        """Accept a scalar or ``(lo, hi)`` pair and return a normalised pair."""
        if isinstance(target_alt, dict):
            lo = float(target_alt["min"])
            hi = float(target_alt["max"])
        elif isinstance(target_alt, (tuple, list)):
            lo, hi = float(target_alt[0]), float(target_alt[1])
        else:
            lo = hi = float(target_alt)
        if hi < lo:
            lo, hi = hi, lo
        return (lo, hi)

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
        self._draw_episode_length()
        self._reset_state()
        self._obs_builder.rng = self._rng
        self._obs_builder.reset()
        self._predictor.reset()
        self._predictor_history.clear()
        obs = self._get_obs()
        info = {"target_alt": float(self._current_target_alt)}
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
        facing_error = 0.0
        desired_yaw = 0.0
        if self._include_target:
            tdict = self._target.step(PH_DT, self._p_WB)
            self._distance = self._target.distance_to(self._p_WB)
            delta = self._target.pos - self._p_WB
            dn = float(np.linalg.norm(delta))
            if dn > 1e-6:
                los_world = delta / dn
                desired_yaw = float(np.arctan2(delta[1], delta[0]))
            R_WB_f = quat2rot(self._q_WB)
            current_yaw = float(np.arctan2(R_WB_f[1, 0], R_WB_f[0, 0]))
            facing_error = float(np.arctan2(
                np.sin(desired_yaw - current_yaw), np.cos(desired_yaw - current_yaw)))
            if self._target.kills_enabled and self._distance < self.intercept.kill_radius:
                killed = True

        # ---- reward / termination -------------------------------------------
        R_WB = quat2rot(self._q_WB)
        tilt = float(np.arccos(np.clip(R_WB[2, 2], -1.0, 1.0)))
        crashed_flip = tilt > self.cfg.tilt_crash_threshold
        oob = self._out_of_bounds()

        alt_err_abs = float(abs(self._p_WB[2] - self._current_target_alt))
        reward, terminated, comp, metrics = self._reward_comp.compute(
            R_WB=R_WB, v_WB=self._v_WB, omega_B=self._omega_B, p_WB=self._p_WB,
            action=action, prev_action=self._prev_action, c_cmd=c_cmd,
            alt_err=alt_err_abs, step_count=self._step_count,
            crashed_flip=crashed_flip, crashed_ground=crashed_ground,
            out_of_bounds=oob, killed=killed,
            distance=self._distance, prev_distance=prev_distance,
            target_visible=(self._target.visible if self._include_target else False),
            los_world=los_world,
            facing_error=facing_error,
        )

        # ---- bookkeeping -----------------------------------------------------
        self._prev_action = action.copy()
        self._step_count += 1
        truncated = bool(self._step_count >= self._max_episode_steps)

        self._ep_reward_total += reward
        for k, v in comp.items():
            self._ep_comp_sums[k] = self._ep_comp_sums.get(k, 0.0) + v
        for k in _EPISODE_METRIC_KEYS:
            self._ep_metric_sums[k] = self._ep_metric_sums.get(k, 0.0) + metrics[k]

        obs = self._get_obs()
        info = self._build_info(comp, metrics, killed, c_cmd, terminated or truncated)
        if self._include_target:
            info["desired_yaw"] = float(desired_yaw)
            info["facing_error"] = float(facing_error)
        if _GYM_OK:
            return obs, float(reward), bool(terminated), truncated, info
        return obs, float(reward), bool(terminated or truncated), info

    def render(self):
        return None

    def close(self):
        pass

    # ------------------------------------------------------------- internals

    def _draw_episode_length(self) -> None:
        """Draw this episode's length in seconds (D-13).

        A fixed (degenerate) length consumes NO rng draw, which keeps Stage 1
        bit-identical to ``hover_env``.  The sample is drawn BEFORE
        ``_reset_state`` so the parity draws inside it stay contiguous.
        """
        lo, hi = self.stage.episode_length_range
        secs = float(self._rng.uniform(lo, hi)) if hi > lo else lo
        self._episode_length_s = secs
        self._max_episode_steps = max(1, int(round(secs / PH_DT)))

    def _reset_state(self):
        """(Re)initialise all simulation state for a fresh episode."""
        # per-episode bounds
        self._z_lo = 0.0
        if self._include_target:
            self._z_hi = TARGET_STAGE_Z_HI
            self._xy_limit = float(self.stage.world_radius)
        else:
            self._z_margin = 10.0
            self._z_hi = self._target_alt_range[1] + self._z_margin
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

        # D-1: per-episode target altitude.  Placed AFTER the six parity draws
        # above, and skipped entirely when the range is degenerate.
        alt_lo, alt_hi = self._target_alt_range
        self._current_target_alt = (
            float(self._rng.uniform(alt_lo, alt_hi)) if alt_hi > alt_lo else alt_lo
        )
        spawn_z = float(self._current_target_alt)

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
            self._target.reset(self._p_WB, episode_length_s=self._episode_length_s)
            self._predictor_history.clear()
            self._predictor_history.append(self._target.pos.copy())
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
            alt_err = float(np.clip(
                self._p_WB[2] - self._current_target_alt, -_MAX_Z_ERROR, _MAX_Z_ERROR))
            frame = make_base_frame(
                R_WB, self._v_WB, self._omega_B, alt_err, include_target=False)
            return self._obs_builder.update(frame)

        # target stages: 19-dim frame with body-frame target fields (E-1: the
        # observed position is the TRUE current position -- no lookahead).
        base = make_base_frame(R_WB, self._v_WB, self._omega_B, 0.0, include_target=True)
        rel_world = self._target.pos - self._p_WB
        rel_vel_world = self._target.vel - self._v_WB
        rel_pos_body = R_WB.T @ rel_world
        rel_vel_body = R_WB.T @ rel_vel_world
        visible = bool(self._target.visible)
        frame = add_target_to_frame(base, rel_pos_body, rel_vel_body, visible)
        self._predictor_history.append(self._target.pos.copy())

        future = self._future_block(R_WB, visible)
        privileged = self._privileged_block(
            R_WB, rel_pos_body, rel_vel_body, visible)
        return self._obs_builder.update(frame, future=future, privileged=privileged)

    def _future_block(self, R_WB: np.ndarray, visible: bool) -> Optional[np.ndarray]:
        """Future block in the drone's CURRENT body frame (D-11, D-16, D-54).

        ``future_source='true'`` reads the simulator's exact polynomial path
        (stages 5-7).  Every other case -- and any target without a
        closed-form path -- uses the predictor module.
        """
        if not self._future_samples:
            return None

        pp, pv = self._future_samples_world()

        # drop the predictor's "now" row: the observation carries exactly the
        # n strictly-future samples (plan E-3/E-4).
        pp = pp[1:]
        pv = pv[1:]

        rows = pp.shape[0]
        rel_p = np.empty((rows, 3), dtype=np.float64)
        rel_v = np.empty((rows, 3), dtype=np.float64)
        for i in range(rows):
            rel_p[i] = R_WB.T @ (pp[i] - self._p_WB)
            rel_v[i] = R_WB.T @ (pv[i] - self._v_WB)
        return make_future_block(rel_p, rel_v, visible)

    def _future_samples_world(self) -> Tuple[np.ndarray, np.ndarray]:
        """WORLD-frame ``(n+1, 3)`` position/velocity samples, row 0 == now."""
        path = getattr(self._target, "path", None) if self._include_target else None
        if self._future_source == "true" and path is not None:
            now = float(self._target.elapsed)
            times = now + self._future_skip * PH_DT * np.arange(
                1, self._future_samples + 1, dtype=np.float64
            )
            pp = np.empty((self._future_samples + 1, 3), dtype=np.float64)
            pv = np.empty_like(pp)
            pp[0] = self._target.pos
            pv[0] = self._target.vel
            for i, t in enumerate(times, start=1):
                pp[i] = path.position_at(float(t))
                pv[i] = path.velocity_at(float(t))
            return pp, pv

        history = (
            np.asarray(self._predictor_history, dtype=np.float64)
            if len(self._predictor_history) > 1 else None
        )
        return self._predictor.predict(
            self._target.pos, self._target.vel,
            history=history,
            future_samples=self._future_samples,
            future_skip=self._future_skip,
            dt=PH_DT,
        )

    def _privileged_block(
        self,
        R_WB: np.ndarray,
        rel_pos_body: np.ndarray,
        rel_vel_body: np.ndarray,
        visible: bool,
    ) -> Optional[np.ndarray]:
        """Exact-truth privileged tail for the critic (D-12).

        ``target_true_pos`` / ``target_true_vel`` are the IDEAL minus CURRENT
        body-frame errors: they tell the critic how far the *predictor* is
        from the truth without re-observing the target.
        """
        if not self._privileged_fields:
            return None
        if not visible:
            zero3 = np.zeros(3, dtype=np.float64)
            ideal_pos, ideal_vel = zero3, zero3
        else:
            pp, pv = self._future_samples_world()
            ideal_pos = R_WB.T @ (pp[0] - self._p_WB)
            ideal_vel = R_WB.T @ (pv[0] - self._v_WB)
        return make_privileged_block(
            self._privileged_fields,
            time_remaining=(
                (self._max_episode_steps - self._step_count) / float(self._max_episode_steps)
            ),
            facing_error=float(self._facing_error()),
            target_true_pos_body=ideal_pos - rel_pos_body,
            target_true_vel_body=ideal_vel - rel_vel_body,
        )

    def _facing_error(self) -> float:
        """Signed body-yaw error (rad, wrapped to [-pi, pi]) toward the target.

        D-4: yaw the nose toward the target, not merely toward its position.
        Returns 0.0 when there is no target.
        """
        if not self._include_target or self._target is None:
            return 0.0
        los = self._target.pos - self._p_WB
        if float(np.linalg.norm(los[:2])) < 1e-6:
            return 0.0
        desired = float(np.arctan2(los[1], los[0]))
        R_WB = quat2rot(self._q_WB)
        current = float(np.arctan2(R_WB[1, 0], R_WB[0, 0]))
        return float(np.arctan2(np.sin(desired - current), np.cos(desired - current)))

    def _build_info(self, comp: Dict[str, float], metrics: Dict[str, float],
                    killed: bool, c_cmd: float, done: bool) -> Dict:
        info: Dict = {}
        if not self._include_target:
            info["target_alt"] = float(self._current_target_alt)
            info["thrust_dev"] = float(abs(c_cmd - PH_C_HOVER))
        for k, v in comp.items():
            info[k] = float(v)
        info.update({
            "alt_err": float(metrics["alt_err"]),
            "tilt": float(metrics["tilt"]),
            "angvel_norm": float(metrics["angvel_norm"]),
            "miss_distance": float(metrics["miss_distance"]),
            "facing_error": float(metrics["facing_error"]),
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
                "episode_length_s": float(self._episode_length_s),
            }
            for k in _EPISODE_METRIC_KEYS:
                stats[f"mean_{k}"] = float(self._ep_metric_sums.get(k, 0.0) / steps)
            for k, v in self._ep_comp_sums.items():
                stats[f"mean_{k}"] = float(v / steps)
            info["episode_stats"] = stats
        return info


def make_env_factory(
    stage,
    history_frames: int = 0,
    history_skip: int = 1,
    target_alt: float | Tuple[float, float] = 5.0,
    future_samples: int = 0,
    future_skip: int = 1,
    predictor: str = "const_vel",
    future_source: str = "pred",
    privileged_fields: Sequence[str] = (),
    seed: Optional[int] = None,
    monitor_dir: Optional[str] = None,
    override: Optional[dict] = None,
):
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
            target_alt=target_alt,
            future_samples=future_samples, future_skip=future_skip,
            predictor=predictor,
            future_source=future_source,
            privileged_fields=tuple(privileged_fields or ()),
            seed=seed, override=override,
        )
        if monitor_dir is not None:
            from stable_baselines3.common.monitor import Monitor
            import os
            os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(
                env,
                filename=os.path.join(monitor_dir, f"env_{seed}"),
                override_existing=False,
            )
        return env

    return _factory
