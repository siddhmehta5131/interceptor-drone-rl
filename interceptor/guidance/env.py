"""Unified 3-D interceptor guidance environment (plan P1.8 / P5).

One environment serves all four arms through ``mode``:

* ``apn_raw``      -- A1: textbook perpendicular APN on raw seeker samples
* ``apn_filtered`` -- A2: filtered seeker, APN + separate axial term
* ``apn_rl``       -- A3: A2 + bounded RL residual
* ``pure_rl``      -- A4: RL alone, same filtered observation

and both target sets through ``target_set``: ``static``, ``polynomial`` (orders
1-3) or ``evasive`` (the 15 notebook laws).

World frame: x north, y west, z up.  A3 with a zero action is bit-identical to
A2.  Kill detection uses the continuous closest point of approach within each
step, against the true target.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAVE_GYM = True
except Exception:  # pragma: no cover
    gym, spaces, _HAVE_GYM = None, None, False

from interceptor.envs.target_generator import PolynomialTargetPath
from .kinematics import (
    apply_interceptor_constraints, closest_approach_distance,
    compute_3d_apn_command, compute_engagement_kinematics,
    compute_optimal_intercept_oracle, heading_rate_from_acceleration,
)
from .maneuvers import ManeuverType, PlatformType
from .observation import AsymmetricObsBuilder
from .platform import InterceptorLimits, PLATFORMS
from .seeker import SeekerSensorFilter, SensorNoiseConfig
from .targets import HARD_MANEUVERS, GenericEvasiveTarget, PolynomialTarget, StaticTarget

__all__ = ["UnifiedInterceptEnv", "MODES", "TARGET_SETS"]

MODES = ("apn_raw", "apn_filtered", "apn_rl", "pure_rl")
TARGET_SETS = ("static", "polynomial", "evasive")
_LEGACY_MODE = {(True, True): "apn_raw", (True, False): "apn_rl", (False, False): "pure_rl"}

_Base = gym.Env if _HAVE_GYM else object


class UnifiedInterceptEnv(_Base):
    """Gymnasium environment for APN / RL / hybrid intercept guidance."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        platform: PlatformType = PlatformType.QUADCOPTER,
        mode: str = "apn_rl",
        target_set: str = "evasive",
        *,
        limits: Optional[InterceptorLimits] = None,
        noise_config: Optional[SensorNoiseConfig] = None,
        noise_scale: float = 1.0,
        fixed_maneuver: Optional[ManeuverType] = None,
        polynomial_order: int = 1,
        N_nav: float = 4.0,
        seed: Optional[int] = None,
        use_apn: Optional[bool] = None,
        pure_apn_mode: Optional[bool] = None,
    ):
        if not _HAVE_GYM:
            raise ImportError("gymnasium is required to use UnifiedInterceptEnv")
        if use_apn is not None or pure_apn_mode is not None:
            mode = _LEGACY_MODE[(bool(use_apn), bool(pure_apn_mode))]
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if target_set not in TARGET_SETS:
            raise ValueError(f"target_set must be one of {TARGET_SETS}, got {target_set!r}")

        self.platform = platform
        self.mode = mode
        self.target_set = target_set
        self.limits = limits if limits is not None else PLATFORMS[platform]
        self.noise_cfg = noise_config or SensorNoiseConfig()
        self.noise_scale = float(noise_scale)
        self.fixed_maneuver = fixed_maneuver
        self.polynomial_order = int(polynomial_order)
        self.N_nav = float(N_nav)
        self.dt = float(self.limits.dt)
        self.max_steps = int(self.limits.max_time / self.dt)
        self.kill_radius = float(self.limits.kill_radius)

        self.rng = np.random.default_rng(seed)
        self.sensor = SeekerSensorFilter(self.noise_cfg, self.limits, self.rng)
        self.obs_builder = AsymmetricObsBuilder(self.limits)

        self._all_maneuvers = list(ManeuverType)
        weights = np.array([2.5 if m in HARD_MANEUVERS else 1.0
                            for m in self._all_maneuvers])
        self._maneuver_probs = weights / weights.sum()

        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -10.0, 10.0, shape=(self.obs_builder.total_dim,), dtype=np.float32)

        self.p_M = np.zeros(3)
        self.v_M = np.zeros(3)
        self.a_M = np.zeros(3)
        self.omega_M = np.zeros(3)
        self.target = None
        self.step_count = 0
        self.prev_dist = 0.0
        self.prev_action = np.zeros(3)
        self.min_miss_distance = float("inf")
        self.traj_M = []
        self.traj_T = []
        self.traj_T_noisy = []
        self.last_info: Dict[str, Any] = {}

    # ------------------------------------------------------------------ helpers
    @property
    def uses_apn(self) -> bool:
        return self.mode != "pure_rl"

    def set_noise_scale(self, scale: float) -> None:
        self.noise_scale = float(scale)

    def _spawn(self):
        lim = self.limits
        self.p_M = np.array([0.0, 0.0, lim.alt_init], dtype=float)
        dist = float(self.rng.uniform(lim.r_init_min, lim.r_init_max))
        az = float(self.rng.uniform(-np.deg2rad(35.0), np.deg2rad(35.0)))
        el = float(self.rng.uniform(-np.deg2rad(15.0), np.deg2rad(15.0)))
        direction = np.array([np.cos(el) * np.cos(az),
                              -np.cos(el) * np.sin(az),
                              np.sin(el)])
        p_T0 = self.p_M + dist * direction
        p_T0[2] = max(p_T0[2], 50.0)

        heading = float(self.rng.uniform(-np.pi, np.pi))
        pitch = float(self.rng.uniform(-np.deg2rad(10.0), np.deg2rad(10.0)))
        v_mag = float(self.rng.uniform(0.75 * lim.v_target_nominal,
                                       1.15 * lim.v_target_nominal))
        v_T0 = v_mag * np.array([np.cos(pitch) * np.cos(heading),
                                 -np.cos(pitch) * np.sin(heading),
                                 np.sin(pitch)])
        init_speed = float(self.rng.uniform(max(lim.v_min, 0.65 * lim.v_max),
                                            0.85 * lim.v_max))
        self.v_M = init_speed * direction
        self.a_M = np.zeros(3)
        self.omega_M = np.zeros(3)

        if self.target_set == "evasive":
            if self.fixed_maneuver is not None:
                maneuver = self.fixed_maneuver
            else:
                idx = int(self.rng.choice(len(self._all_maneuvers),
                                          p=self._maneuver_probs))
                maneuver = self._all_maneuvers[idx]
            self.target = GenericEvasiveTarget(
                p_T0, v_T0, maneuver, lim.v_target_nominal,
                lim.a_target_max, self.rng)
        elif self.target_set == "static":
            self.target = StaticTarget(p_T0)
        else:
            end = p_T0 + self.rng.standard_normal(3) * 0.5 * dist
            end[2] = float(max(end[2], 50.0))
            path = PolynomialTargetPath(
                start=p_T0, end=end, duration=lim.max_time,
                order=self.polynomial_order, rng=self.rng,
                speed_cap=1.15 * lim.v_target_nominal, shape="random")
            self.target = PolynomialTarget(path)

        return p_T0

    # -------------------------------------------------------------------- reset
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.sensor.rng = self.rng
        lim = self.limits
        self.step_count = 0
        self.prev_action = np.zeros(3)
        self._spawn()
        self.sensor.reset(self.target.p_W, self.target.v_W, self.target.a_W)
        self.obs_builder.reset()

        est_p, est_v, est_a, visible = self.sensor.observe(
            self.p_M, self.target.p_W, self.target.v_W, self.target.a_W,
            self.dt, self.noise_scale)
        kin = compute_engagement_kinematics(
            self.target.p_W, self.target.v_W, self.target.a_W, self.p_M, self.v_M)
        opt_a, opt_v = compute_optimal_intercept_oracle(
            self.p_M, self.v_M, self.target.p_W, self.target.v_W, self.target.a_W,
            lim.v_max, lim.a_n_max, kin.t_go)
        tf_p, tf_v, tf_a = self.target.state_at_future(15.0 * self.dt)

        self.prev_dist = kin.r
        self.min_miss_distance = kin.r
        self.traj_M = [self.p_M.copy()]
        self.traj_T = [self.target.p_W.copy()]
        self.traj_T_noisy = [self.sensor.raw_pos.copy()]
        obs = self.obs_builder.build(
            p_M=self.p_M, v_M=self.v_M, est_p_T=est_p, est_v_T=est_v,
            est_a_T=est_a, visible=visible, dt=self.dt,
            time_remaining=lim.max_time, optimal_accel=opt_a, optimal_vel=opt_v,
            true_future_pos=tf_p, true_future_vel=tf_v, true_acc=tf_a)
        maneuver = getattr(self.target, "maneuver", ManeuverType.NONE)
        self.last_info = {"maneuver": maneuver.value, "distance": kin.r,
                          "platform": self.platform.value, "mode": self.mode}
        return obs, dict(self.last_info)

    # --------------------------------------------------------------------- step
    def step(self, action):
        assert self.target is not None, "reset() must be called before step()"
        lim = self.limits
        act = np.clip(np.asarray(action, dtype=float).reshape(3), -1.0, 1.0)
        t = self.step_count * self.dt

        self.target.step(self.dt, t, self.p_M, self.v_M)
        est_p, est_v, est_a, visible = self.sensor.observe(
            self.p_M, self.target.p_W, self.target.v_W, self.target.a_W,
            self.dt, self.noise_scale)

        if self.uses_apn:
            if self.mode == "apn_raw":
                kin_cmd = compute_engagement_kinematics(
                    self.sensor.raw_pos, self.sensor.raw_vel, self.sensor.raw_acc,
                    self.p_M, self.v_M, r_min_clamp=2.0)
                a_apn = compute_3d_apn_command(
                    kin_cmd, self.sensor.raw_acc, self.v_M, lim,
                    self.N_nav, include_axial_boost=False)
                a_rl = np.zeros(3)
            else:
                kin_cmd = compute_engagement_kinematics(
                    est_p, est_v, est_a, self.p_M, self.v_M, r_min_clamp=15.0)
                a_apn = compute_3d_apn_command(
                    kin_cmd, est_a, self.v_M, lim, self.N_nav,
                    include_axial_boost=True)
                a_rl = act * (0.35 * lim.a_n_max) if self.mode == "apn_rl" else np.zeros(3)
            a_raw = a_apn + a_rl
        else:
            a_apn = np.zeros(3)
            speed = float(np.linalg.norm(self.v_M))
            if not lim.omnidirectional:
                t_hat = self.v_M / max(speed, 1e-6)
                a_rl = act * lim.a_n_max + 0.5 * lim.a_t_max * t_hat
            else:
                a_rl = act * lim.a_n_max
            a_raw = a_rl

        a_cmd = apply_interceptor_constraints(a_raw, self.v_M, lim)
        da_dt = (a_cmd - self.a_M) / max(lim.tau_accel, 1e-3)
        jerk = float(np.linalg.norm(da_dt))
        if jerk > lim.max_jerk:
            da_dt *= lim.max_jerk / jerk

        p_M_prev = self.p_M.copy()
        p_T_prev = self.target.p_W - self.target.v_W * self.dt
        self.a_M = self.a_M + da_dt * self.dt
        self.v_M = self.v_M + self.a_M * self.dt
        speed = float(np.linalg.norm(self.v_M))
        if not lim.can_hover and speed < lim.v_min:
            self.v_M = self.v_M / max(speed, 1e-6) * lim.v_min
        elif speed > lim.v_max:
            self.v_M = self.v_M / speed * lim.v_max
        self.p_M = self.p_M + self.v_M * self.dt
        self.omega_M = heading_rate_from_acceleration(self.v_M, self.a_M, lim.omega_max)
        self.step_count += 1
        velocity_output = np.concatenate([self.v_M, self.omega_M])

        self.traj_M.append(self.p_M.copy())
        self.traj_T.append(self.target.p_W.copy())
        self.traj_T_noisy.append(self.sensor.raw_pos.copy())

        substep_dist = closest_approach_distance(
            p_T_prev, self.target.p_W, p_M_prev, self.p_M)
        kin_true = compute_engagement_kinematics(
            self.target.p_W, self.target.v_W, self.target.a_W, self.p_M, self.v_M)
        eff_dist = min(kin_true.r, substep_dist)
        self.min_miss_distance = min(self.min_miss_distance, eff_dist)

        killed = bool(eff_dist <= lim.kill_radius)
        crashed = bool(self.p_M[2] < 0.0)
        diverged = bool(self.uses_apn and self.step_count > 30
                        and kin_true.v_cl < -0.20 * lim.v_max
                        and kin_true.r > self.min_miss_distance + 60.0)
        timeout = bool(self.step_count >= self.max_steps)

        r_progress = 2.5 * (self.prev_dist - kin_true.r) / max(lim.v_max * self.dt, 1e-3)
        v_hat = self.v_M / max(float(np.linalg.norm(self.v_M)), 1e-6)
        r_align = 0.35 * float(np.dot(v_hat, kin_true.los_vec))
        r_los = -0.15 * float(np.linalg.norm(kin_true.los_rate_vec))
        r_zem = -0.25 * (kin_true.zem_norm / max(kin_true.r, 10.0))
        if self.mode == "apn_rl":
            r_residual = -0.04 * float(np.linalg.norm(act)) ** 2
        elif self.mode == "pure_rl":
            r_residual = -0.015 * float(np.linalg.norm(act)) ** 2
        else:
            r_residual = 0.0
        r_smooth = -0.03 * float(np.sum((act - self.prev_action) ** 2))
        reward = r_progress + r_align + r_los + r_zem + r_residual + r_smooth - 0.005
        if killed:
            reward += 150.0 + 50.0 * (1.0 - self.step_count / self.max_steps)
        elif crashed:
            reward -= 80.0
        elif diverged:
            reward -= 25.0 * (self.min_miss_distance / lim.r_init_max)
        self.prev_dist = kin_true.r
        self.prev_action = act.copy()

        time_remaining = max(0.0, (self.max_steps - self.step_count) * self.dt)
        opt_a, opt_v = compute_optimal_intercept_oracle(
            self.p_M, self.v_M, self.target.p_W, self.target.v_W, self.target.a_W,
            lim.v_max, lim.a_n_max, kin_true.t_go)
        tf_p, tf_v, tf_a = self.target.state_at_future(15.0 * self.dt)
        obs = self.obs_builder.build(
            p_M=self.p_M, v_M=self.v_M, est_p_T=est_p, est_v_T=est_v,
            est_a_T=est_a, visible=visible, dt=self.dt,
            time_remaining=time_remaining, optimal_accel=opt_a, optimal_vel=opt_v,
            true_future_pos=tf_p, true_future_vel=tf_v, true_acc=tf_a)

        terminated = killed or crashed or diverged
        truncated = bool(timeout and not terminated)
        info = {
            "killed": killed, "crashed": crashed, "diverged": diverged,
            "miss_distance": self.min_miss_distance,
            "maneuver": getattr(self.target, "maneuver", ManeuverType.NONE).value,
            "platform": self.platform.value, "mode": self.mode,
            "velocity_output": velocity_output, "visible": visible,
            "steps": self.step_count,
            "a_des": a_cmd.copy(), "v_des": self.v_M.copy(),
            "omega_des": self.omega_M.copy(),
            "a_apn": np.asarray(a_apn).copy(), "a_rl": a_rl.copy(),
            "distance": kin_true.r,
        }
        self.last_info = info
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------- misc
    def get_trajectory_log(self):
        return {"interceptor": np.asarray(self.traj_M),
                "target": np.asarray(self.traj_T),
                "target_noisy": np.asarray(self.traj_T_noisy)}

    def render(self):
        return None

    def close(self):
        return None
