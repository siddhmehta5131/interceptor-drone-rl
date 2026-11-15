"""Closed-loop guidance -> flight-controller -> physics (plan P8.5).

The outer loop runs the guidance law at 50 Hz, the inner rate loop at 500 Hz
and the rigid-body/aerodynamic integration at 1000 Hz.  The target and seeker
are the same objects and seed as the kinematic environment, so a closed-loop
episode differs only in how ``a_des`` is realised.

Inner controllers share the ``controller(omega_cmd, collective, omega) -> cmd``
interface of :mod:`interceptor.fc.rollout`: the project PID (B1), the firmware
PID (B2), or a wrapped SB3 policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..fc.firmware_pid import FirmwarePID
from ..fc.pid import RatePID
from ..guidance.env import UnifiedInterceptEnv
from ..guidance.kinematics import (
    closest_approach_distance, compute_3d_apn_command,
    compute_engagement_kinematics,
)
from ..guidance.maneuvers import ManeuverType, PlatformType
from ..guidance.observation import AsymmetricObsBuilder
from ..guidance.platform import InterceptorLimits, PLATFORMS
from ..sim import pipeline
from ..sim.constants import PH_C_HOVER, PH_M, PH_OMEGA_HOVER
from ..sim.wind import DrydenWind
from .setpoint import AccelBridge

__all__ = ["ClosedLoopResult", "run_closed_loop", "PolicyRateController"]

GUIDANCE_DT = 0.02
CONTROL_DT = 0.002
PHYSICS_DT = 0.001


@dataclass
class ClosedLoopResult:
    killed: bool
    miss_distance: float
    steps: int
    time: float
    a_des: np.ndarray
    a_actual: np.ndarray
    accel_tracking_rms: float
    pos: np.ndarray
    target: np.ndarray
    diagnostics: dict = field(default_factory=dict)


class PolicyRateController:
    """Wrap an SB3 rate policy behind the inner-controller interface."""

    def __init__(self, model: Any, rate_max: float = 12.0, n_frames: int = 3):
        self.model = model
        self.rate_max = float(rate_max)
        self.n_frames = int(n_frames)
        self.priv_dim = 16
        self.reset()

    def reset(self):
        self.prev_a = np.zeros(4)
        self.frames = [np.zeros(11) for _ in range(self.n_frames)]

    def _obs(self, omega_cmd, collective, omega) -> np.ndarray:
        e = np.asarray(omega_cmd) - np.asarray(omega)
        frame = np.concatenate([
            np.asarray(omega) / self.rate_max,
            e / self.rate_max,
            [(collective - PH_C_HOVER) / PH_C_HOVER],
            self.prev_a,
        ])
        self.frames = [frame] + self.frames[:-1]
        return np.concatenate(self.frames + [np.zeros(self.priv_dim)]).astype(np.float32)

    def __call__(self, omega_cmd, collective, omega):
        obs = self._obs(omega_cmd, collective, omega)
        action, _ = self.model.predict(obs, deterministic=True)
        a = np.clip(np.asarray(action, dtype=float).reshape(4), -1.0, 1.0)
        self.prev_a = a
        return (a + 1.0) / 2.0


def _inner(name, policy, rate_max):
    if name == "b1":
        return RatePID.tuned(dt=CONTROL_DT)
    if name == "b2":
        return FirmwarePID(dt=CONTROL_DT)
    if name == "rl":
        if policy is None:
            raise ValueError("inner='rl' requires a policy")
        if isinstance(policy, str):
            from stable_baselines3 import PPO
            policy = PPO.load(policy, device="cpu")
        return PolicyRateController(policy, rate_max=rate_max)
    raise ValueError(f"unknown inner controller {name!r}")


def run_closed_loop(
    *,
    platform: PlatformType = PlatformType.QUADCOPTER,
    mode: str = "apn_filtered",
    seed: int = 0,
    inner: str = "b1",
    policy: Any = None,
    inner_policy: Any = None,
    target_set: str = "evasive",
    fixed_maneuver: Optional[ManeuverType] = None,
    N_nav: Optional[float] = None,
    limits: Optional[InterceptorLimits] = None,
    wind: bool = True,
    max_time: Optional[float] = None,
    residual_authority: float = 0.35,
    bridge_rate_limit: float = 12.0,
    noise_scale: float = 1.0,
) -> ClosedLoopResult:
    """Run one closed-loop episode and return the summary and histories."""
    limits = limits or PLATFORMS[platform]
    env = UnifiedInterceptEnv(
        platform=platform, mode=mode, target_set=target_set,
        fixed_maneuver=fixed_maneuver, N_nav=N_nav, seed=seed,
        limits=limits, noise_scale=noise_scale)
    env.reset(seed=seed)

    duration = float(max_time if max_time is not None else limits.max_time)
    n_guidance = int(round(duration / GUIDANCE_DT))
    substeps = int(round(GUIDANCE_DT / CONTROL_DT))

    p = env.p_M.copy()
    v = env.v_M.copy()
    q = np.array([1.0, 0.0, 0.0, 0.0])
    omega = np.zeros(3)
    state = (p, q, v, omega, np.full(4, PH_OMEGA_HOVER))
    params = pipeline.PhysicsParams.rotor_drag()

    w = None
    if wind:
        w = DrydenWind(100.0, np.array([1.0, 0.5, 0.0]), W20=3.0, airspeed=5.0,
                       rng=np.random.default_rng(seed + 991))
    controller = _inner(inner, inner_policy, env.limits.omega_max)
    bridge = AccelBridge(mass=PH_M, rate_limit=bridge_rate_limit)  # scripts/sweep_bridge_rate.py

    obs_builder = AsymmetricObsBuilder(env.limits)
    obs_builder.reset()
    opt_a, opt_v = np.zeros(3), v.copy()
    obs_builder.build(
        p_M=p, v_M=v, est_p_T=env.target.p_W, est_v_T=env.target.v_W,
        est_a_T=env.target.a_W, visible=True, dt=GUIDANCE_DT,
        time_remaining=duration, optimal_accel=opt_a, optimal_vel=opt_v,
        true_future_pos=env.target.p_W, true_future_vel=env.target.v_W,
        true_acc=env.target.a_W)

    a_des_hist: list[np.ndarray] = []
    a_actual_hist: list[np.ndarray] = []
    pos_hist = [p.copy()]
    vel_hist = [v.copy()]
    target_hist = [env.target.p_W.copy()]
    min_miss = float(np.linalg.norm(env.target.p_W - p))
    killed = False
    t = 0.0

    for _ in range(n_guidance):
        env.target.step(GUIDANCE_DT, t, p, v)
        est_p, est_v, est_a, visible = env.sensor.observe(
            p, env.target.p_W, env.target.v_W, env.target.a_W,
            GUIDANCE_DT, env.noise_scale)

        if mode == "apn_raw":
            kin = compute_engagement_kinematics(
                env.sensor.raw_pos, env.sensor.raw_vel, env.sensor.raw_acc,
                p, v, r_min_clamp=2.0)
            a_cmd = compute_3d_apn_command(
                kin, env.sensor.raw_acc, v, env.limits, env.N_nav,
                include_axial_boost=False)
        else:
            kin = compute_engagement_kinematics(
                est_p, est_v, est_a, p, v, r_min_clamp=15.0)
            a_cmd = compute_3d_apn_command(
                kin, est_a, v, env.limits, env.N_nav, include_axial_boost=True)
            if mode == "apn_rl":
                obs_full = obs_builder.build(
                    p_M=p, v_M=v, est_p_T=est_p, est_v_T=est_v, est_a_T=est_a,
                    visible=visible, dt=GUIDANCE_DT,
                    time_remaining=max(0.0, duration - t),
                    optimal_accel=opt_a, optimal_vel=opt_v,
                    true_future_pos=env.target.p_W,
                    true_future_vel=env.target.v_W, true_acc=env.target.a_W)
                if policy is not None:
                    action, _ = (policy.predict(obs_full, deterministic=True)
                                 if hasattr(policy, "predict")
                                 else (policy(obs_full), None))
                    action = np.clip(np.asarray(action, dtype=float).reshape(3), -1, 1)
                    a_cmd = a_cmd + action * (residual_authority * env.limits.a_n_max)
        sp = bridge(q, a_cmd)
        a_des_hist.append(a_cmd.copy())

        for _ in range(substeps):
            cmd = controller(sp.omega_cmd, sp.collective, omega)
            p_prev = state[0].copy()
            ww = None if w is None else w.step(CONTROL_DT)
            state = pipeline.control_step(
                state, cmd, 4.2, params, CONTROL_DT, PHYSICS_DT, wind_W=ww)
            p, q, v, omega, om = state
            seg_miss = closest_approach_distance(
                env.target.p_W, env.target.p_W, p_prev, p)
            min_miss = min(min_miss, seg_miss)
            if seg_miss <= env.limits.kill_radius:
                killed = True
                break
        pos_hist.append(p.copy())
        vel_hist.append(v.copy())
        target_hist.append(env.target.p_W.copy())
        t += GUIDANCE_DT
        if killed:
            break

    a_des_arr = np.asarray(a_des_hist)
    vel_arr = np.asarray(vel_hist)
    if len(vel_arr) >= 2:
        a_actual = np.diff(vel_arr, axis=0) / GUIDANCE_DT
        n = min(len(a_actual), len(a_des_arr))
        rms = float(np.sqrt(np.mean(
            np.sum((a_actual[-n:] - a_des_arr[-n:]) ** 2, axis=1)))) if n else float("nan")
    else:
        a_actual = np.zeros((0, 3))
        rms = float("nan")
    return ClosedLoopResult(
        killed=killed, miss_distance=float(min_miss),
        steps=len(a_des_hist), time=float(len(a_des_hist) * GUIDANCE_DT),
        a_des=a_des_arr, a_actual=a_actual, accel_tracking_rms=rms,
        pos=np.asarray(pos_hist), target=np.asarray(target_hist),
        diagnostics={"inner": inner, "mode": mode, "platform": platform.value,
                     "wind": bool(wind)})
