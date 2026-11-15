"""Flight-controller Gymnasium environment (plan P4.1, P4.3).

Task: track body-rate and collective setpoints.  Action: four values in [-1, 1]
mapped to motor commands ``(a + 1) / 2``.  Observation (actor, 3 frames x 11):
``[omega/r_max (3), e/r_max (3), collective error (1), previous action (4)]``,
followed by a privileged critic block of 16 values (mass, inertia diagonal,
c_l, c_d, rotor speeds, wind, body velocity) that actors must not see.
Reward: EQ-FC-2.
"""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..common.seeding import make_rng
from ..common.sources import load_yaml
from ..sim import pipeline
from ..sim.constants import PH_C_HOVER, PH_M, PH_OMEGA_HOVER, esc_omega_ss
from ..sim.quaternion import quat_mult
from ..sim.wind import DrydenWind
from . import setpoints
from .thrust import collective_for_thrust

ROOT = Path(__file__).resolve().parents[2]
FRAME = 11
N_FRAMES = 3
ACTOR_DIM = FRAME * N_FRAMES  # 33
PRIV_DIM = 16


def load_fc_config() -> dict:
    raw = load_yaml(ROOT / "configs" / "fc.yaml")
    def unwrap(v):
        if isinstance(v, dict) and "value" in v:
            return v["value"]
        if isinstance(v, dict):
            return {k: unwrap(x) for k, x in v.items()}
        return v
    return {k: unwrap(v) for k, v in raw.items()}


def reward_terms(e, rate_max, thrust_err, d_action, sat_frac, cfg):
    """EQ-FC-2.  Returns the scalar reward."""
    return float(-cfg["w_omega"] * np.sum((e / rate_max) ** 2)
                 - cfg["w_thrust"] * thrust_err ** 2
                 - cfg["w_smooth"] * np.sum(d_action ** 2)
                 - cfg["w_sat"] * sat_frac)


def _random_quat(rng, max_tilt):
    axis = rng.normal(size=3)
    axis[2] = 0.0
    n = np.linalg.norm(axis)
    axis = axis / n if n > 1e-9 else np.array([1.0, 0, 0])
    ang = rng.uniform(0, max_tilt)
    yaw = rng.uniform(-np.pi, np.pi)
    qt = np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * axis])
    qy = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    return quat_mult(qy, qt)


class FlightControllerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None, domain_randomization: bool = True,
                 wind: bool = True, kinds=setpoints.KINDS, cfg: dict | None = None):
        self.cfg = cfg or load_fc_config()
        self.dt = float(self.cfg["control_dt"])
        self.pdt = float(self.cfg["physics_dt"])
        self.n_steps = int(round(self.cfg["episode_length"] / self.dt))
        self.rate_max = float(self.cfg["rate_max"])
        self.dr, self.use_wind, self.kinds = domain_randomization, wind, tuple(kinds)
        self.rng = make_rng(seed)
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (ACTOR_DIM + PRIV_DIM,), np.float32)
        self._seed = seed

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = make_rng(seed)
        r = self.rng
        self.params = pipeline.sample_physics_params(r) if self.dr else pipeline.PhysicsParams.rotor_drag()
        q = _random_quat(r, self.cfg["init_tilt_max"])
        omega = r.uniform(-1, 1, 3) * self.cfg["init_rate_max"]
        v = r.normal(0, 1.0, 3)
        Om = np.full(4, PH_OMEGA_HOVER)
        self.state = (np.array([0.0, 0.0, 100.0]), q, v, omega, Om)
        self.wind = None
        if self.use_wind:
            mean = r.normal(0, 1.0, 3)
            self.wind = DrydenWind(100.0, mean, W20=max(np.linalg.norm(mean), 1.0),
                                   airspeed=5.0, rng=r)
        kind = self.kinds[int(r.integers(len(self.kinds)))]
        self.sp = setpoints.sample(kind, self.n_steps + 1, self.dt, r, self.rate_max)
        self.k = 0
        self.prev_a = np.zeros(4)
        self.frames = [np.zeros(FRAME) for _ in range(N_FRAMES)]
        self._push_frame()
        return self._obs(), {"kind": kind}

    def _push_frame(self):
        om = self.state[3]
        cmd = self.sp[self.k]
        e = cmd[:3] - om
        f = np.concatenate([om / self.rate_max, e / self.rate_max,
                            [(cmd[3] - PH_C_HOVER) / PH_C_HOVER], self.prev_a])
        self.frames = [f] + self.frames[:-1]

    def _privileged(self):
        _, q, v, _, Om = self.state
        w = np.zeros(3) if self.wind is None else self.wind.mean_wind
        from ..sim.quaternion import quat2rot
        vb = quat2rot(q).T @ v
        p = self.params
        return np.concatenate([[p.mass / PH_M], np.diag(p.J) / np.diag(pipeline.PH_J),
                               [p.c_l / pipeline.NOM_C_L, p.c_d / pipeline.NOM_C_D],
                               Om / PH_OMEGA_HOVER - 1.0, w, vb / 5.0])[:PRIV_DIM]

    def _obs(self):
        return np.concatenate(self.frames + [self._privileged()]).astype(np.float32)

    def step(self, action):
        a = np.clip(np.asarray(action, float), -1.0, 1.0)
        cmd = (a + 1.0) / 2.0
        w = None if self.wind is None else self.wind.step(self.dt)
        sp = self.sp[self.k]
        self.state = pipeline.control_step(self.state, cmd, 4.2, self.params, self.dt,
                                           self.pdt, wind_W=w)
        self.k += 1
        om = self.state[3]
        e = sp[:3] - om
        # relative thrust error: actual total thrust vs thrust for the commanded collective
        f_actual = float(np.sum(self.params.c_l * np.maximum(self.state[4], 0.0) ** 2))
        f_cmd_c = 4.0 * self.params.c_l * float(esc_omega_ss(sp[3])) ** 2
        thrust_err = (f_actual - f_cmd_c) / (PH_M * 9.81)
        sat = float(np.mean((cmd <= 1e-9) | (cmd >= 1 - 1e-9)))
        rew = reward_terms(e, self.rate_max, thrust_err, a - self.prev_a, sat, self.cfg)
        self.prev_a = a
        self._push_frame()
        diverged = bool(np.max(np.abs(om)) > self.cfg["safety_rate"] or not np.all(np.isfinite(om)))
        if diverged:
            rew -= 10.0
        trunc = self.k >= self.n_steps
        return self._obs(), rew, diverged, trunc, {"omega": om.copy(), "omega_cmd": sp[:3].copy()}
