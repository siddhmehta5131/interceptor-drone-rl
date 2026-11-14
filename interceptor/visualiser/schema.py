"""Single trajectory-log schema for guidance, flight-controller and closed-loop runs (P10.1).

Every run, whatever produced it, is converted to a :class:`TrajectoryLog`.
The renderer only ever sees this object, so a new run type needs one adapter
and no renderer change.  ``to_frames`` yields the legacy per-step dicts that
:class:`~interceptor.visualiser.animator.InterceptAnimator` already consumes,
so the original stage videos keep rendering.

Conventions: world frame ENU-like with +z up (as the simulator), quaternions
(w, x, y, z) body -> world, SI units.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = ["TrajectoryLog", "from_guidance_env", "rollout_guidance",
           "from_fc_rollout", "from_closed_loop", "from_legacy_frames", "KINDS"]

KINDS = ("guidance", "fc", "closed_loop")
_VEC3 = ("interceptor_pos", "target_pos", "a_cmd", "a_apn", "a_rl",
         "omega_cmd", "omega")
_QUAT = ("interceptor_quat", "target_quat")


@dataclass
class TrajectoryLog:
    """Time series of one run.  Optional channels are zero-filled when absent."""

    kind: str
    dt: float
    interceptor_pos: np.ndarray                    # (N, 3)
    target_pos: np.ndarray                         # (N, 3), may be zeros for fc runs
    interceptor_quat: Optional[np.ndarray] = None  # (N, 4)
    target_quat: Optional[np.ndarray] = None       # (N, 4)
    a_cmd: Optional[np.ndarray] = None             # total commanded acceleration (N, 3)
    a_apn: Optional[np.ndarray] = None             # (N, 3)
    a_rl: Optional[np.ndarray] = None              # (N, 3)
    omega_cmd: Optional[np.ndarray] = None         # commanded body rate (N, 3), fc / closed loop
    omega: Optional[np.ndarray] = None             # measured body rate (N, 3)
    kill_radius: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ basics
    def __len__(self) -> int:
        return int(self.interceptor_pos.shape[0])

    @property
    def t(self) -> np.ndarray:
        return np.arange(len(self)) * self.dt

    def validate(self) -> "TrajectoryLog":
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if not self.dt > 0:
            raise ValueError("dt must be positive")
        n = len(self)
        if n < 1:
            raise ValueError("empty log")
        for name in _VEC3:
            arr = getattr(self, name)
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=float)
            if arr.shape != (n, 3):
                raise ValueError(f"{name} must have shape ({n}, 3), got {arr.shape}")
            setattr(self, name, arr)
        for name in _QUAT:
            arr = getattr(self, name)
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=float)
            if arr.shape != (n, 4):
                raise ValueError(f"{name} must have shape ({n}, 4), got {arr.shape}")
            setattr(self, name, arr)
        if not np.all(np.isfinite(self.interceptor_pos)):
            raise ValueError("interceptor_pos contains non-finite values")
        return self

    def distance(self) -> np.ndarray:
        return np.linalg.norm(self.target_pos - self.interceptor_pos, axis=1)

    def _channel(self, name: str, width: int = 3) -> np.ndarray:
        arr = getattr(self, name)
        return np.zeros((len(self), width)) if arr is None else arr

    # ------------------------------------------------------------------ files
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {"kind": self.kind, "dt": self.dt,
                                   "kill_radius": np.nan if self.kill_radius is None
                                   else self.kill_radius,
                                   "meta": repr(self.meta)}
        for name in _VEC3 + _QUAT:
            arr = getattr(self, name)
            if arr is not None:
                payload[name] = arr
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TrajectoryLog":
        import ast
        with np.load(Path(path), allow_pickle=False) as z:
            kw: Dict[str, Any] = {name: z[name] for name in _VEC3 + _QUAT if name in z.files}
            kr = float(z["kill_radius"])
            meta = ast.literal_eval(str(z["meta"])) if "meta" in z.files else {}
            return cls(kind=str(z["kind"]), dt=float(z["dt"]),
                       kill_radius=None if np.isnan(kr) else kr, meta=meta, **kw).validate()

    # ------------------------------------------------------------ legacy frames
    def to_frames(self) -> List[Dict[str, Any]]:
        """Per-step dicts in the format consumed by ``InterceptAnimator``."""
        from .scene_builder import quat_from_direction
        dist = self.distance()
        a_cmd = self._channel("a_cmd")
        frames = []
        for k in range(len(self)):
            nxt = min(k + 1, len(self) - 1)
            prv = max(k - 1, 0)
            vel = (self.interceptor_pos[nxt] - self.interceptor_pos[prv]) / (
                max(nxt - prv, 1) * self.dt)
            tvel = (self.target_pos[nxt] - self.target_pos[prv]) / (
                max(nxt - prv, 1) * self.dt)
            quat = (self.interceptor_quat[k] if self.interceptor_quat is not None
                    else quat_from_direction(vel))
            closing = float(-np.dot(self.target_pos[k] - self.interceptor_pos[k],
                                    tvel - vel) / max(dist[k], 1e-9))
            rel = self.target_pos[k] - self.interceptor_pos[k]
            los_rate = float(np.linalg.norm(np.cross(rel, tvel - vel)) / max(dist[k], 1e-9) ** 2)
            frames.append({
                "step": k, "interceptor_pos": self.interceptor_pos[k],
                "interceptor_vel": vel, "interceptor_quat": quat,
                "target_pos": self.target_pos[k], "target_vel": tvel,
                "a_apn": self._channel("a_apn")[k], "a_rl": self._channel("a_rl")[k],
                "a_total": a_cmd[k],
                "velocity_output": np.concatenate([vel, np.zeros(3)]),
                "distance": float(dist[k]), "closing_vel": closing,
                "LOS_rate": los_rate, "t_go": float(dist[k] / max(abs(closing), 1e-6)),
            })
        return frames


# ----------------------------------------------------------------------- adapters
def from_guidance_env(env, a_apn: List[np.ndarray], a_rl: List[np.ndarray],
                      meta: Optional[dict] = None) -> TrajectoryLog:
    """Build a log from a finished :class:`UnifiedInterceptEnv` episode.

    Position channels include the reset state; command channels are
    left-padded with a zero row so every channel has the same length.
    """
    log = env.get_trajectory_log()
    n = len(log["interceptor"])

    def align(rows):
        """Command channels have one row per step; positions also hold the reset state."""
        arr = np.asarray(rows, dtype=float).reshape(-1, 3)
        pad = np.zeros((max(n - len(arr), 0), 3))
        return np.concatenate([pad, arr])[:n]

    a_apn_arr, a_rl_arr, a_des = align(a_apn), align(a_rl), align(log["a_des"])
    return TrajectoryLog(
        kind="guidance", dt=float(env.dt),
        interceptor_pos=log["interceptor"], target_pos=log["target"],
        a_cmd=a_des, a_apn=a_apn_arr, a_rl=a_rl_arr,
        kill_radius=float(env.limits.kill_radius),
        meta={"platform": env.platform.value, "mode": env.mode, **(meta or {})},
    ).validate()


def rollout_guidance(platform, mode: str, behavior, seed: int, policy=None,
                     target_set: str = "evasive", noise_scale: float = 1.0) -> TrajectoryLog:
    """Run one guidance episode (any arm) and return its log (P10.2).

    ``behavior`` is a :class:`ManeuverType` for the evasive set.  ``policy`` is
    an SB3 model (or ``None`` for the non-learned arms).
    """
    from ..guidance.env import UnifiedInterceptEnv
    kwargs: Dict[str, Any] = {"fixed_maneuver": behavior} if target_set == "evasive" else {}
    env = UnifiedInterceptEnv(platform=platform, mode=mode, target_set=target_set,
                              noise_scale=noise_scale, seed=seed, **kwargs)
    obs, _ = env.reset(seed=seed)
    a_apn, a_rl, done = [], [], False
    while not done:
        if policy is not None:
            action, _ = policy.predict(obs, deterministic=True)
        else:
            action = np.zeros(3, dtype=np.float32)
        obs, _, terminated, truncated, info = env.step(action)
        a_apn.append(info["a_apn"])
        a_rl.append(info["a_rl"])
        done = bool(terminated or truncated)
    return from_guidance_env(env, a_apn, a_rl, meta={
        "seed": seed, "killed": bool(info["killed"]),
        "miss_distance": float(env.min_miss_distance),
        "behavior": getattr(behavior, "value", str(behavior))})


def from_fc_rollout(log: dict, label: str = "") -> TrajectoryLog:
    """Adapt :func:`interceptor.fc.rollout.rollout` output (dt from ``log['t']``)."""
    t = np.asarray(log["t"])
    dt = float(t[1] - t[0]) if len(t) > 1 else 0.002
    sp = np.asarray(log["setpoint"])
    pos = np.asarray(log["p"])
    return TrajectoryLog(
        kind="fc", dt=dt, interceptor_pos=pos, target_pos=np.zeros_like(pos),
        interceptor_quat=np.asarray(log["q"]), omega_cmd=sp[:, :3],
        omega=np.asarray(log["omega"]), meta={"label": label},
    ).validate()


def from_closed_loop(result, dt: float = 0.02, label: str = "") -> TrajectoryLog:
    """Adapt a :class:`ClosedLoopResult` (positions sampled at the guidance rate)."""
    n = min(len(result.pos), len(result.target))
    a = np.zeros((n, 3))
    m = min(len(result.a_des), n)
    a[:m] = result.a_des[:m]
    from ..guidance.platform import PLATFORMS
    from ..guidance.maneuvers import PlatformType
    kill = PLATFORMS[PlatformType(result.diagnostics.get("platform", "quadcopter"))].kill_radius
    return TrajectoryLog(
        kind="closed_loop", dt=dt, interceptor_pos=result.pos[:n],
        target_pos=result.target[:n], a_cmd=a, kill_radius=float(kill),
        meta={"label": label, "killed": bool(result.killed),
              "miss_distance": float(result.miss_distance), **result.diagnostics},
    ).validate()


def from_legacy_frames(frames: List[Dict[str, Any]], dt: float = 0.02,
                       kill_radius: Optional[float] = None) -> TrajectoryLog:
    """Adapt the old ``InterceptEnv`` trajectory-log list (existing stage videos)."""
    def col(key, width=3):
        return np.asarray([np.asarray(f.get(key, np.zeros(width)), float) for f in frames])
    return TrajectoryLog(
        kind="guidance", dt=dt, interceptor_pos=col("interceptor_pos"),
        target_pos=col("target_pos"), interceptor_quat=col("interceptor_quat", 4),
        a_apn=col("a_apn"), a_rl=col("a_rl"), a_cmd=col("a_total"),
        kill_radius=kill_radius, meta={"legacy": True},
    ).validate()
