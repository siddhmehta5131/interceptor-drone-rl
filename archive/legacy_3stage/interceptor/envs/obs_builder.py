"""obs_builder.py -- observation assembly for the APN + RL interceptor.

The actor observation is a flat vector made of three blocks:

1. **History block** -- ``frame_dim * (1 + m)`` values.  The current 16-dim
   kinematic frame plus ``m`` previous frames sampled every ``p`` steps.
2. **Future block** -- ``7 * n`` values.  ``n`` predicted samples of
   ``(rel_pos(3), rel_vel(3), visible(1))`` at horizons ``q, 2q, ... , nq``
   steps ahead, produced by a lightweight predictor (``const_vel`` / ``kalman``)
   or by the ground-truth target path (``future_source == "true"``).

With the plan defaults (m=3, p=2, n=3, q=5) the actor observation is
``16*4 + 7*3 = 85`` values.

The privileged (critic-only) block is 16 values assembled by the environment
from the optimal-trajectory oracle and the true target future:

``time_remaining(1) + optimal_accel(3) + optimal_vel(3)
+ target_true_future_pos(3) + target_true_future_vel(3) + target_true_accel(3)``

The single 16-dim kinematic frame layout (plan §7.2):

======  =================  ========================================
index   name               meaning
======  =================  ========================================
0-2     rel_pos            target - interceptor position (world)
3-5     rel_vel            target - interceptor velocity (world)
6-8     interceptor_vel    interceptor velocity (world)
9       range              |rel_pos|
10      closing_vel        -dot(rel_vel, rel_pos_hat)
11-13   LOS_rate           line-of-sight rotation rate vector
14      t_go_est           range / max(closing_vel, eps), clamped
15      visible            target visibility flag
======  =================  ========================================
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np

from .stage_config import ObsStageConfig

__all__ = [
    "ObsBuilder",
    "PRIVILEGED_FIELDS",
    "FRAME_FIELDS",
    "FRAME_DIM",
    "FUTURE_SAMPLE_DIM",
    "NORM_SCALES",
]

#: Single-frame kinematic layout (name -> width).
FRAME_FIELDS: Dict[str, int] = {
    "rel_pos": 3,
    "rel_vel": 3,
    "interceptor_vel": 3,
    "range": 1,
    "closing_vel": 1,
    "LOS_rate": 3,
    "t_go_est": 1,
    "visible": 1,
}
FRAME_DIM = int(sum(FRAME_FIELDS.values()))  # 16

#: Privileged (critic) layout -- consumed by AsymmetricActorCriticPolicy.
PRIVILEGED_FIELDS: Dict[str, int] = {
    "time_remaining": 1,
    "optimal_accel": 3,
    "optimal_vel": 3,
    "target_true_future_pos": 3,
    "target_true_future_vel": 3,
    "target_true_accel": 3,
}
PRIVILEGED_DIM = int(sum(PRIVILEGED_FIELDS.values()))  # 16

#: Width of one future sample: rel_pos(3) + rel_vel(3) + visible(1).
FUTURE_SAMPLE_DIM = 7

#: Per-field normalisation scales applied before stacking (keeps inputs O(1)).
NORM_SCALES: Dict[str, float] = {
    "rel_pos": 100.0,
    "rel_vel": 60.0,
    "interceptor_vel": 60.0,
    "range": 100.0,
    "closing_vel": 60.0,
    "LOS_rate": 1.0,
    "t_go_est": 20.0,
    "future_pos": 100.0,
    "future_vel": 60.0,
    "time_remaining": 20.0,
    "optimal_accel": 15.0,
    "optimal_vel": 60.0,
}

_TGO_CLAMP = 20.0
_EPS = 1e-6


class ObsBuilder:
    """Builds actor and privileged observations for one environment."""

    def __init__(self, obs_config: ObsStageConfig, predictor=None):
        self.cfg = obs_config
        self.predictor = predictor
        self._m = int(obs_config.history_frames)
        self._p = max(1, int(obs_config.history_skip))
        self._n = int(obs_config.future_samples)
        self._q = max(1, int(obs_config.future_skip))
        # Need frames at indices 0, p, 2p, ..., m*p.
        self._history: deque = deque(maxlen=self._m * self._p + 1)
        self._last_frame: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ dims
    @property
    def actor_dim(self) -> int:
        return self.cfg.actor_dim

    @property
    def privileged_dim(self) -> int:
        return PRIVILEGED_DIM

    def reset(self) -> None:
        self._history.clear()
        self._last_frame = None

    # ---------------------------------------------------------------- frame
    def compute_frame(
        self,
        interceptor_pos: np.ndarray,
        interceptor_vel: np.ndarray,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
        target_visible: bool,
    ) -> np.ndarray:
        """Assemble the raw (un-normalised) 16-dim current frame."""
        rel_pos = np.asarray(target_pos, dtype=np.float64) - np.asarray(interceptor_pos, dtype=np.float64)
        rel_vel = np.asarray(target_vel, dtype=np.float64) - np.asarray(interceptor_vel, dtype=np.float64)
        rng = float(np.linalg.norm(rel_pos))
        rhat = rel_pos / rng if rng > _EPS else np.zeros(3)
        closing = -float(np.dot(rel_vel, rhat))
        los_rate = np.cross(rel_pos, rel_vel) / (rng * rng + _EPS)
        t_go = float(np.clip(rng / max(closing, 0.1), 0.0, _TGO_CLAMP))

        frame = np.concatenate([
            rel_pos,
            rel_vel,
            np.asarray(interceptor_vel, dtype=np.float64),
            [rng],
            [closing],
            los_rate,
            [t_go],
            [1.0 if target_visible else 0.0],
        ])
        return frame.astype(np.float64)

    def _normalise_frame(self, frame: np.ndarray) -> np.ndarray:
        s = NORM_SCALES
        out = frame.copy()
        out[0:3] /= s["rel_pos"]
        out[3:6] /= s["rel_vel"]
        out[6:9] /= s["interceptor_vel"]
        out[9] /= s["range"]
        out[10] /= s["closing_vel"]
        out[11:14] /= s["LOS_rate"]
        out[14] /= s["t_go_est"]
        # out[15] visible already in {0,1}
        return out

    # --------------------------------------------------------------- history
    def _push(self, frame: np.ndarray) -> None:
        self._history.appendleft(frame)
        self._last_frame = frame

    def _history_block(self) -> np.ndarray:
        frames = list(self._history)  # newest first
        block = []
        for k in range(self._m + 1):
            idx = k * self._p
            if idx < len(frames):
                block.append(self._normalise_frame(frames[idx]))
            else:
                block.append(self._normalise_frame(frames[-1]) if frames else np.zeros(FRAME_DIM))
        return np.concatenate(block)

    # ---------------------------------------------------------------- future
    def _future_block(
        self,
        dt: float,
        interceptor_pos: np.ndarray,
        interceptor_vel: np.ndarray,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
        target_acc: np.ndarray,
        target_visible: bool,
        target_state_fn=None,
    ) -> np.ndarray:
        ip = np.asarray(interceptor_pos, dtype=np.float64)
        iv = np.asarray(interceptor_vel, dtype=np.float64)
        tpos = np.asarray(target_pos, dtype=np.float64)
        tvel = np.asarray(target_vel, dtype=np.float64)
        tacc = np.asarray(target_acc, dtype=np.float64)
        use_true = self.cfg.future_source == "true" and target_state_fn is not None

        samples = []
        for k in range(self._n):
            ahead = float(self._q * (k + 1)) * dt
            if use_true:
                fp, fv, _ = target_state_fn(ahead)
            elif self.predictor is not None:
                fp, fv = self.predictor.predict(tpos, tvel, tacc, ahead)
            else:
                fp, fv = tpos + tvel * ahead, tvel
            int_future = ip + iv * ahead
            rel_p = (np.asarray(fp) - int_future) / NORM_SCALES["future_pos"]
            rel_v = (np.asarray(fv) - iv) / NORM_SCALES["future_vel"]
            samples.append(np.concatenate([rel_p, rel_v, [1.0 if target_visible else 0.0]]))
        if not samples:
            return np.zeros(0)
        return np.concatenate(samples)

    # ------------------------------------------------------------- privileged
    def make_privileged_dict(
        self,
        time_remaining: float,
        optimal_accel: np.ndarray,
        optimal_vel: np.ndarray,
        target_true_future_pos: np.ndarray,
        target_true_future_vel: np.ndarray,
        target_true_accel: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Return each privileged field as a normalised array, keyed by name."""
        s = NORM_SCALES
        return {
            "time_remaining": np.array([float(time_remaining) / s["time_remaining"]]),
            "optimal_accel": np.asarray(optimal_accel, dtype=np.float64) / s["optimal_accel"],
            "optimal_vel": np.asarray(optimal_vel, dtype=np.float64) / s["optimal_vel"],
            "target_true_future_pos": np.asarray(target_true_future_pos, dtype=np.float64) / s["future_pos"],
            "target_true_future_vel": np.asarray(target_true_future_vel, dtype=np.float64) / s["future_vel"],
            "target_true_accel": np.asarray(target_true_accel, dtype=np.float64) / s["optimal_accel"],
        }

    @staticmethod
    def select_privileged(priv: Dict[str, np.ndarray],
                          fields) -> np.ndarray:
        """Concatenate the selected privileged fields (in the given order)."""
        fields = tuple(fields or ())
        if not fields:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate([np.asarray(priv[name], dtype=np.float64).ravel()
                               for name in fields])

    def make_privileged_block(self, *args, **kwargs) -> np.ndarray:
        """Full 16-dim privileged block (all known fields, plan order)."""
        priv = self.make_privileged_dict(*args, **kwargs)
        return self.select_privileged(priv, tuple(PRIVILEGED_FIELDS.keys()))

    # ----------------------------------------------------------------- build
    def build(
        self,
        *,
        interceptor_pos: np.ndarray,
        interceptor_vel: np.ndarray,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
        target_acc: np.ndarray,
        target_visible: bool,
        dt: float,
        time_remaining: float,
        optimal_accel: np.ndarray,
        optimal_vel: np.ndarray,
        target_true_future_pos: np.ndarray,
        target_true_future_vel: np.ndarray,
        target_true_accel: np.ndarray,
        target_state_fn=None,
        privileged_fields=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(actor_obs, privileged_obs)`` as float32 arrays.

        ``privileged_fields`` selects (and orders) the critic-only columns;
        when ``None`` all :data:`PRIVILEGED_FIELDS` are emitted.
        """
        frame = self.compute_frame(interceptor_pos, interceptor_vel,
                                   target_pos, target_vel, target_visible)
        self._push(frame)
        history = self._history_block()
        future = self._future_block(dt, interceptor_pos, interceptor_vel,
                                    target_pos, target_vel, target_acc,
                                    target_visible, target_state_fn)
        actor = np.concatenate([history, future]).astype(np.float32)
        priv = self.make_privileged_dict(
            time_remaining, optimal_accel, optimal_vel,
            target_true_future_pos, target_true_future_vel, target_true_accel,
        )
        if privileged_fields is None:
            privileged_fields = tuple(PRIVILEGED_FIELDS.keys())
        privileged = self.select_privileged(priv, privileged_fields).astype(np.float32)
        return actor, privileged
