"""obs_builder.py  --  observation construction + history stacking.

Stage 1 (no target): flat 14-dim vector, bit-identical to ``hover_env``::

    [ rot6d(6), v_WB(3), omega_B(3), alt_err(1), v_z(1) ]

Target stages (2+): a single frame is 19-dim::

    [ rot6d(6), v_WB(3), omega_B(3),
      rel_pos_body(3), rel_vel_body(3), visible(1) ]

History stacking (plan §5.3): with ``(m, past, skip=s)`` the observation
concatenates frames at ``t, t-s, ..., t-m*s`` **current-first** (most recent
first), total dim ``(m+1)*19``.  When fewer than ``m*s+1`` frames have been
observed the missing OLDER slots are zero-filled (consistent with the
"target fields zeroed when not visible" convention).

Sensor noise (plan decision: sigma=0.01 everywhere) is applied in the same
RNG draw order used by ``hover_env`` (rot, then v, then omega), so Stage 1
observations are statistically identical.  Target relative fields are drawn
after the body states, but ONLY when the target is visible -- hidden target
fields must remain exactly zero (plan §5.4).
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from .stage_config import ObsStageConfig

__all__ = ["ObsBuilder", "make_base_frame", "add_target_to_frame", "TARGET_FRAME_DIM"]


TARGET_FRAME_DIM = 19
STAGE1_FRAME_DIM = 14


def make_base_frame(
    R_WB: np.ndarray,
    v_WB: np.ndarray,
    omega_B: np.ndarray,
    alt_err: float,
    *,
    include_target: bool,
) -> np.ndarray:
    """Raw (un-noised) single frame body states (rot6 + v + omega).

    ``include_target=False`` appends ``alt_err`` and ``v_z`` (Stage-1 layout);
    ``include_target=True`` leaves the frame open for target fields.
    """
    rot6d = R_WB[:, :2].ravel()
    if not include_target:
        return np.array(
            [*rot6d, *v_WB, *omega_B, float(alt_err), float(v_WB[2])],
            dtype=np.float64,
        )
    return np.array([*rot6d, *v_WB, *omega_B], dtype=np.float64)


def add_target_to_frame(
    frame: np.ndarray,
    rel_pos_body: np.ndarray,
    rel_vel_body: np.ndarray,
    visible: bool,
) -> np.ndarray:
    """Append target-relative fields to a 12-dim base frame -> 19-dim frame.

    When ``visible`` is False the target fields are zeroed (plan §5.4).
    """
    out = np.zeros(TARGET_FRAME_DIM, dtype=np.float32)
    out[:12] = frame[:12]
    if visible:
        out[12:15] = rel_pos_body
        out[15:18] = rel_vel_body
        out[18] = 1.0
    else:
        out[18] = 0.0
    return out


class ObsBuilder:
    """Applies per-step sensor noise and (for target stages) frame stacking.

    The env calls :meth:`update` once per step/reset with the RAW frame and
    receives the final observation.  RNG draws happen here so the parity of
    Stage 1's noise order with ``hover_env`` is preserved regardless of how
    the frame was assembled by the caller.
    """

    def __init__(self, obs_cfg: ObsStageConfig, history_frames: int, history_skip: int, rng: np.random.Generator):
        self.cfg = obs_cfg
        self.include_target = obs_cfg.include_target
        self.history_frames = int(history_frames) if self.include_target else 0
        self.history_skip = max(1, int(history_skip)) if self.include_target else 1
        self.rng = rng

        self.frame_dim = TARGET_FRAME_DIM if self.include_target else STAGE1_FRAME_DIM
        self.window = self.history_frames * self.history_skip + 1
        self._hist: Optional[deque] = None
        self.reset()

    # -- properties -----------------------------------------------------------

    @property
    def obs_dim(self) -> int:
        if self.include_target:
            return TARGET_FRAME_DIM * (self.history_frames + 1)
        return STAGE1_FRAME_DIM

    # -- lifecycle -------------------------------------------------------------

    def reset(self) -> None:
        """Clear history (no frames)."""
        self._hist = deque(maxlen=self.window)

    def update(self, frame: np.ndarray) -> np.ndarray:
        """Push one raw frame; return the stacked, noised observation."""
        noised = self._noise_frame(frame)
        self._hist.append(noised)
        if not self.include_target:
            # single end-of-pipeline cast, mirroring hover_env._get_obs's
            # np.array([...], dtype=np.float32) for bit-level parity
            return np.array(noised, dtype=np.float32)
        stacked = self._stack()
        return np.array(stacked, dtype=np.float32)

    # -- internals --------------------------------------------------------------

    def _noise_frame(self, frame: np.ndarray) -> np.ndarray:
        """Sensor noise; draw order rot(6), v(3), omega(3) then target fields.
        All arithmetic stays float64 until the final cast (hover parity).

        Target relative fields are noised only when the target is visible
        (plan §5.4): hidden fields stay exactly zero.
        """
        s = self.cfg.obs_noise_std
        out = np.asarray(frame, dtype=np.float64).copy()
        if s > 0.0:
            out[0:6] += self.rng.normal(0.0, s, 6)          # rot6d
            out[6:9] += self.rng.normal(0.0, s, 3)          # v_WB
            out[9:12] += self.rng.normal(0.0, s, 3)         # omega_B
            visible = (
                self.include_target
                and frame.shape[0] >= TARGET_FRAME_DIM
                and float(frame[TARGET_FRAME_DIM - 1]) > 0.5
            )
            if visible:
                out[12:15] += self.rng.normal(0.0, s, 3)    # rel_pos_body
                out[15:18] += self.rng.normal(0.0, s, 3)    # rel_vel_body
        if not self.include_target and frame.shape[0] >= 14:
            # Stage-1 parity: obs[13] (v_z) is the NOISED vertical velocity
            # (hover_env: `vz_n = float(v_n[2])` after v_n was perturbed).
            out[13] = out[8]
        return out

    def _stack(self) -> np.ndarray:
        """Concatenate history frames current-first: [t, t-s, ..., t-m*s].

        Missing older slots are zero-padded at the end (plan §5.3).
        """
        hist = list(self._hist)  # oldest -> newest
        n = len(hist)
        frames = []
        for k in range(self.history_frames + 1):
            idx = k * self.history_skip
            if idx < n:
                frames.append(hist[n - 1 - idx])
            else:
                frames.append(np.zeros(self.frame_dim, dtype=np.float32))
        return np.concatenate(frames).astype(np.float32)