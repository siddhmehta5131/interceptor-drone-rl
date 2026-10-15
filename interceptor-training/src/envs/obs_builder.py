"""obs_builder.py  --  observation construction: history, future, privileged.

Stage 1 (no target): flat 14-dim vector, bit-identical to ``hover_env``::

    [ rot6d(6), v_WB(3), omega_B(3), alt_err(1), v_z(1) ]

Target stages (2+): a single frame is 19-dim::

    [ rot6d(6), v_WB(3), omega_B(3),
      rel_pos_body(3), rel_vel_body(3), visible(1) ]

The final observation is three concatenated blocks (plan D-11, D-12, D-13):

1. **History** -- ``m + 1`` frames at ``t, t-s, ..., t-m*s``, current-first,
   each 19-dim  =>  ``19 * (m + 1)``.
2. **Future** -- ``n`` samples at ``t = q*dt, 2*q*dt, ..., n*q*dt``
   (``q = future_skip``, D-18/D-38), each
   ``[rel_pos_body(3), rel_vel_body(3), visible(1)]``  =>  ``7 * n``.
   Every sample is *strictly* in the future; the current instant already lives
   in the newest history frame, so it is not repeated here.
3. **Privileged** -- the fields selected by the model definition
   (D-12), appended **only** when the model declares ``privileged_critic``.
   Privileged values are exact ground truth and carry **no** sensor noise.

Missing OLDER history slots are zero-filled (consistent with the
"target fields zeroed when not visible" convention).

Sensor noise (sigma = 0.01) is applied in the same RNG draw order used by
``hover_env`` (rot, then v, then omega), so Stage 1 observations are
statistically identical.  Target relative fields are drawn after the body
states, but ONLY when the target is visible; hidden target fields must remain
exactly zero.  All arithmetic stays float64 until the single final cast.
"""

from __future__ import annotations

from collections import deque
from typing import Optional, Sequence

import numpy as np

from .stage_config import ObsStageConfig

__all__ = [
    "ObsBuilder",
    "make_base_frame",
    "add_target_to_frame",
    "make_future_block",
    "make_privileged_block",
    "TARGET_FRAME_DIM",
    "STAGE1_FRAME_DIM",
    "FUTURE_SAMPLE_DIM",
    "PRIVILEGED_FIELDS",
    "privileged_dim",
    "observation_dim",
]


TARGET_FRAME_DIM = 19
STAGE1_FRAME_DIM = 14

#: one future sample is rel_pos(3) + rel_vel(3) + visible(1)
FUTURE_SAMPLE_DIM = 7

#: name -> width of every privileged field the model definitions may request
PRIVILEGED_FIELDS = {
    "time_remaining": 1,
    "facing_error": 1,
    "target_true_pos": 3,
    "target_true_vel": 3,
}


def privileged_dim(fields: Sequence[str]) -> int:
    """Total width of ``fields`` (0 when the critic gets no privileged input)."""
    return sum(PRIVILEGED_FIELDS[f] for f in fields or ())


def observation_dim(
    *,
    include_target: bool,
    history_frames: int = 0,
    future_samples: int = 0,
    privileged_fields: Sequence[str] = (),
) -> int:
    """Total actor-observation width for the given settings."""
    if not include_target:
        return STAGE1_FRAME_DIM
    return (
        TARGET_FRAME_DIM * (int(history_frames) + 1)
        + FUTURE_SAMPLE_DIM * int(future_samples)
        + privileged_dim(privileged_fields)
    )


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
    out = np.zeros(TARGET_FRAME_DIM, dtype=np.float64)
    out[:12] = frame[:12]
    if visible:
        out[12:15] = rel_pos_body
        out[15:18] = rel_vel_body
        out[18] = 1.0
    return out


def make_future_block(
    pred_rel_pos_body: np.ndarray,
    pred_rel_vel_body: np.ndarray,
    visible: bool,
) -> np.ndarray:
    """Build the ``(n, 7)`` raw future block.

    ``pred_rel_pos_body`` / ``pred_rel_vel_body`` are ``(n, 3)`` arrays of
    *body-frame* relative position / velocity for the ``n`` strictly-future
    samples ``t + q*dt, ..., t + n*q*dt`` (the caller strips the predictor's
    "now" row).  When the target is hidden every sample is zeroed except the
    ``visible`` flag.
    """
    pp = np.asarray(pred_rel_pos_body, dtype=np.float64).reshape(-1, 3)
    pv = np.asarray(pred_rel_vel_body, dtype=np.float64).reshape(-1, 3)
    n = pp.shape[0]
    if pv.shape[0] != n:
        raise ValueError(f"future pos/vel length mismatch: {pp.shape} vs {pv.shape}")
    block = np.zeros((n, FUTURE_SAMPLE_DIM), dtype=np.float64)
    if visible:
        block[:, 0:3] = pp
        block[:, 3:6] = pv
        block[:, 6] = 1.0
    return block


def make_privileged_block(
    fields: Sequence[str],
    *,
    time_remaining: float = 0.0,
    facing_error: float = 0.0,
    target_true_pos_body: Optional[np.ndarray] = None,
    target_true_vel_body: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Concatenate the requested privileged fields in ``fields`` order (D-12).

    ``target_true_pos_body`` is the *ideal minus current* body-frame position
    error and ``target_true_vel_body`` the corresponding velocity error, so
    the critic sees what the predictor got wrong rather than a second copy of
    the actor's own measurement.
    """
    vals = []
    for name in fields or ():
        if name == "time_remaining":
            vals.append(float(time_remaining))
        elif name == "facing_error":
            vals.append(float(facing_error))
        elif name == "target_true_pos":
            vals.extend(np.asarray(target_true_pos_body, dtype=np.float64).reshape(3).tolist())
        elif name == "target_true_vel":
            vals.extend(np.asarray(target_true_vel_body, dtype=np.float64).reshape(3).tolist())
        else:  # pragma: no cover - guarded by model_loader validation
            raise ValueError(f"unknown privileged field {name!r}")
    if not vals:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(vals, dtype=np.float64)


class ObsBuilder:
    """Applies per-step sensor noise, history stacking, and block assembly.

    The env calls :meth:`update` once per step/reset with the RAW frame plus
    the raw future and privileged blocks and receives the final observation.
    RNG draws happen here so the Stage-1 noise order with ``hover_env`` is
    preserved regardless of how the frame was assembled by the caller.
    """

    def __init__(
        self,
        obs_cfg: ObsStageConfig,
        history_frames: int,
        history_skip: int,
        rng: np.random.Generator,
        future_samples: int = 0,
        future_skip: int = 1,
        privileged_fields: Sequence[str] = (),
    ):
        self.cfg = obs_cfg
        self.include_target = bool(obs_cfg.include_target)
        self.history_frames = int(history_frames) if self.include_target else 0
        self.history_skip = max(1, int(history_skip)) if self.include_target else 1
        self.rng = rng

        self.future_samples = int(future_samples) if self.include_target else 0
        self.future_skip = max(1, int(future_skip))
        self.privileged_fields = tuple(privileged_fields or ()) if self.include_target else ()

        self.frame_dim = TARGET_FRAME_DIM if self.include_target else STAGE1_FRAME_DIM
        self.window = self.history_frames * self.history_skip + 1
        self.future_rows = self.future_samples if self.include_target else 0
        self.priv_dim = privileged_dim(self.privileged_fields)
        self._hist: Optional[deque] = None
        self.reset()

    # -- properties -----------------------------------------------------------

    @property
    def obs_dim(self) -> int:
        """Width of the vector :meth:`update` returns."""
        return observation_dim(
            include_target=self.include_target,
            history_frames=self.history_frames,
            future_samples=self.future_samples,
            privileged_fields=self.privileged_fields,
        )

    @property
    def privileged_dim(self) -> int:
        """Width of the privileged tail (the critic's extra input)."""
        return self.priv_dim

    @property
    def actor_obs_dim(self) -> int:
        """Width of everything the actor sees (obs minus the privileged tail)."""
        return self.obs_dim - self.priv_dim

    # -- lifecycle -------------------------------------------------------------

    def reset(self) -> None:
        """Clear history (no frames)."""
        self._hist = deque(maxlen=self.window)

    def update(
        self,
        frame: np.ndarray,
        future: Optional[np.ndarray] = None,
        privileged: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Push one raw frame; return the stacked, noised observation.

        ``future`` is the raw ``(n, 7)`` block (see :func:`make_future_block`)
        and ``privileged`` the raw 1-D exact-truth vector; both may be ``None``
        on stages that do not use them.
        """
        noised = self._noise_frame(frame)
        self._hist.append(noised)
        if not self.include_target:
            # single end-of-pipeline cast, mirroring hover_env._get_obs's
            # np.array([...], dtype=np.float32) for bit-level parity
            return np.array(noised, dtype=np.float32)

        parts = [self._stack()]
        if self.future_rows:
            parts.append(self._future(future))
        if self.priv_dim:
            parts.append(self._privileged(privileged))
        return np.concatenate(parts).astype(np.float32)

    # -- internals --------------------------------------------------------------

    def _noise_frame(self, frame: np.ndarray) -> np.ndarray:
        """Sensor noise; draw order rot(6), v(3), omega(3) then target fields.

        All arithmetic stays float64 until the final cast (hover parity).
        Target relative fields are noised only when the target is visible;
        hidden fields stay exactly zero.
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
                frames.append(np.zeros(self.frame_dim, dtype=np.float64))
        return np.concatenate(frames)

    def _future(self, future: Optional[np.ndarray]) -> np.ndarray:
        """Flatten the future block, noising target fields when visible.

        Every row is a strictly-future sample, so all of them take fresh noise
        draws (the current instant is already in the newest history frame).
        """
        rows = self.future_rows
        if future is None:
            return np.zeros(rows * FUTURE_SAMPLE_DIM, dtype=np.float64)
        blk = np.asarray(future, dtype=np.float64).reshape(rows, FUTURE_SAMPLE_DIM).copy()
        s = self.cfg.obs_noise_std
        if s > 0.0 and float(blk[0, FUTURE_SAMPLE_DIM - 1]) > 0.5:
            blk[:, 0:3] += self.rng.normal(0.0, s, (rows, 3))
            blk[:, 3:6] += self.rng.normal(0.0, s, (rows, 3))
        return blk.ravel()

    def _privileged(self, privileged: Optional[np.ndarray]) -> np.ndarray:
        """Privileged tail -- exact truth, never noised (D-12)."""
        if privileged is None:
            return np.zeros(self.priv_dim, dtype=np.float64)
        arr = np.asarray(privileged, dtype=np.float64).ravel()
        if arr.shape[0] != self.priv_dim:
            raise ValueError(
                f"privileged block width {arr.shape[0]} != expected {self.priv_dim}"
            )
        return arr
