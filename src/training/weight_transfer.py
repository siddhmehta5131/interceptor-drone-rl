"""weight_transfer.py -- carry a policy's weights across an obs-size change.

D-33: when the curriculum moves from Stage 1 (14-dim observation, no target)
to a target stage (19-dim frames + history + future + privileged block) the
policy network has to grow.  Throwing the previous weights away throws away a
learned hover controller; loading them blindly is impossible because the
first layer's input width changed.

The transfer rule implemented here:

* Every tensor whose shape is unchanged is copied verbatim (all hidden
  layers, all biases, the value head, the log-std parameters, the action head
  when the action space did not change).
* Every first-layer / observation-touching weight whose **output** width is
  unchanged but whose **input** width grew is copied for the leading columns
  and the remaining columns are initialised with small random noise
  (``scale``), so the new network starts as a perturbation of the old one
  rather than a fresh draw.
* Anything whose output width also changed (e.g. a different ``net_arch``) is
  reported as ``skipped`` instead of being silently mangled.

Only the first ``shared_prefix`` observation columns are semantically shared
between Stage 1 and the target stages:

===============  ==========================================
observation idx  meaning
===============  ==========================================
``0:6``          rotation matrix (first two columns)
``6:9``          world velocity
``9:12``         body angular rate
``12``           Stage 1: altitude error / Stage 2+: rel_pos[0]
``13``           Stage 1: vertical velocity / Stage 2+: rel_pos[1]
===============  ==========================================

so ``SHARED_STAGE1_PREFIX = 12``.  Columns from 12 onward are re-initialised.

The planning logic is pure ``dict``/``shape`` arithmetic in
:func:`plan_transfer`, which keeps this module importable (and testable)
without torch; only :func:`transfer_weights` needs torch, and it is imported
lazily.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

__all__ = [
    "SHARED_STAGE1_PREFIX",
    "plan_transfer",
    "transfer_weights",
    "transfer_model",
    "transfer_state_dicts",
    "stage1_prefix_for",
    "summarise",
]

# rot6 + v3 + omega3 -- the only observation columns whose meaning is
# identical in Stage 1 and in the target stages (D-33).
SHARED_STAGE1_PREFIX = 12

# Width of the Stage-1 observation (rot6 + v3 + omega3 + alt_err + v_z).
STAGE1_OBS_DIM = 14


def _shape(value: Any) -> Tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return ()
    return tuple(int(s) for s in shape)


def _ndim(value: Any) -> int:
    ndim = getattr(value, "ndim", None)
    if ndim is not None:
        return int(ndim)
    dim = getattr(value, "dim", None)
    if callable(dim):
        return int(dim())
    return len(_shape(value))


def _keep_columns(
    old_in: int,
    new_in: int,
    shared_prefix: Optional[int],
    obs_width: Optional[int],
) -> int:
    """How many leading input columns of a grown tensor to copy.

    ``shared_prefix is None``  -> copy as much as fits (``min(old, new)``).

    ``shared_prefix`` given   -> only the *observation* block is truncated.
    The observation block occupies the **trailing** ``obs_width`` columns of
    the old tensor, so everything before it (the MLP latent activations of
    ``action_net``) is copied in full:

    ``obs_start = old_in - obs_width``  and  ``keep = obs_start + shared_prefix``

    For a first-layer extractor weight ``old_in == obs_width`` so
    ``obs_start == 0`` and this collapses to ``keep == shared_prefix``.
    """
    k = min(old_in, new_in)
    if shared_prefix is None or obs_width is None:
        return k
    obs_start = max(0, old_in - int(obs_width))
    return min(k, obs_start + max(0, int(shared_prefix)))


def plan_transfer(
    old_state: Mapping[str, Any],
    new_state: Mapping[str, Any],
    *,
    shared_prefix: Optional[int] = None,
    obs_width: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Describe how ``old_state`` maps onto ``new_state``.

    Returns a mapping ``{param_name: {...}}`` where ``action`` is one of
    ``"copy"`` (identical shape), ``"pad"`` (copy the leading ``keep`` input
    columns, small-random init the rest), ``"init"`` (only in ``new_state``)
    or ``"skip"`` (shape change we refuse to bridge).

    ``shared_prefix``  caps how many *observation* columns are carried over
    (:data:`SHARED_STAGE1_PREFIX` when crossing the Stage 1 -> Stage 2
    boundary, because columns 12+ change meaning).  ``obs_width`` is the old
    observation width; together they keep the non-observation prefix of
    ``action_net`` intact.  Both may be ``None`` ("copy as much as fits").
    """
    plan: Dict[str, Dict[str, Any]] = {}
    for name, new_val in new_state.items():
        new_shape = _shape(new_val)
        if name not in old_state:
            plan[name] = {"action": "init", "shape": new_shape}
            continue
        old_shape = _shape(old_state[name])
        if old_shape == new_shape:
            plan[name] = {"action": "copy", "shape": new_shape}
            continue
        # [rows, cols] where rows == outputs; either the cols or the rows grow
        if (
            len(new_shape) == 2
            and len(old_shape) == 2
            and (old_shape[0] == new_shape[0] or old_shape[1] == new_shape[1])
        ):
            if old_shape[0] == new_shape[0]:
                keep = _keep_columns(old_shape[1], new_shape[1], shared_prefix, obs_width)
                axis = "cols"
            else:
                keep = _keep_columns(old_shape[0], new_shape[0], shared_prefix, obs_width)
                axis = "rows"
            plan[name] = {
                "action": "pad" if keep > 0 else "init",
                "shape": new_shape,
                "old_shape": old_shape,
                "keep": keep,
                "axis": axis,
            }
            continue
        # 1-D -- biases / log-std are never observation-dependent: copy fully
        if len(new_shape) == 1 and len(old_shape) == 1:
            keep = min(old_shape[0], new_shape[0])
            plan[name] = {
                "action": "pad" if keep > 0 else "init",
                "shape": new_shape,
                "old_shape": old_shape,
                "keep": keep,
                "axis": "flat",
            }
            continue
        plan[name] = {
            "action": "skip",
            "shape": new_shape,
            "old_shape": old_shape,
            "reason": "output and input widths both changed",
        }
    if verbose:
        for name, info in plan.items():
            print(f"  {name}: {info}")
    return plan


def transfer_state_dicts(
    old_state: Mapping[str, Any],
    new_state: Dict[str, Any],
    *,
    shared_prefix: Optional[int] = None,
    obs_width: Optional[int] = None,
    scale: float = 1e-3,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a new ``state_dict`` from ``old_state`` matching ``new_state``.

    Torch-free core (works on torch tensors and numpy arrays).  Returns a
    report dict::

        {"copied": [...], "padded": {name: {"keep": k, "axis": ...}},
         "initialised": [...], "skipped": [...], "plan": {...}}
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    plan = plan_transfer(
        old_state, new_state, shared_prefix=shared_prefix, obs_width=obs_width
    )

    copied: list = []
    initialised: list = []
    skipped: list = []
    padded: Dict[str, Dict[str, Any]] = {}

    for name, info in plan.items():
        action = info["action"]
        if action == "copy":
            _copy_into(old_state[name], new_state[name])
            copied.append(name)
        elif action == "pad":
            keep = int(info["keep"])
            axis = info.get("axis", "cols")
            old_v = old_state[name]
            new_v = new_state[name]
            if axis == "cols" and _ndim(new_v) == 2:
                _copy_into(old_v[:, :keep], new_v[:, :keep])
                if new_v.shape[1] > keep:
                    _fill(new_v[:, keep:], rng, scale)
            elif axis == "rows" and _ndim(new_v) == 2:
                _copy_into(old_v[:keep, :], new_v[:keep, :])
                if new_v.shape[0] > keep:
                    _fill(new_v[keep:, :], rng, scale)
            else:
                _copy_into(old_v[:keep], new_v[:keep])
                if new_v.shape[0] > keep:
                    _fill(new_v[keep:], rng, scale)
            padded[name] = {"keep": keep, "axis": axis, "shape": info["shape"]}
        elif action == "init":
            if info["shape"]:
                _fill(new_state[name], rng, scale)
            initialised.append(name)
        else:
            skipped.append(name)

    return {"copied": copied, "padded": padded, "initialised": initialised,
            "skipped": skipped, "plan": plan}


def _is_torch(t: Any) -> bool:
    return type(t).__module__.startswith("torch")


def _copy_into(src: Any, dst: Any) -> None:
    """Copy ``src`` into ``dst`` for torch tensors and numpy arrays alike."""
    if _is_torch(dst):
        import torch as th

        with th.no_grad():
            dst.copy_(src)
        return
    dst[...] = src


def _fill(dst: Any, rng: Any, scale: float) -> None:
    """Uniform small random init for torch tensors and numpy arrays alike."""
    import numpy as np

    shape = tuple(int(s) for s in dst.shape)
    if _is_torch(dst):
        import torch as th

        with th.no_grad():
            dst.uniform_(-float(scale), float(scale))
        return
    dst[...] = rng.uniform(-float(scale), float(scale), size=shape).astype(
        np.asarray(dst).dtype, copy=False
    )


def transfer_weights(
    target_policy: Any,
    source_policy: Any,
    *,
    shared_prefix: Optional[int] = None,
    obs_width: Optional[int] = None,
    scale: float = 1e-3,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Copy ``source_policy``'s weights into ``target_policy`` (D-33).

    Both arguments are SB3 policy objects (anything exposing
    ``state_dict``/``load_state_dict``).  Returns the report from
    :func:`transfer_state_dicts`.
    """
    old_state = source_policy.state_dict()
    new_state = dict(target_policy.state_dict())
    report = transfer_state_dicts(
        old_state,
        new_state,
        shared_prefix=shared_prefix,
        obs_width=obs_width,
        scale=scale,
        seed=seed,
    )
    target_policy.load_state_dict(new_state, strict=False)
    return report


def transfer_model(
    old_model: Any,
    new_model: Any,
    *,
    shared_prefix: Optional[int] = None,
    obs_width: Optional[int] = None,
    scale: float = 1e-3,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """:func:`transfer_weights` for SB3 model objects (``.policy``)."""
    return transfer_weights(
        new_model.policy,
        old_model.policy,
        shared_prefix=shared_prefix,
        obs_width=obs_width,
        scale=scale,
        seed=seed,
    )


def th_tensor(value: Any) -> Any:  # pragma: no cover - tiny helper
    import torch as th

    return th.as_tensor(value)


def stage1_prefix_for(old_obs_dim: int, new_obs_dim: int) -> Optional[int]:
    """Shared-prefix advice for a Stage 1 -> target-stage observation jump.

    Returns :data:`SHARED_STAGE1_PREFIX` when the old observation was the
    14-dim Stage-1 one and the new one is wider (so columns 12+ change
    meaning), otherwise ``None`` ("copy as much as fits").
    """
    if int(new_obs_dim) > int(old_obs_dim) and int(old_obs_dim) <= STAGE1_OBS_DIM:
        return SHARED_STAGE1_PREFIX
    return None


def summarise(report: Mapping[str, Any]) -> str:
    """One-line human summary of a transfer report."""
    return (
        f"copied {len(report.get('copied', []))}, "
        f"padded {len(report.get('padded', {}))}, "
        f"init {len(report.get('initialised', []))}, "
        f"skipped {len(report.get('skipped', []))}"
    )
