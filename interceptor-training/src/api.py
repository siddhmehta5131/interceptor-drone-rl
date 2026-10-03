"""api.py -- the single public entry point for training (plan B-1).

``train_model(source, stages, name, ...) -> TrainResult``

Everything the user needs lives behind this one function.  The orchestrator
(:class:`src.training.orchestrator.Orchestrator`) is an internal engine; it is
never called directly from user code.

Two kinds of source are accepted (B-1)
--------------------------------------
1. A **path to a model file** (``configs/model.yaml``) -- a fresh run.  The
   model definitions come from the YAML and the training settings from
   ``config.yaml``.
2. **A previous :class:`~src.results.TrainResult`** -- a continuation run.
   The architecture is recovered from the ``_training_meta.json`` written next
   to the saved checkpoint (C-3), the weights are loaded from that checkpoint,
   and the training settings are reused verbatim: if the ``config.yaml`` on
   hand does not reproduce the source's ``config_hash`` the call fails loudly
   rather than silently changing the architecture or the observation layout
   (D-28).

Per-model outcomes (D-26, D-27, D-31, D-55, D-56) are recorded as a
:class:`~src.results.ModelResult` with one of the statuses in
:data:`src.results.STATUSES`; a model that fails never takes the other models
down with it.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from .results import (
    ModelResult,
    TrainResult,
    check_continuation_eligibility,
)
from .utils.config_loader import ConfigError, TrainConfig, load_config
from .utils.model_loader import ModelDef, load_models, model_def_from_dict

__all__ = ["train_model", "DEFAULT_CONFIG"]

#: Sentinel so ``config=None``-style detection is not needed: when the caller
#: leaves ``config`` at its default *and* the source is a continuation, the
#: config path recorded by the source run is reused.
DEFAULT_CONFIG = "configs/config.yaml"

_MAX_STAGE = 8


# ---------------------------------------------------------------------------
# source resolution
# ---------------------------------------------------------------------------
def _resolve_config_path(source: Any, config: Any) -> Optional[Path]:
    """Pick the config file for this call.

    An explicit ``config=`` always wins.  When it is left at the default and
    the source is a continuation, reuse the config the source run recorded.
    Returns ``None`` when the caller passed a ready-made
    :class:`~src.utils.config_loader.TrainConfig` (the smoke CLI builds one so
    it can shrink the per-stage budgets without writing a temporary file).
    """
    if isinstance(config, TrainConfig):
        return None
    if isinstance(config, str) and config == DEFAULT_CONFIG:
        recorded = getattr(source, "config_path", None)
        if isinstance(source, TrainResult) and recorded:
            return Path(recorded)
    return Path(str(config))


def _models_from_result(
    source: TrainResult, requested: Sequence[str]
) -> Dict[str, ModelDef]:
    """Rebuild :class:`ModelDef` objects from a previous run (D-28).

    Prefers the ``model_def`` block saved in ``ModelResult.meta`` (which is the
    same payload as ``_training_meta.json``).  Falls back to re-reading the
    source run's ``model.yaml`` when the meta block is absent, so results
    produced by an older version still continue.
    """
    models: Dict[str, ModelDef] = {}
    for model_id in requested:
        prior = source.get(model_id)
        if prior is None:
            raise ConfigError(
                f"model_id {model_id!r} is not present in the source TrainResult "
                f"(have: {sorted(source.models)})"
            )
        saved = (prior.meta or {}).get("model_def")
        if isinstance(saved, Mapping) and saved:
            models[model_id] = model_def_from_dict(
                model_id,
                saved,
                source_path=f"{source.name or '<TrainResult>'}:{model_id}",
            )
            continue
        model_path = source.model_path
        if model_path and Path(model_path).exists():
            fallback = load_models(model_path)
            if model_id in fallback:
                models[model_id] = fallback[model_id]
                continue
        raise ConfigError(
            f"cannot continue {model_id!r}: the source TrainResult has no saved "
            f"model definition (meta['model_def'] missing) and its model file "
            f"{model_path or '<none>'!r} is unavailable"
        )
    return models


def _requested_model_ids(
    models: Mapping[str, ModelDef], model_ids: Optional[Sequence[str]]
) -> List[str]:
    """The subset of models to train; unknown ids are ignored (B-1)."""
    if not model_ids:
        return list(models.keys())
    keep = [str(m) for m in model_ids if str(m) in models]
    return keep


def _check_eligibility(
    model_id: str,
    prior: Optional[ModelResult],
    stages: Sequence[int],
) -> Optional[tuple]:
    """C-2: decide whether a source model may train ``stages``.

    Returns ``None`` when the model is allowed to run, otherwise
    ``(status, reason)`` with ``status`` in ``{'denied', 'skipped'}``.

    * A capped source is refused outright (D-56): it ran out of budget rather
      than reaching its success target, so continuing from it would train a
      policy that never learned the stage.
    * A source whose last completed stage is not a valid predecessor of the
      first requested stage would silently jump the curriculum, so it is
      skipped with an explanation (D-27).

    The rule itself lives in :func:`src.results.check_continuation_eligibility`
    so the API and the engine can never disagree about it.
    """
    return check_continuation_eligibility(model_id, prior, stages)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def train_model(
    source: Union[str, Path, TrainResult],
    stages: Sequence[int],
    name: str,
    *,
    config: Union[str, Path, TrainConfig] = DEFAULT_CONFIG,
    model_ids: Optional[Sequence[str]] = None,
    seed: Optional[int] = None,
    vecenv: str = "auto",
    resume: bool = True,
) -> TrainResult:
    """Train one or more models through a curriculum and return the results.

    Parameters
    ----------
    source:
        Path to a ``model.yaml`` (fresh run) or a previous
        :class:`~src.results.TrainResult` (continuation).
    stages:
        Curriculum stage numbers to train, in order, e.g. ``[1, 2, 3, 4]``.
        A continuation may repeat the source's last stage to give it more
        budget.
    name:
        Run name.  Everything this call writes lives under
        ``<data_dir>/runs/<name>/`` (D-22): checkpoints, TensorBoard logs,
        resume state, results and logs.
    config:
        Path to ``config.yaml`` (or an already-built
        :class:`~src.utils.config_loader.TrainConfig`).  Ignored for a
        continuation whose source recorded one, unless given explicitly.
    model_ids:
        Train only these models.  Ids that are not in the source are skipped
        silently; the remaining models train normally.
    seed:
        Overrides ``global.seed`` for this run (D-36).  ``None`` keeps it.
    vecenv:
        ``"auto"`` uses ``SubprocVecEnv`` when more than one environment is
        configured, otherwise ``DummyVecEnv``.  ``"dummy"`` forces in-process
        environments, which is handy on Windows or when debugging.
    resume:
        When ``False``, ``run_state.json`` is ignored and every stage starts
        from scratch instead of continuing the last checkpoint.

    Returns
    -------
    TrainResult
        Keyed by model id; ``result.<model_id>.status`` is one of
        :data:`src.results.STATUSES` and ``.reason`` explains anything that is
        not ``completed``.
    """
    # -- validate the call ---------------------------------------------------
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("train_model: 'name' must be a non-empty string")
    name = name.strip()

    stage_list = [int(s) for s in (stages or [])]
    if not stage_list:
        raise ConfigError("train_model: 'stages' must contain at least one stage")
    for s in stage_list:
        if not 1 <= s <= _MAX_STAGE:
            raise ConfigError(
                f"train_model: stage {s} is out of range (1..{_MAX_STAGE})"
            )
    if stage_list != sorted(stage_list):
        stage_list = sorted(stage_list)

    is_continuation = isinstance(source, TrainResult)

    # -- resolve model definitions ------------------------------------------
    if is_continuation:
        source_result: Optional[TrainResult] = source
        requested = _requested_model_ids(
            {str(k): None for k in source.models}, model_ids
        )
        if not requested:
            raise ConfigError(
                "train_model: none of the requested model_ids are present in the "
                f"source TrainResult (have: {sorted(source.models)})"
            )
        models = _models_from_result(source, requested)
        model_source = f"{source.name or '<TrainResult>'}"
    else:
        source_result = None
        source_path = Path(str(source))
        all_models = load_models(source_path)
        requested = _requested_model_ids(all_models, model_ids)
        if not requested:
            raise ConfigError(
                f"train_model: none of the requested model_ids are present in "
                f"{source_path} (have: {sorted(all_models)})"
            )
        models = {mid: all_models[mid] for mid in requested}
        model_source = str(source_path)

    # -- load + bind the training settings ----------------------------------
    cfg_path = _resolve_config_path(source, config)
    if cfg_path is None:
        cfg = config
        cfg.bind_models(models, model_source=model_source)
        cfg_path = Path(str(cfg.source_config_path))
    else:
        cfg = load_config(cfg_path)
        cfg.bind_models(models, model_source=model_source)
    if seed is not None:
        cfg.global_cfg["seed"] = int(seed)
        cfg.recompute_hash()

    # -- continuation guard (C-3 / D-28) ------------------------------------
    if is_continuation and source_result is not None:
        _verify_continuation(source_result, requested, cfg, cfg_path)

    # -- build the (empty) result and hand it to the engine -----------------
    run_dir = cfg.data_dir / "runs" / name
    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result = TrainResult(
        {},
        name=name,
        run_dir=run_dir,
        config_path=str(cfg_path),
        model_path=model_source,
        stages=stage_list,
        config_hash=cfg.config_hash,
        seed=int(cfg.seed),
        started=started,
        source=model_source,
    )

    from .training.orchestrator import Orchestrator  # local: pulls in SB3

    engine = Orchestrator(
        cfg,
        name=name,
        run_dir=run_dir,
        stages=stage_list,
        source_result=source_result,
        result=result,
        vecenv=str(vecenv),
        resume=bool(resume),
    )
    t0 = time.time()
    engine.run()
    result.seconds = time.time() - t0
    result.finished = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result.save()
    return result


def _verify_continuation(
    source: TrainResult,
    model_ids: Sequence[str],
    cfg: TrainConfig,
    cfg_path: Path,
) -> None:
    """Fail loudly when a continuation would change the setup (D-28).

    "No overrides allowed" means: the architecture (algo, net_arch,
    hyperparameters, observation layout) recovered from the source must still
    be exactly what ``cfg_path`` describes.  The cheap, complete check is the
    ``config_hash`` -- it spans the config file *and* the model definitions.
    """
    expected = {
        str(k): v
        for k, v in (source.models or {}).items()
        if str(k) in set(model_ids) and getattr(v, "config_hash", "")
    }
    source_hash = next(iter(expected.values())) if expected else ""
    if source_hash and source_hash != cfg.config_hash:
        raise ConfigError(
            f"cannot continue from {source.name or '<TrainResult>'}: the settings in "
            f"{cfg_path} produce config_hash {cfg.config_hash[:12]}... but the source "
            f"run was trained with {source_hash[:12]}....\n"
            f"Continuation runs must not override the architecture or the observation "
            f"layout -- reuse the original config.yaml, or start a fresh run with a "
            f"new model id."
        )

    # Stage-level guard: the saved metadata must agree with the live config.
    for model_id in model_ids:
        prior = source.get(model_id)
        if prior is None:
            continue
        saved = (prior.meta or {}).get("config_hash")
        if saved and str(saved) != cfg.config_hash:
            raise ConfigError(
                f"cannot continue {model_id!r}: saved config_hash {str(saved)[:12]}... "
                f"does not match {cfg.config_hash[:12]}... from {cfg_path}"
            )
