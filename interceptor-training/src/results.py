"""results.py -- public result objects returned by ``train_model()``.

Implements D-10, D-21, D-23, D-26, D-32, D-34:

``TrainResult``
    Mapping of ``model_id -> ModelResult`` with attribute access, so
    ``result.ppo_baseline`` and ``result["ppo_baseline"]`` both work (D-34:
    model ids are valid Python identifiers).  Also carries the run-level
    metadata needed to continue a run (name, run dir, seeds, config hash).

``ModelResult``
    One model's outcome: per-stage :class:`StageOutcome` records, the final
    in-memory SB3 model, a status, and a plain-language reason.

``StageOutcome``
    D-32: what happened in one stage (``advance`` / ``capped`` / ``stuck`` /
    ``rollback``), with the step count and the achieved success rate.

Statuses (D-24, D-27, D-31, D-55, D-56)
----------------------------------------
``completed``
    Every requested stage reached its target success rate.
``capped``
    A stage hit its step budget.  With ``on_capped: stop`` the run halts
    here; with ``on_capped: continue`` later stages may still complete, and
    the *final* status stays ``capped`` so a continuation call can deny it.
``stuck``
    Rollback retries exhausted.  Isolated to this model (D-55).
``skipped``
    The requested stage list does not follow the source's last completed
    stage (D-27).
``crashed``
    An exception escaped this model's training loop (D-31).
``denied``
    The source model is capped; continuing is refused (D-56).

Persistence (D-23)
-------------------
``ModelResult.save`` / ``ModelResult.load`` write ``<name>.json`` next to the
model zip.  The zip itself is written by the caller (``ModelResult.final_path``)
so ``load`` returns a descriptor that can re-instantiate the SB3 model via
``ModelResult.load_model()``.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Union

__all__ = [
    "STATUS_COMPLETED",
    "STATUS_CAPPED",
    "STATUS_STUCK",
    "STATUS_SKIPPED",
    "STATUS_CRASHED",
    "STATUS_DENIED",
    "STATUSES",
    "StageOutcome",
    "ModelResult",
    "TrainResult",
    "check_continuation_eligibility",
    "save_result_json",
    "load_result_json",
]

STATUS_COMPLETED = "completed"
STATUS_CAPPED = "capped"
STATUS_STUCK = "stuck"
STATUS_SKIPPED = "skipped"
STATUS_CRASHED = "crashed"
STATUS_DENIED = "denied"

STATUSES = (
    STATUS_COMPLETED,
    STATUS_CAPPED,
    STATUS_STUCK,
    STATUS_SKIPPED,
    STATUS_CRASHED,
    STATUS_DENIED,
)

_SUFFIX = ".result.json"


# ----------------------------------------------------------------------------
# D-32: per-stage outcome
# ----------------------------------------------------------------------------
@dataclass
class StageOutcome:
    """What happened in one curriculum stage (D-32)."""

    stage: int
    result: str
    steps: int = 0
    success_rate: float = 0.0
    budget: int = 0
    retries: int = 0
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": int(self.stage),
            "result": str(self.result),
            "steps": int(self.steps),
            "success_rate": float(self.success_rate),
            "budget": int(self.budget),
            "retries": int(self.retries),
            "note": str(self.note),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "StageOutcome":
        return cls(
            stage=int(d.get("stage", 0)),
            result=str(d.get("result", "")),
            steps=int(d.get("steps", 0)),
            success_rate=float(d.get("success_rate", 0.0)),
            budget=int(d.get("budget", 0)),
            retries=int(d.get("retries", 0)),
            note=str(d.get("note", "")),
        )


# ----------------------------------------------------------------------------
# atomic JSON helpers
# ----------------------------------------------------------------------------
def _atomic_write_json(path: Union[str, Path], payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.stem + "_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return path


def _read_json(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


# ----------------------------------------------------------------------------
# C-2 / D-27 / D-35 / D-56: may this model continue from this source?
# ----------------------------------------------------------------------------
def check_continuation_eligibility(
    model_id: str,
    prior: Optional["ModelResult"],
    stages: Sequence[int],
) -> Optional[tuple]:
    """Decide whether ``model_id`` may run ``stages`` given ``prior``.

    Returns ``None`` when the model is allowed to train, otherwise
    ``(status, reason)``:

    * a source that *capped* a stage is **denied** -- continuing from it would
      train on a half-finished curriculum, so the caller must retrain under a new
      model id (D-56, D-35);
    * a source whose last finished stage does not immediately precede the first
      requested stage is **skipped** -- e.g. finishing stage 3 and asking for
      stage 7 means stage 4-6 never happened (D-27, D-35).

    A missing ``prior`` (fresh training) is always eligible.
    """
    from .training.curriculum import valid_predecessors

    if prior is None:
        return None
    ordered = sorted({int(s) for s in stages})
    if not ordered:
        return None
    if str(prior.status) in (STATUS_CAPPED, STATUS_DENIED):
        return (
            STATUS_DENIED,
            f"Model '{model_id}' has a capped stage; retrain with a new model id",
        )
    last = prior.last_stage
    if last is None or int(last) not in valid_predecessors(ordered[0]):
        return (
            STATUS_SKIPPED,
            f"Model '{model_id}' last completed stage {last} "
            f"does not fit requested stages {ordered}",
        )
    return None


# ----------------------------------------------------------------------------
# D-26 / D-32: per-model result
# ----------------------------------------------------------------------------
class ModelResult:
    """Result for one ``model_id`` (D-26, D-32).

    Attributes
    ----------
    model_id:
        Key inside the model file.  Also the attribute name usable on
        :class:`TrainResult` (D-34).
    algo:
        ``PPO`` / ``SAC`` / ``TD3``.
    stages:
        ``{stage_number: StageOutcome}`` for every stage that was entered.
    final_model:
        The live SB3 model object (saveable).  ``None`` for denied/skipped.
    final_path:
        Filesystem path of the saved final model zip, when written.
    status / reason:
        Terminal status plus a plain-language explanation (D-27, D-31, D-56).
    """

    def __init__(
        self,
        model_id: str,
        *,
        algo: str = "",
        stages: Optional[Mapping[int, StageOutcome]] = None,
        final_model: Any = None,
        final_path: Optional[Union[str, Path]] = None,
        config_hash: str = "",
        status: str = STATUS_COMPLETED,
        reason: str = "",
        run_dir: Optional[Union[str, Path]] = None,
        started: str = "",
        finished: str = "",
        seconds: float = 0.0,
        steps: int = 0,
        last_stage: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_id = str(model_id)
        self.algo = str(algo)
        self.stages: Dict[int, StageOutcome] = {
            int(k): v for k, v in (stages or {}).items()
        }
        self.final_model = final_model
        self.final_path = Path(final_path) if final_path else None
        self.config_hash = str(config_hash)
        self.status = str(status)
        self.reason = str(reason)
        self.run_dir = Path(run_dir) if run_dir else None
        self.started = str(started)
        self.finished = str(finished)
        self.seconds = float(seconds)
        self.steps = int(steps)
        self._last_stage = last_stage
        self.meta: Dict[str, Any] = dict(meta or {})

    # -- accessors ----------------------------------------------------------
    def __getattr__(self, item: str) -> Any:
        # Only reached when normal attribute lookup fails.
        stages = self.__dict__.get("stages") or {}
        if str(item) in stages:
            return stages[str(item)]
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {item!r}"
        )

    def __getitem__(self, key: Union[int, str]) -> StageOutcome:
        k = int(key)
        if k not in self.stages:
            raise KeyError(k)
        return self.stages[k]

    def __contains__(self, key: object) -> bool:
        try:
            return int(key) in self.stages  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False

    def __repr__(self) -> str:
        return (
            f"ModelResult(model_id={self.model_id!r}, status={self.status!r}, "
            f"stages={sorted(self.stages)}, steps={self.steps})"
        )

    @property
    def last_stage(self) -> Optional[int]:
        """Highest stage number entered, or ``None`` if nothing ran."""
        if self._last_stage is not None:
            return int(self._last_stage)
        return max(self.stages) if self.stages else None

    @last_stage.setter
    def last_stage(self, value: Optional[int]) -> None:
        self._last_stage = value

    @property
    def total_steps(self) -> int:
        if self.steps:
            return int(self.steps)
        return int(sum(o.steps for o in self.stages.values()))

    @property
    def stage_outcomes(self) -> List[StageOutcome]:
        return [self.stages[k] for k in sorted(self.stages)]

    @property
    def success_rates(self) -> Dict[int, float]:
        return {k: self.stages[k].success_rate for k in sorted(self.stages)}

    # -- D-23 persistence ---------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "algo": self.algo,
            "status": self.status,
            "reason": self.reason,
            "config_hash": self.config_hash,
            "final_path": str(self.final_path) if self.final_path else None,
            "run_dir": str(self.run_dir) if self.run_dir else None,
            "started": self.started,
            "finished": self.finished,
            "seconds": self.seconds,
            "steps": self.total_steps,
            "last_stage": self.last_stage,
            "stages": {
                str(k): self.stages[k].to_dict() for k in sorted(self.stages)
            },
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ModelResult":
        stages = {
            int(k): StageOutcome.from_dict(v)
            for k, v in (d.get("stages") or {}).items()
        }
        return cls(
            model_id=str(d.get("model_id", "")),
            algo=str(d.get("algo", "")),
            stages=stages,
            final_model=None,
            final_path=d.get("final_path"),
            config_hash=str(d.get("config_hash", "")),
            status=str(d.get("status", STATUS_COMPLETED)),
            reason=str(d.get("reason", "")),
            run_dir=d.get("run_dir"),
            started=str(d.get("started", "")),
            finished=str(d.get("finished", "")),
            seconds=float(d.get("seconds", 0.0)),
            steps=int(d.get("steps", 0) or 0),
            last_stage=d.get("last_stage"),
            meta=dict(d.get("meta") or {}),
        )

    def save(self, path: Union[str, Path]) -> Path:
        """Save the final model plus a sidecar JSON (D-23).

        ``path`` may be a directory (a ``<model_id>.zip`` +
        ``<model_id>.result.json`` pair is written inside it) or a path ending
        in ``.zip``.  Writes the model when ``final_model`` is live, and
        always writes the sidecar.  Returns the zip path.
        """
        p = Path(path)
        if p.is_dir() or not p.suffix:
            p.mkdir(parents=True, exist_ok=True)
            zip_path = p / f"{self.model_id}.zip"
            json_path = p / f"{self.model_id}{_SUFFIX}"
        else:
            zip_path = p if p.suffix == ".zip" else p.with_suffix(".zip")
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            json_path = zip_path.with_suffix("").with_name(zip_path.stem + _SUFFIX)

        if self.final_model is not None:
            self.final_model.save(str(zip_path))
            self.final_path = zip_path
            self._write_training_meta(zip_path)
        payload = self.to_dict()
        payload["final_path"] = str(self.final_path) if self.final_path else None
        _atomic_write_json(json_path, payload)
        return zip_path

    def _write_training_meta(self, zip_path: Path) -> None:
        """Drop the architecture sidecar next to a saved model (C-3 / D-28).

        A continuation reuses the source checkpoint, so the file it reads back
        must describe the architecture that produced it.  Failures here are not
        worth breaking a finished run over.
        """
        meta = dict(self.meta or {})
        if not meta.get("model_def"):
            return
        try:
            from .training.checkpoint_manager import CheckpointManager

            CheckpointManager._atomic_write_json(
                CheckpointManager.training_meta_path(zip_path), meta
            )
        except Exception:  # pragma: no cover - best effort only
            pass

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ModelResult":
        """Load a :class:`ModelResult` descriptor from disk (D-23).

        ``path`` may be the sidecar JSON, the model zip, or a directory
        containing either.  The returned object has ``final_model is None``;
        call :meth:`load_model` to re-instantiate the SB3 model.
        """
        p = Path(path)
        if p.is_dir():
            for candidate in sorted(p.glob(f"*{_SUFFIX}")):
                p = candidate
                break
            else:
                raise FileNotFoundError(f"no {_SUFFIX} descriptor under {path}")
        if p.suffix == ".zip":
            p = p.with_name(p.stem + _SUFFIX)
        data = _read_json(p)
        if data is None:
            raise FileNotFoundError(f"no result descriptor at {path}")
        result = cls.from_dict(data)
        if result.final_path and Path(result.final_path).exists():
            result.meta.setdefault("load_path", str(result.final_path))
        return result

    def load_model(self, env: Any = None, device: str = "auto", **kwargs: Any) -> Any:
        """Re-instantiate the saved SB3 model (``env`` optional)."""
        if self.final_model is not None:
            return self.final_model
        if self.final_path is None or not Path(self.final_path).exists():
            raise FileNotFoundError(
                f"no saved model for model_id={self.model_id!r} (final_path={self.final_path!r})"
            )
        from .training.checkpoint_manager import CheckpointManager  # local import

        self.final_model = CheckpointManager.load_model(
            self.algo, str(self.final_path), env=env, device=device, **kwargs
        )
        return self.final_model


# ----------------------------------------------------------------------------
# D-26: run-level result
# ----------------------------------------------------------------------------
class TrainResult:
    """Result of a ``train_model()`` call, keyed by ``model_id`` (D-26)."""

    def __init__(
        self,
        models: Optional[Mapping[str, ModelResult]] = None,
        *,
        name: str = "",
        run_dir: Optional[Union[str, Path]] = None,
        config_path: Optional[Union[str, Path]] = None,
        model_path: Optional[Union[str, Path]] = None,
        stages: Optional[List[int]] = None,
        config_hash: str = "",
        seed: int = 0,
        started: str = "",
        finished: str = "",
        seconds: float = 0.0,
        source: str = "",
    ) -> None:
        self._models: Dict[str, ModelResult] = dict(models or {})
        self.name = str(name)
        self.run_dir = Path(run_dir) if run_dir else None
        self.config_path = str(config_path) if config_path else None
        self.model_path = str(model_path) if model_path else None
        self.stages: List[int] = [int(s) for s in (stages or [])]
        self.config_hash = str(config_hash)
        self.seed = int(seed)
        self.started = str(started)
        self.finished = str(finished)
        self.seconds = float(seconds)
        self.source = str(source)

    # -- mapping / attribute access (D-34) -----------------------------------
    def __getitem__(self, model_id: str) -> ModelResult:
        try:
            return self._models[str(model_id)]
        except KeyError:
            raise KeyError(
                f"model_id {model_id!r} not in result "
                f"(have: {sorted(self._models)})"
            ) from None

    def __getattr__(self, item: str) -> ModelResult:
        models = self.__dict__.get("_models") or {}
        if str(item) in models:
            return models[str(item)]
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {item!r} "
            f"(have: {sorted(models)})"
        )

    def __contains__(self, item: object) -> bool:
        return str(item) in self._models

    def __iter__(self) -> Iterator[str]:
        return iter(self._models)

    def __len__(self) -> int:
        return len(self._models)

    def __repr__(self) -> str:
        return (
            f"TrainResult(name={self.name!r}, stages={self.stages}, "
            f"models={ {k: v.status for k, v in self._models.items()} })"
        )

    # -- mutation / queries --------------------------------------------------
    @property
    def models(self) -> Dict[str, ModelResult]:
        return self._models

    def get(self, model_id: str, default: Any = None) -> Any:
        return self._models.get(str(model_id), default)

    def set(self, model_id: str, result: ModelResult) -> None:
        self._models[str(model_id)] = result

    def statuses(self) -> Dict[str, str]:
        return {k: v.status for k, v in self._models.items()}

    @property
    def capped_models(self) -> List[str]:
        return [k for k, v in self._models.items() if v.status == STATUS_CAPPED]

    @property
    def completed(self) -> bool:
        """True when every requested model finished without a bad status."""
        if not self._models:
            return False
        return all(
            v.status in (STATUS_COMPLETED, STATUS_CAPPED) for v in self._models.values()
        )

    # -- D-23 persistence ----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "run_dir": str(self.run_dir) if self.run_dir else None,
            "config_path": self.config_path,
            "model_path": self.model_path,
            "stages": [int(s) for s in self.stages],
            "config_hash": self.config_hash,
            "seed": int(self.seed),
            "started": self.started,
            "finished": self.finished,
            "seconds": float(self.seconds),
            "source": self.source,
            "models": {k: v.to_dict() for k, v in self._models.items()},
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TrainResult":
        models = {
            str(k): ModelResult.from_dict(v) for k, v in (d.get("models") or {}).items()
        }
        return cls(
            models,
            name=str(d.get("name", "")),
            run_dir=d.get("run_dir"),
            config_path=d.get("config_path"),
            model_path=d.get("model_path"),
            stages=[int(s) for s in (d.get("stages") or [])],
            config_hash=str(d.get("config_hash", "")),
            seed=int(d.get("seed", 0) or 0),
            started=str(d.get("started", "")),
            finished=str(d.get("finished", "")),
            seconds=float(d.get("seconds", 0.0)),
            source=str(d.get("source", "")),
        )

    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Persist the run descriptor (``run_summary.json``) and each model.

        ``path`` defaults to ``<run_dir>/results/run_summary.json``.  Per-model
        zips + sidecars go to ``<run_dir>/results/<model_id>``.
        """
        base = Path(path) if path is not None else (
            Path(self.run_dir) / "results" / "run_summary.json" if self.run_dir
            else Path.cwd() / "run_summary.json"
        )
        if base.is_dir() or base.suffix == "":
            base.mkdir(parents=True, exist_ok=True)
            base = base / "run_summary.json"
        base.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(base, self.to_dict())

        models_root = base.parent
        for model_id, result in self._models.items():
            if result.final_model is None and result.final_path is None:
                continue
            try:
                result.save(models_root / model_id)
            except Exception:  # pragma: no cover - best effort
                continue
        return base


def save_result_json(path: Union[str, Path], result: Any) -> Path:
    """Write ``result`` (``TrainResult``/``ModelResult``/dict) as JSON."""
    if hasattr(result, "to_dict"):
        payload = result.to_dict()
    else:
        payload = dict(result)
    return _atomic_write_json(path, payload)


def load_result_json(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Read a JSON descriptor written by :func:`save_result_json`."""
    return _read_json(path)
