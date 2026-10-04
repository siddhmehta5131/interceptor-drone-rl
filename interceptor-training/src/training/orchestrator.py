"""orchestrator.py -- the internal multi-model training engine (plan B-5, C-1..C-9).

This module is **not** the public API.  Users call
:func:`src.api.train_model`, which validates the request and then hands over to
:class:`Orchestrator`.  Everything here is an implementation detail:

* the curriculum stages come from the caller, not from a config block (D-21);
* every artifact lives under ``<data_dir>/runs/<name>/`` (D-22);
* the seed may be overridden per call (D-36);
* the return value is a :class:`~src.results.TrainResult` keyed by model id,
  not a summary ``dict`` (D-26);
* a model that fails never takes the other models down with it (D-31, D-55).

Layout, relative to ``run_dir`` (which *is* the ``data_dir`` handed to
:class:`~src.training.checkpoint_manager.CheckpointManager`)::

    checkpoints/<model_id>/stage_<N>/<algo>_<steps>_steps.zip   periodic
    checkpoints/<model_id>/stage_<N>/<algo>_stage_<N>_final.zip stage final
    checkpoints/<model_id>/stage_<N>/monitor/                   Monitor CSVs
    tb_logs/<model_id>/stage_<N>/                               TensorBoard
    results/<model_id>_final.zip                                last model
    results/run_summary.json                                    run descriptor
    curriculum_state/run_state.json                             mid-stage resume

Off-policy algorithms (SAC/TD3) additionally persist their replay buffer as
``<checkpoint>_buffer.pkl`` so a resumed stage does not restart with an empty
buffer (D-47).
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..results import (
    STATUS_CAPPED,
    STATUS_COMPLETED,
    STATUS_CRASHED,
    STATUS_STUCK,
    ModelResult,
    StageOutcome,
    TrainResult,
    check_continuation_eligibility,
)
from ..utils.config_loader import TrainConfig
from ..utils.logger import configure_file_logging, get_logger
from .checkpoint_manager import CheckpointManager
from .curriculum import ADVANCE, CAPPED, ROLLBACK, CurriculumScheduler

__all__ = [
    "Orchestrator",
    "RetryLedger",
    "resolve_device",
    "build_policy_kwargs",
    "obs_dim_for",
]


# ---------------------------------------------------------------------------
# lazy SB3 access (keeps this module importable for py_compile on a bare host)
# ---------------------------------------------------------------------------
_SB3: Dict[str, Any] = {}
_SB3_NOISE: Dict[str, Any] = {}
_CB: Dict[str, Any] = {}

_OFF_POLICY = frozenset({"SAC", "TD3"})


def _algo(name: str):
    """Import and cache one SB3 algorithm class."""
    key = str(name).upper()
    if key not in _SB3:
        try:
            from stable_baselines3 import PPO, SAC, TD3
        except ImportError as exc:  # pragma: no cover - container always has SB3
            raise RuntimeError(
                "training requires torch and stable-baselines3 "
                "(pip install stable-baselines3[extra])"
            ) from exc
        _SB3.update({"PPO": PPO, "SAC": SAC, "TD3": TD3})
    return _SB3[key]


def _action_noise(algo_name: str, n_actions: int, sigma: float):
    """TD3 needs an exploration noise object at construction time."""
    if "NormalActionNoise" not in _SB3_NOISE:
        from stable_baselines3.common.noise import NormalActionNoise

        _SB3_NOISE["NormalActionNoise"] = NormalActionNoise
    cls = _SB3_NOISE["NormalActionNoise"]
    return cls(np.zeros(n_actions), sigma * np.ones(n_actions))


# ---------------------------------------------------------------------------
# small pure helpers (kept at module scope so tests can use them directly)
# ---------------------------------------------------------------------------
def resolve_device(device: str) -> str:
    """``auto`` -> ``cuda`` when torch sees a GPU, else ``cpu``."""
    if str(device) != "auto":
        return str(device)
    try:
        import torch
    except ImportError:  # pragma: no cover
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_policy_kwargs(algo_name: str, net_arch: Mapping[str, Sequence[int]],
                        activation: str) -> Dict[str, Any]:
    """``policy_kwargs`` for a plain SB3 ``MlpPolicy``.

    ``net_arch`` is the model's ``{pi: [...], vf: [...]}`` mapping.  SAC and TD3
    read ``net_arch["qf"]``, so it is mirrored from ``vf`` for them.
    """
    import torch.nn as nn

    arch = {"pi": [int(x) for x in net_arch.get("pi", [])],
            "vf": [int(x) for x in net_arch.get("vf", [])]}
    if str(algo_name).upper() in _OFF_POLICY:
        arch["qf"] = list(arch["vf"])
    act = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}.get(
        str(activation).strip().lower(), nn.ReLU
    )
    return {"net_arch": arch, "activation_fn": act}


def obs_dim_for(stage, model_def, cfg: TrainConfig) -> int:
    """Observation width a model sees on ``stage``.

    Stage 1 has no target, so it stays 14-dim (bit-exact parity with
    ``hover_env``).  Every later stage is
    ``19 * (1 + m) + 7 * n + privileged``.
    """
    from ..envs.obs_builder import observation_dim

    return int(
        observation_dim(
            include_target=bool(stage.obs.include_target),
            history_frames=int(model_def.obs_frames),
            future_samples=int(cfg.observation.future_samples),
            privileged_fields=tuple(model_def.privileged_critic or ()),
        )
    )


# ---------------------------------------------------------------------------
# D-45: rollback retry counts that outlive a single train_model() call
# ---------------------------------------------------------------------------
class RetryLedger:
    """Persistent ``(model_id, stage) -> rollback count`` map.

    Lives at ``<data_dir>/curriculum_state/retries.json`` (shared by every run)
    so a stage that already burned its retries cannot get a fresh budget just
    because the user started a new run (D-45).  The map is reset when the
    ``config_hash`` changes, because the training setup is then a different
    experiment.
    """

    def __init__(self, path: Path, config_hash: str) -> None:
        self.path = Path(path)
        self.config_hash = str(config_hash)
        self._counts: Dict[str, int] = {}
        self._load()

    @staticmethod
    def key(model_id: str, stage_number: int) -> str:
        return f"{model_id}:stage_{int(stage_number)}"

    def _load(self) -> None:
        data = CheckpointManager._atomic_read_json(self.path)
        if not isinstance(data, Mapping):
            return
        if str(data.get("config_hash", "")) != self.config_hash:
            return  # different experiment -> start from a clean slate
        counts = data.get("retries")
        if isinstance(counts, Mapping):
            self._counts = {
                str(k): int(v) for k, v in counts.items() if isinstance(v, (int, float))
            }

    def get(self, model_id: str, stage_number: int) -> int:
        return int(self._counts.get(self.key(model_id, stage_number), 0))

    def bump(self, model_id: str, stage_number: int) -> int:
        k = self.key(model_id, stage_number)
        self._counts[k] = self.get(model_id, stage_number) + 1
        self.save()
        return self._counts[k]

    def reset(self, model_id: str, stage_number: int) -> None:
        self._counts.pop(self.key(model_id, stage_number), None)
        self.save()

    def save(self) -> Path:
        CheckpointManager._atomic_write_json(
            self.path, {"config_hash": self.config_hash, "retries": dict(self._counts)}
        )
        return self.path


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------
class Orchestrator:
    """Run the curriculum for every model bound to ``cfg``.

    Parameters
    ----------
    cfg:
        A :class:`~src.utils.config_loader.TrainConfig` whose models have
        already been bound (the API does that so the ``config_hash`` spans the
        config *and* the model file).
    name:
        Run name; ``run_dir`` holds every artifact (D-22).
    run_dir:
        ``<data_dir>/runs/<name>``.  Defaults to ``cfg.data_dir / "runs" / name``.
    stages:
        Stage numbers to run, in order (D-21).
    source_result:
        The previous run when this is a continuation; used for weight transfer
        and for the skip/deny checks (D-27, D-28, D-56).
    result:
        The (empty) :class:`~src.results.TrainResult` to fill in and return.
    """

    def __init__(
        self,
        cfg: TrainConfig,
        *,
        name: str = "",
        run_dir: Optional[Path] = None,
        stages: Optional[Sequence[int]] = None,
        source_result: Optional[TrainResult] = None,
        result: Optional[TrainResult] = None,
        logger: Any = None,
        vecenv: str = "auto",
        resume: bool = True,
    ) -> None:
        self.cfg = cfg
        self.name = str(name)
        self.run_dir = Path(run_dir) if run_dir else cfg.data_dir / "runs" / self.name
        self.stages: List[int] = [int(s) for s in (stages or cfg.stage_numbers())]
        self.source_result = source_result
        self.result = result if result is not None else TrainResult(
            name=self.name, run_dir=self.run_dir, stages=list(self.stages),
            config_hash=cfg.config_hash, seed=cfg.seed,
        )
        self.log = logger or get_logger("orchestrator")
        self.vecenv = str(vecenv)
        self.resume_enabled = bool(resume)
        self.device = resolve_device(cfg.device)
        self._run_id = time.strftime("%Y%m%d-%H%M%S")
        self.retries = RetryLedger(
            cfg.data_dir / "curriculum_state" / "retries.json", cfg.config_hash
        )

    # -- entry point --------------------------------------------------------

    def run(self) -> TrainResult:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        configure_file_logging(self.run_dir / "results" / "logs")
        self.log.info(
            "run=%s models=%s stages=%s device=%s run_dir=%s",
            self.name or self._run_id, self.cfg.model_ids(), self.stages,
            self.device, self.run_dir,
        )
        for model_id in self.cfg.model_ids():
            prior = self.source_result.get(model_id) if self.source_result else None
            verdict = check_continuation_eligibility(model_id, prior, self.stages)
            if verdict is not None:
                status, reason = verdict
                self.log.warning("skipping %s: %s (%s)", model_id, reason, status)
                self.result.set(
                    model_id,
                    ModelResult(
                        model_id,
                        algo=self._algo_name(model_id),
                        config_hash=self.cfg.config_hash,
                        run_dir=str(self.run_dir),
                        status=status,
                        reason=reason,
                        meta=self._training_meta(model_id),
                    ),
                )
                continue
            try:
                self.result.set(model_id, self._run_model(model_id, prior))
            except Exception as exc:  # C-5 / D-31: isolate the failure
                self.log.error("model %s crashed: %s", model_id, exc, exc_info=True)
                self.result.set(
                    model_id,
                    ModelResult(
                        model_id,
                        algo=self._algo_name(model_id),
                        config_hash=self.cfg.config_hash,
                        run_dir=str(self.run_dir),
                        status=STATUS_CRASHED,
                        reason=f"Training crashed: {exc}",
                        finished=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        meta=self._training_meta(model_id),
                    ),
                )
        return self.result

    # -- per-model ----------------------------------------------------------

    def _algo_name(self, model_id: str) -> str:
        try:
            return str(self.cfg.model(model_id).algo)
        except Exception:  # pragma: no cover
            return ""

    def _training_meta(self, model_id: str) -> Dict[str, Any]:
        """C-3 payload: the architecture a continuation must reproduce."""
        mdef = self.cfg.model(model_id)
        return {
            "model_id": model_id,
            "algo": str(mdef.algo),
            "hyperparameters": dict(mdef.hyperparameters),
            "net_arch": {k: list(v) for k, v in mdef.net_arch.items()},
            "config_hash": self.cfg.config_hash,
            "model_def": mdef.to_dict(),
        }

    def _run_model(self, model_id: str, prior: Optional[ModelResult]) -> ModelResult:
        mdef = self.cfg.model(model_id)
        algo = str(mdef.algo)
        cm = CheckpointManager(self.run_dir, model_id, algo)
        cm.ensure()
        configure_file_logging(cm.log_dir())

        stage_cfgs = [self.cfg.stage(n) for n in self.stages]
        roll = self.cfg.rollback_cfg
        max_attempts = int(roll.get("max_attempts", 2))
        retry_scale = float(roll.get("retry_budget_scale", 0.5))

        mr = ModelResult(
            model_id,
            algo=algo,
            config_hash=self.cfg.config_hash,
            run_dir=str(self.run_dir),
            started=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            meta=self._training_meta(model_id),
        )

        resume_state = self._resume_state(model_id)
        idx = 0
        stage_scale = 1.0
        reentry = False
        last_model = None
        last_path: Optional[Path] = None
        total_steps = 0
        t0 = time.time()

        while idx < len(stage_cfgs):
            stage = stage_cfgs[idx]
            stage_id = stage.id
            budget = max(1, int(stage.max_training_steps * stage_scale))
            sched = self._make_scheduler(stage, budget)
            vec = None
            try:
                vec = self._build_vec_env(stage, mdef, cm, self.cfg.seed + 1000 * idx)
                model, source_note = self._resolve_model(
                    model_id, mdef, stage, vec, cm, idx, stage_cfgs, prior,
                    reentry, resume_state,
                )
                callbacks, cb = self._curriculum_callbacks(sched, cm, stage_id)
                remaining = max(1, budget - int(model.num_timesteps))

                self.log.info(
                    "[%s] %s: budget=%d remaining=%d (%s)",
                    model_id, stage_id, budget, remaining, source_note,
                )
                model.learn(
                    total_timesteps=remaining,
                    reset_num_timesteps=False,
                    callback=callbacks,
                    tb_log_name=f"{algo}_{stage_id}",
                    log_interval=self._log_interval,
                    progress_bar=False,
                )
                result = cb.stage_finished or CAPPED
                steps_here = int(model.num_timesteps)
                rate = float(sched.rate())
                total_steps += steps_here
            finally:
                if vec is not None:
                    vec.close()

            outcome = StageOutcome(
                stage=stage.stage_number,
                result=result,
                steps=steps_here,
                success_rate=rate,
                budget=budget,
                retries=self.retries.get(model_id, stage.stage_number),
            )
            mr.stages[stage.stage_number] = outcome

            if result in (ADVANCE, CAPPED):
                last_path = cm.save_model(model, stage_id, steps_here, final=True)
                cm.save_training_meta(last_path, mr.meta or {})
                self.retries.reset(model_id, stage.stage_number)
                last_model = model
                self._write_run_state(
                    model_id, stage, steps_here, last_path, stage_scale, sched
                )
                resume_state = None
                reentry = False
                stage_scale = 1.0
                if result == CAPPED and self.cfg.on_capped == "stop":
                    # D-24: 'stop' means this model is done here.
                    mr.status = STATUS_CAPPED
                    mr.reason = (
                        f"stage {stage.stage_number} hit its "
                        f"{budget}-step budget and on_capped is 'stop'"
                    )
                    outcome.result = result
                    break
                idx += 1
                continue

            if result == ROLLBACK:
                count = self.retries.bump(model_id, stage.stage_number)
                outcome.retries = count
                if idx == 0 or count > max_attempts:
                    mr.status = STATUS_STUCK
                    mr.reason = (
                        f"stage {stage.stage_number} rolled back {count} time(s) "
                        f"(max_attempts={max_attempts})"
                    )
                    outcome.result = "stuck"
                    break
                idx -= 1
                reentry = True
                stage_scale = retry_scale
                self._write_run_state(
                    model_id, stage, steps_here, last_path, stage_scale, sched
                )
                resume_state = None
                continue

            # An unknown terminal result must not loop forever.
            mr.status = STATUS_STUCK
            mr.reason = f"stage {stage.stage_number}: unexpected result {result!r}"
            outcome.result = "stuck"
            break

        mr.steps = int(total_steps)
        mr.seconds = float(time.time() - t0)
        mr.finished = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if last_model is not None:
            mr.final_model = last_model
            mr.final_path = last_path
            mr.last_stage = max(mr.stages) if mr.stages else None
            config_final = cm.save_config_final(last_model)
            cm.save_training_meta(config_final, mr.meta or {})
            mr.meta = dict(mr.meta or {})
            mr.meta["final_checkpoint"] = str(last_path)
            mr.meta["config_final"] = str(config_final)
        if mr.status == STATUS_COMPLETED and not mr.stages:
            mr.status = STATUS_CRASHED
            mr.reason = "no stage ran"
        self.log.info(
            "[%s] %s after %.1fs (%d steps)", model_id, mr.status, mr.seconds, mr.steps
        )
        return mr

    # -- curriculum plumbing ------------------------------------------------

    @property
    def _log_interval(self) -> int:
        try:
            return max(1, int(self.cfg.global_cfg.get("log_interval", 1000)))
        except Exception:  # pragma: no cover
            return 1000

    def _curriculum_callbacks(self, sched: CurriculumScheduler,
                              cm: CheckpointManager, stage_id: str):
        """Build the SB3 callback list for one stage (C-1 drives ``sched``)."""
        cls = _cb_classes()
        cb = cls["CurriculumCallback"](sched)
        callbacks = cls["CallbackList"]([
            cls["CurriculumCheckpointCallback"](
                cm, stage_id, self.cfg.checkpoint_interval_steps
            ),
            cls["RewardComponentCallback"](self._log_interval),
            cls["MetricsCallback"](self._log_interval),
            cb,
        ])
        return callbacks, cb

    def _make_scheduler(self, stage, budget: int) -> CurriculumScheduler:
        roll = self.cfg.rollback_cfg
        threshold = stage.rollback_threshold
        if threshold is None:
            threshold = float(stage.success_rate) * float(roll.get("threshold_scale", 0.5))
        return CurriculumScheduler(
            stage,
            budget=budget,
            rollback_threshold=float(threshold),
            rollback_min_steps=int(budget * float(roll.get("min_steps_scale", 0.25))),
        )

    def _build_vec_env(self, stage, mdef, cm: CheckpointManager, base_seed: int):
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

        from ..envs.base_env import make_env_factory

        obs = self.cfg.observation
        monitor_dir = str(cm.monitor_dir(stage.id))

        def _one(rank: int):
            return make_env_factory(
                stage,
                history_frames=int(mdef.obs_frames),
                history_skip=int(mdef.obs_skip),
                target_alt=self.cfg.target_alt_range,
                future_samples=int(obs.future_samples),
                future_skip=int(obs.future_skip),
                predictor=str(obs.predictor),
                future_source=str(obs.future_source),
                privileged_fields=tuple(mdef.privileged_critic or ()),
                seed=int(base_seed) + rank,
                monitor_dir=monitor_dir,
            )

        n = max(1, int(self.cfg.n_parallel_envs))
        factories = [_one(r) for r in range(n)]
        if self.vecenv in ("auto", "subproc") and n > 1:
            return SubprocVecEnv(factories)
        return DummyVecEnv(factories)

    def _create_model(self, mdef, stage, vec, cm: CheckpointManager):
        from .asymmetric_policy import policy_kwargs_for

        algo = str(mdef.algo)
        obs_dim = obs_dim_for(stage, mdef, self.cfg)
        policy_cls, policy_kwargs = policy_kwargs_for(
            mdef,
            obs_dim,
            policy_kwargs=build_policy_kwargs(algo, mdef.net_arch, mdef.activation),
            privileged_fields=tuple(mdef.privileged_critic or ()),
        )
        kwargs: Dict[str, Any] = {
            "env": vec,
            "policy": policy_cls,
            "policy_kwargs": policy_kwargs,
            "seed": int(self.cfg.seed) + int(stage.stage_number),
            "device": self.device,
        }
        if self.cfg.tensorboard:
            kwargs["tensorboard_log"] = str(cm.tb_dir(stage.id))
        kwargs.update(dict(mdef.hyperparameters))
        if algo == "TD3" and "action_noise" not in kwargs:
            kwargs["action_noise"] = _action_noise(
                algo, int(vec.action_space.shape[0]),
                float(kwargs.get("action_noise_sigma", 0.1)),
            )
        return _algo(algo)(**kwargs)

    def _load_model(self, mdef, path: Path, vec):
        model = CheckpointManager.load_model(
            str(mdef.algo), str(path), env=vec, device=self.device
        )
        if str(mdef.algo).upper() in _OFF_POLICY:
            CheckpointManager.load_replay_buffer(model, Path(path))
        return model

    # -- model selection ----------------------------------------------------

    def _resolve_model(
        self,
        model_id: str,
        mdef,
        stage,
        vec,
        cm: CheckpointManager,
        idx: int,
        stage_cfgs: Sequence[Any],
        prior: Optional[ModelResult],
        reentry: bool,
        resume_state: Optional[Mapping[str, Any]],
    ) -> Tuple[Any, str]:
        """Pick the weights this stage starts from.

        Order: mid-stage resume checkpoint, own final (rollback re-entry),
        cross-stage weight transfer, fresh.  Returns ``(model, note)``.
        """
        stage_id = stage.id

        # 1) mid-stage resume (C-7 / D-46)
        if resume_state:
            ckpt = resume_state.get("last_checkpoint_path")
            if ckpt and str(resume_state.get("stage_num")) == str(stage.stage_number) \
                    and Path(str(ckpt)).exists():
                model = self._load_model(mdef, Path(str(ckpt)), vec)
                model.num_timesteps = int(resume_state.get("stage_steps") or 0)
                self.log.info(
                    "[%s] %s: resuming from %s at %d steps",
                    model_id, stage_id, ckpt, model.num_timesteps,
                )
                return model, f"resumed from {Path(str(ckpt)).name}"

        # 2) rollback re-entry: restart the same stage from its own final
        if reentry:
            own = cm.latest_final(stage_id)
            if own is not None:
                model = self._load_model(mdef, own, vec)
                model.num_timesteps = 0
                return model, "rollback re-entry"

        # 3) cross-stage transfer (C-4 / D-33)
        donor = self._donor_checkpoint(idx, stage_cfgs, prior, cm)
        if donor is not None:
            old = self._load_model(mdef, donor, None)
            model = self._create_model(mdef, stage, vec, cm)
            note = self._transfer(old, model, donor)
            del old
            return model, note

        return self._create_model(mdef, stage, vec, cm), "fresh"

    def _donor_checkpoint(
        self,
        idx: int,
        stage_cfgs: Sequence[Any],
        prior: Optional[ModelResult],
        cm: CheckpointManager,
    ) -> Optional[Path]:
        """Where the incoming weights come from, or ``None`` for a fresh model."""
        if idx > 0:
            prev = stage_cfgs[idx - 1]
            own = cm.latest_final(prev.id)
            if own is not None:
                return own
        if prior is not None and prior.final_path:
            p = Path(str(prior.final_path))
            if p.exists():
                return p
        return None

    def _transfer(self, old_model, new_model, donor: Path) -> str:
        """Copy what matches, random-init what is new (D-33)."""
        from .weight_transfer import stage1_prefix_for, transfer_model, summarise

        new_dim = int(new_model.observation_space.shape[0])
        old_dim = int(old_model.observation_space.shape[0])
        if new_dim == old_dim:
            new_model.set_parameters(old_model.get_parameters(), exact_match=True)
            return f"weights from {donor.name} ({new_dim} dims)"
        prefix = stage1_prefix_for(old_dim, new_dim)
        report = transfer_model(
            old_model, new_model, shared_prefix=prefix, obs_width=new_dim,
            seed=int(self.cfg.seed),
        )
        return (
            f"weights from {donor.name} ({old_dim}->{new_dim}, prefix={prefix}): "
            f"{summarise(report)}"
        )

    # -- resume / run state -------------------------------------------------

    def _resume_state(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Load this run's ``run_state.json`` when it matches (C-7 / D-46)."""
        if not self.resume_enabled:
            return None
        state = CheckpointManager._atomic_read_json(
            CheckpointManager.run_state_path(self.run_dir)
        )
        if not isinstance(state, Mapping):
            return None
        if str(state.get("config_hash", "")) != self.cfg.config_hash:
            self.log.info("ignoring resume state: config_hash changed")
            return None
        if str(state.get("name", self.name)) != self.name:
            return None
        if str(state.get("model_id", model_id)) != model_id:
            return None
        if not state.get("last_checkpoint_path"):
            return None
        return dict(state)

    def _write_run_state(
        self,
        model_id: str,
        stage,
        steps: int,
        checkpoint: Optional[Path],
        stage_scale: float,
        sched: Optional[CurriculumScheduler],
    ) -> None:
        records = []
        if sched is not None:
            try:
                records = list(sched.state_dict().get("records", []))
            except Exception:  # pragma: no cover
                records = []
        CheckpointManager._atomic_write_json(
            CheckpointManager.run_state_path(self.run_dir),
            {
                "run_id": self._run_id,
                "name": self.name,
                "model_id": model_id,
                "algo": str(self.cfg.model(model_id).algo),
                "config_file": str(self.cfg.source),
                "config_hash": self.cfg.config_hash,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "stages": list(self.stages),
                "stage_num": int(stage.stage_number),
                "stage_steps": int(steps),
                "stage_scale": float(stage_scale),
                "last_checkpoint_path": str(checkpoint) if checkpoint else "",
                "episode_buffer": records,
                "retries": {
                    f"{model_id}:stage_{n}": self.retries.get(model_id, n)
                    for n in self.stages
                },
            },
        )


def _cb_classes():
    """Import SB3's ``CallbackList`` and this project's callbacks on demand."""
    if not _CB:
        from stable_baselines3.common.callbacks import CallbackList

        from .callbacks import (
            CurriculumCallback,
            CurriculumCheckpointCallback,
            MetricsCallback,
            RewardComponentCallback,
        )

        _CB.update({
            "CallbackList": CallbackList,
            "CurriculumCallback": CurriculumCallback,
            "CurriculumCheckpointCallback": CurriculumCheckpointCallback,
            "MetricsCallback": MetricsCallback,
            "RewardComponentCallback": RewardComponentCallback,
        })
    return _CB
