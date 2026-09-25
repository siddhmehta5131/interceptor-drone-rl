"""orchestrator.py -- plan §7 per-config curriculum training loop.

Imports of stable_baselines3/torch are module-level **inside functions** so
the module still imports without them (harmless locally; train.py runs it
inside the container).

Flow per config (plan §7):

1. for each stage in the config's stage list:
   a. build a :class:`CurriculumScheduler` (budgeted)
   b. build a vectorised environment (SubprocVecEnv, ``n_parallel_envs``)
   c. build/load the SB3 model:
      - resume                      -> load mid-stage periodic checkpoint
      - rollback re-entry           -> load the stage's own final checkpoint
      - obs space unchanged vs prev -> fresh model + ``set_parameters``
        transfer from the previous stage's final checkpoint
      - otherwise                   -> fresh model
   d. ``model.learn(remaining_budget, reset_num_timesteps=False)`` with
      Reward/Metrics/Checkpoint/Curriculum callbacks (the last returns
      False from ``on_step`` to stop learning on advance/cap/rollback)
2. react to the terminal result:
      advance  -> save stage final checkpoint, advance to next stage
      capped   -> log "capped -- did not converge", save best, advance
      rollback -> retries[stage]+=1; either re-enter previous stage on a
                  reduced budget (max_attempts limit) or mark the stage
                  stuck and move to the next config
3. save the config's final model to ``results/<config>_final.zip`` and
   append the per-config summary to ``results/run_summary.json``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

from ..utils.config_loader import RunConfig, config_hash
from ..utils.logger import get_logger
from .checkpoint_manager import CheckpointManager
from .curriculum import ADVANCE, CAPPED, ROLLBACK, CurriculumScheduler

__all__ = ["Orchestrator", "resolve_device", "build_policy_kwargs"]

_SB3: Dict[str, Any] = {}
_SB3_NOISE = None


def _algo(name: str):
    """Lazy stable_baselines3 import (module must stay importable without it)."""
    global _SB3_NOISE
    if name not in _SB3:
        from stable_baselines3 import PPO, SAC, TD3  # noqa: F401

        _SB3.update(PPO=PPO, SAC=SAC, TD3=TD3)
    if _SB3_NOISE is None:
        from stable_baselines3.common.noise import NormalActionNoise

        _SB3_NOISE = NormalActionNoise
    return _SB3[name]


def resolve_device(device: str) -> str:
    """'auto' -> cuda when available, else cpu."""
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def build_policy_kwargs(algo_name: str, net_arch: Dict[str, List[int]], activation: str) -> Dict[str, Any]:
    """SB3 policy_kwargs (net_arch dict + activation_fn *class*)."""
    import torch.nn as nn

    act = {"ReLU": nn.ReLU, "Tanh": nn.Tanh, "ELU": nn.ELU}[activation]
    # PPO: dict(pi=[...], vf=[...]); SAC/TD3: dict(pi=[...], qf=[...])
    return {"net_arch": dict(net_arch), "activation_fn": act}


def obs_dim_for(stage, history: Dict[str, int]) -> int:
    """Observation dimension for a stage + per-config history settings."""
    frames, skip = int(history["frames"]), int(history["skip"])
    if not stage.obs.include_target:
        return 14
    return 19 * (frames + 1)


class Orchestrator:
    """Runs every config of a :class:`RunConfig` through the curriculum."""

    def __init__(
        self,
        run_cfg: RunConfig,
        *,
        data_dir: Optional[str] = None,
        logger=None,
        vecenv: str = "auto",
    ) -> None:
        self.cfg = run_cfg
        self.data_dir = Path(data_dir or run_cfg.data_dir)
        self.log = logger or get_logger("orchestrator")
        self.vecenv = vecenv
        self.device = resolve_device(run_cfg.device)
        self._run_id = time.strftime("%Y%m%d-%H%M%S")

    # ======================================================================
    # top-level
    # ======================================================================
    def run(self) -> Dict[str, Any]:
        self.log.info("run_id=%s device=%s data_dir=%s", self._run_id, self.device, self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        resume = CheckpointManager.run_state_path(self.data_dir)
        state: Optional[Dict[str, Any]] = CheckpointManager._atomic_read_json(resume)
        h = config_hash(self.cfg)
        if state is not None and state.get("config_hash") != h:
            self.log.warning(
                "run_state.json exists but config hash differs -- starting fresh "
                "(found %s, expected %s)", str(state.get("config_hash"))[:12], h[:12]
            )
            state = None

        smoke = self.cfg.smoke_block()
        if smoke:
            self.log.info("SMOKE mode: config=%s stages=%s steps/stage=%s",
                          smoke.get("config"), smoke.get("stages"), smoke.get("steps_per_stage"))
        configs = self.cfg.config_names()
        if smoke and smoke.get("config") in configs:
            configs = [smoke["config"]]

        started = time.time()
        completed: Set[str] = set(state.get("configs_completed", [])) if state else set()
        summary: Dict[str, Any] = {
            "run_id": self._run_id,
            "config_file": self.cfg.source,
            "config_hash": h,
            "device": self.device,
            "data_dir": str(self.data_dir),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "configs": {},
        }

        total_all = int(state.get("total_steps_all_stages", 0)) if state else 0

        for cname in configs:
            if cname in completed:
                self.log.info("config %s already completed -- skipping", cname)
                continue
            resume_for_cfg = None
            if state is not None and state.get("current_config") == cname:
                resume_for_cfg = dict(state)
            cfg_result = self._run_config(
                cname, resume_for_cfg, smoke=smoke if smoke else None
            )
            summary["configs"][cname] = cfg_result["summary"]
            total_all += int(cfg_result.get("steps_added", 0))
            if cfg_result["completed"]:
                completed.add(cname)
                cm = CheckpointManager(self.data_dir, cname, cfg_result["algo"])
                self._write_state(
                    configs_completed=sorted(completed),
                    current_config=None,
                    current_stage=0,
                    stage_num=None,
                    current_stage_steps=0,
                    total_steps_all_stages=total_all,
                    last_checkpoint_path=str(cfg_result.get("final_model", "")),
                    episode_buffer=None,
                    retries={},
                )

        # final summary file
        summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        summary["wall_seconds"] = round(time.time() - started, 1)
        out = self.data_dir / "results" / "run_summary.json"
        self._atomic_json(out, summary)
        self.log.info("run_summary written to %s", out)
        return summary

    # ======================================================================
    # helpers
    # ======================================================================
    @staticmethod
    def _atomic_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = os.open(str(path.parent / (path.stem + ".tmp")), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        import json

        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        os.replace(str(path.parent / (path.stem + ".tmp")), str(path))

    def _write_state(self, **kw: Any) -> None:
        payload = dict(kw)
        payload["run_id"] = self._run_id
        payload["config_file"] = self.cfg.source
        payload["config_hash"] = config_hash(self.cfg)
        payload["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        path = CheckpointManager.run_state_path(str(self.data_dir))
        CheckpointManager._atomic_write_json(path, payload)

    def _build_vec_envs(self, stage, history: Dict[str, int], monitor_dir: Optional[str], base_seed: int):
        """Return (vec_env, factories).  SubprocVecEnv with DummyVecEnv
        fallback when the platform can't spawn subprocesses."""
        n = self.cfg.n_parallel_envs
        frames, skip = int(history["frames"]), int(history["skip"])
        from ..envs.base_env import make_env_factory

        factories = [
            make_env_factory(stage, frames, skip, seed=base_seed + rank,
                             monitor_dir=monitor_dir)
            for rank in range(n)
        ]
        use_subproc = self.vecenv in ("auto", "subproc")
        if use_subproc:
            try:
                from stable_baselines3.common.vec_env import SubprocVecEnv

                return SubprocVecEnv(factories)
            except Exception:
                self.log.warning("SubprocVecEnv unavailable -- using DummyVecEnv")
        from stable_baselines3.common.vec_env import DummyVecEnv

        return DummyVecEnv(factories)

    def _make_scheduler(self, stage, budget: int, roll_cfg: Dict[str, Any], smoke: bool):
        if smoke:
            return CurriculumScheduler(stage, budget=budget, rollback_threshold=0.0)
        rt = stage.rollback_threshold
        if rt is None:
            # plan §6.2: roll back when success drops below
            # (threshold_scale * required success_rate).  An explicit
            # per-stage override wins.
            rt = stage.success_rate * float(roll_cfg.get("threshold_scale", 0.5))
        return CurriculumScheduler(
            stage,
            budget=budget,
            rollback_threshold=rt,
            rollback_min_steps=int(budget * float(roll_cfg.get("min_steps_scale", 0.25))),
        )

    def _create_model(self, cname: str, stage, history, vec):
        conf = self.cfg.config(cname)
        algo_cls = _algo(conf["algo"])  # also initialises _SB3_NOISE
        kwargs = dict(
            env=vec,
            policy="MlpPolicy",
            policy_kwargs=build_policy_kwargs(conf["algo"], conf["net_arch"], conf["activation"]),
            seed=self.cfg.seed + int(stage.id.split("_")[1]),
            device=self.device,
            tensorboard_log=(
                str(self.cfg.data_dir / "tb_logs" / cname / stage.id)
                if self.cfg.tensorboard else None
            ),
        )
        hyper = dict(conf.get("hyperparameters", {}))
        kwargs.update(hyper)
        if conf["algo"] == "TD3" and "action_noise" not in kwargs:
            n_act = int(vec.action_space.shape[0])
            sigma = float(conf.get("action_noise_sigma", 0.1))
            kwargs["action_noise"] = _SB3_NOISE(np.zeros(n_act), sigma * np.ones(n_act))
        return algo_cls(**kwargs)

    def _load_model(self, cname: str, path: str, vec) -> Any:
        algo = self.cfg.config(cname)["algo"]
        model = _algo(algo).load(str(path), env=vec, device=self.device)
        # restore the off-policy replay buffer alongside the checkpoint
        try:
            from .checkpoint_manager import CheckpointManager as _CM

            _CM.load_replay_buffer(model, Path(path))
        except Exception:
            pass
        return model

    # ======================================================================
    # per-config loop
    # ======================================================================
    def _run_config(
        self, cname: str, resume: Optional[Dict[str, Any]], smoke: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        conf = self.cfg.config(cname)
        algo_name = conf["algo"]
        history = self.cfg.config_obs_history(cname)
        cm = CheckpointManager(self.data_dir, cname, algo_name)
        cm.ensure()

        smoke_stages = [int(s) for s in smoke["stages"]] if smoke and smoke.get("stages") else None
        stage_numbers: List[int] = [int(s) for s in smoke_stages] if smoke_stages \
            else self.cfg.config_stages(cname)

        roll = self.cfg.rollback_cfg
        max_attempts = int(roll.get("max_attempts", 2))
        last_model = None
        # JSON round-trip turns int keys into strings -- coerce back so
        # retries.get(stage_num, 0) keeps working across a resume.
        raw_retries = resume.get("retries", {}) if resume else {}
        retries: Dict[int, int] = {int(k): int(v) for k, v in raw_retries.items()}
        idx = int(resume.get("current_stage", 0)) if resume else 0
        # True when this config was resumed from run_state.json (enables
        # mid-stage periodic-checkpoint discovery for interrupted stages).
        resumed: bool = resume is not None
        reentry: bool = False
        next_scale: float = 1.0
        resume_configs_completed: List[str] = []
        if resume is not None:
            next_scale = float(resume.get("current_stage_scale", 1.0))
            resume_configs_completed = list(resume.get("configs_completed", []))
            # interrupted between the last stage transition and completion:
            # replay the final stage (it resumes from its own final ckpt and
            # immediately returns CAPPED with remaining budget 0).
            idx = min(idx, max(0, len(stage_numbers) - 1))
        last_ckpt: Optional[str] = None
        steps_added = 0
        router: Dict[str, Any] = {}

        self.log.info("=== config %s | algo %s | device %s | stages %s",
                      cname, algo_name, self.device, stage_numbers)

        while idx < len(stage_numbers):
            stage_num = int(stage_numbers[idx])
            stage = self.cfg.stage(stage_num)
            stage_scale = next_scale
            budget = max(1, int(stage.max_training_steps * stage_scale))
            if smoke and smoke.get("steps_per_stage"):
                budget = min(budget, int(smoke["steps_per_stage"]))

            sched = self._make_scheduler(stage, budget, roll, bool(smoke))
            resume_ck: Optional[str] = None
            if (
                resume is not None
                and idx == int(resume.get("current_stage", -1))
            ):
                # run_state.json describes the stage that JUST finished; its
                # checkpoint is only loadable directly when it belongs to THIS
                # stage (the interrupted-just-before-completion "replay final
                # stage" edge).  Otherwise the ckpt belongs to a previous stage
                # and the transfer / periodic / reentry paths below handle it.
                if int(resume.get("stage_num") or -1) == stage_num:
                    resume_ck = resume.get("last_checkpoint_path")
                resume = None  # only the first stage uses resume state

            vec = None
            try:
                self.log.info("-- stage %s (%s) budget=%d retries=%d reentry=%s",
                              stage.id, stage.name, budget, retries.get(stage_num, 0), reentry)
                monitor_dir = str(cm.monitor_dir(stage.id))
                vec = self._build_vec_envs(stage, history, monitor_dir, base_seed=self.cfg.seed)

                model = None
                # --- model selection --------------------------------------
                if resume_ck and Path(resume_ck).exists():
                    model = self._load_model(cname, resume_ck, vec)
                    self.log.info("resumed from %s", resume_ck)
                if model is None and reentry:
                    own_final = cm.latest_final(stage.id)
                    if own_final is not None:
                        model = self._load_model(cname, str(own_final), vec)
                        # reduced-budget re-entry (plan §6.2) must actually
                        # train: the loaded final already consumed the original
                        # budget, so reset the step count or `remaining` would
                        # be <= 0 and the retrain would be skipped entirely.
                        model.num_timesteps = 0
                        self.log.info("re-entering stage %s from %s (budget=%d)",
                                      stage.id, own_final.name, budget)
                if model is None and resumed:
                    # resumed run, no direct resume ckpt / no re-entry: if the
                    # process died MID-stage, the most recent periodic
                    # checkpoint of THIS stage is the right continuation
                    # (it also restores the off-policy replay buffer).
                    periodic = cm.latest_periodic(stage.id)
                    if periodic is not None:
                        model = self._load_model(cname, str(periodic), vec)
                        self.log.info("resumed mid-stage %s from periodic %s",
                                      stage.id, periodic.name)
                if model is None and idx > 0:
                    prev_stage = self.cfg.stage(stage_numbers[idx - 1])
                    if obs_dim_for(stage, history) == obs_dim_for(prev_stage, history):
                        prev_final = cm.latest_final(prev_stage.id)
                        if prev_final is not None:
                            old = _algo(algo_name).load(str(prev_final), device=self.device)
                            new = self._create_model(cname, stage, history, vec)
                            new.set_parameters(old.get_parameters(), exact_match=True)
                            model = new
                            self.log.info("transferred weights from %s", prev_final.name)
                if model is None:
                    model = self._create_model(cname, stage, history, vec)

                # --- learn -------------------------------------------------
                remaining = budget - int(model.num_timesteps)
                from stable_baselines3.common.callbacks import CallbackList
                from .callbacks import (
                    CurriculumCallback,
                    CurriculumCheckpointCallback,
                    MetricsCallback,
                    RewardComponentCallback,
                )

                cur_cb = CurriculumCallback(sched)
                log_interval = 10 if algo_name == "PPO" \
                    else max(10000, self.cfg.checkpoint_interval_steps)
                cb = CallbackList([
                    CurriculumCheckpointCallback(cm, stage.id, self.cfg.checkpoint_interval_steps),
                    RewardComponentCallback(log_interval=log_interval),
                    MetricsCallback(log_interval=log_interval),
                    cur_cb,
                ])

                if remaining > 0:
                    self.log.info("learning: remaining=%d (already %d)", remaining, int(model.num_timesteps))
                    model.learn(
                        total_timesteps=remaining,
                        reset_num_timesteps=False,
                        callback=cb,
                        tb_log_name=f"{algo_name}_{stage.id}",
                        log_interval=log_interval,
                        progress_bar=False,
                    )

                result = cur_cb.stage_finished if cur_cb.stage_finished is not None else CAPPED
                steps_here = int(model.num_timesteps)
                steps_added += steps_here
                success_rate = sched.rate()
                last_model = model
                self.log.info("stage %s result=%s steps=%d success_rate=%.3f",
                              stage.id, result, steps_here, success_rate)

                # --- result handling ---------------------------------------
                if result == ADVANCE or result == CAPPED:
                    final_path = cm.save_model(model, stage.id, steps_here, final=True)
                    last_ckpt = str(final_path)
                    retries.pop(stage_num, None)
                    router[stage_num] = {
                        "result": result, "steps": steps_here, "success_rate": success_rate,
                    }
                    if result == CAPPED:
                        self.log.warning("stage %s capped -- did not converge", stage.id)
                    idx += 1
                    next_scale = 1.0
                    reentry = False
                elif result == ROLLBACK:
                    retries[stage_num] = retries.get(stage_num, 0) + 1
                    router[stage_num] = {
                        "result": "rollback", "steps": steps_here, "success_rate": success_rate,
                    }
                    self.log.warning("stage %s rolled back (attempt %d/%d)",
                                     stage.id, retries[stage_num], max_attempts)
                    if idx == 0 or retries[stage_num] > max_attempts:
                        self.log.warning("stage %s marked STUCK -- skipping rest of config %s",
                                         stage.id, cname)
                        router[stage_num]["result"] = "stuck"
                        break
                    # re-enter the previous stage on a reduced budget (plan §6.2)
                    idx -= 1
                    reentry = True
                    next_scale = float(roll.get("retry_budget_scale", 0.5))
                else:  # pragma: no cover - defensive
                    self.log.error("unexpected stage result %r", result)
                    break
            finally:
                if vec is not None:
                    vec.close()

            # persist resume state after every transition (current_stage
            # is the *next* stage to run; current_stage_scale applies to it)
            ep_buffer = list(sched.state_dict().get("records", []))
            self._write_state(
                configs_completed=resume_configs_completed,
                current_config=cname,
                current_stage=idx,
                stage_num=stage_num,
                current_stage_steps=steps_here,
                total_steps_all_stages=steps_added,
                last_checkpoint_path=last_ckpt,
                episode_buffer=ep_buffer,
                retries=retries,
                current_stage_scale=next_scale,
            )

        conf_done = bool(router) and all(
            r["result"] in (ADVANCE, CAPPED) for r in router.values()
        )
        final_model_path: Optional[str] = None
        if conf_done and last_model is not None:
            final_model_path = str(cm.save_config_final(last_model))
        self.log.info("config %s %s%s", cname,
                      "COMPLETED" if conf_done else "INTERRUPTED",
                      f" final_model={final_model_path}" if final_model_path else "")
        return {
            "algo": algo_name,
            "summary": {
                "algo": algo_name,
                "stages": router,
                "final_model": final_model_path,
                "steps": steps_added,
            },
            "completed": conf_done,
            "steps_added": steps_added,
            "final_model": final_model_path,
        }