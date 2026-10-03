#!/usr/bin/env python3
"""evaluate.py -- run a trained model for N episodes on a stage and report
aggregated metrics (plan success criteria + kill/miss statistics).

The model is located from a *run* produced by ``train_model()``.  Three
sources are accepted for ``--run``:

* a run folder (``<data_dir>/runs/<name>/``) -- the ``results/run_summary.json``
  inside it is read,
* a ``run_summary.json`` / ``<model_id>.result.json`` file,
* a saved model artifact (``<model_id>.zip`` or its ``.result.json`` sidecar).

Usage::

    python scripts/evaluate.py --run /data/runs/experiment_1 \
        --models /data/configs/model.yaml --model-id ppo_baseline --stage 3

    python scripts/evaluate.py --model my_model/ppo_baseline.zip \
        --models /data/configs/model.yaml --stage 5 --episodes 100

``--model`` (an explicit ``.zip``) wins over ``--run``.  The stage-specific
checkpoints inside the run are preferred over the run's config-wide final
model, because the latter is the LAST stage's policy after a full curriculum
run and is not representative of earlier stages.

Results are printed and also persisted to
``<run_dir>/results/eval_<model_id>_stage_<N>.json`` (or ``--out``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_MODEL_FILE = os.environ.get("INTERCEPTOR_MODEL", "/data/configs/model.yaml")
DEFAULT_CONFIG = os.environ.get("INTERCEPTOR_CONFIG", "/data/configs/config.yaml")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate an interceptor model")
    p.add_argument("--run", default=None,
                   help="run folder / run_summary.json / result.json / .zip produced by train_model()")
    p.add_argument("--model", default=None, help="explicit path to a saved .zip model")
    p.add_argument("--model-id", default="ppo_baseline",
                   help="model id to evaluate when several models share a run")
    p.add_argument("--models", default=DEFAULT_MODEL_FILE,
                   help="configs/model.yaml describing architecture + observation layout")
    p.add_argument("--config", default=DEFAULT_CONFIG, help="configs/config.yaml")
    p.add_argument("--data-dir", default=None, help="override global.data_dir")
    p.add_argument("--stage", type=int, default=1)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None, help="explicit output JSON path")
    return p


# ---------------------------------------------------------------- discovery
def _resolve_run_dir(run: str) -> Path:
    """Return the run FOLDER for a run path (``run_summary.json`` -> parent)."""
    path = Path(run)
    if path.name == "run_summary.json" and path.parent.name == "results":
        return path.parent.parent
    if path.is_dir():
        return path
    if path.is_file():
        return path.parent
    return path


def _load_train_result(run: Optional[str], model_path: Optional[str]):
    """Load the :class:`TrainResult` / ``ModelResult`` describing a model."""
    from src.results import ModelResult, TrainResult

    source = run or model_path
    if not source:
        return None
    path = Path(source)
    if path.is_dir():
        candidates = [path / "results" / "run_summary.json",
                      path / "run_summary.json"]
    elif path.name.endswith(".result.json"):
        try:
            return ModelResult.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return None
    else:
        candidates = [path]
    for cand in candidates:
        if cand.name.endswith(".result.json"):
            try:
                return ModelResult.from_dict(json.loads(cand.read_text(encoding="utf-8")))
            except Exception:
                continue
        if cand.is_file():
            try:
                return TrainResult.from_dict(json.loads(cand.read_text(encoding="utf-8")))
            except Exception:
                continue
    return None


def _find_model_path(result, model_id: str, run_dir: Path, algo: str,
                     stage) -> Optional[str]:
    """Prefer the stage's own checkpoints over the run-wide final model."""
    from src.results import ModelResult

    if isinstance(result, ModelResult):
        final = result.final_path or result.final_model
        if final and Path(final).exists():
            return str(final)
        result = None
    if result is not None:
        entry = result.models.get(model_id)
        if entry is None:
            return None
        final = entry.final_path or entry.final_model
        if final and Path(final).exists():
            return str(final)

    from src.training.checkpoint_manager import CheckpointManager

    cm = CheckpointManager(run_dir, model_id, algo)
    candidates = [
        cm.latest_final(stage.id),
        cm.latest_periodic(stage.id),
        cm.config_final_path(),
    ]
    for cand in candidates:
        if cand is not None and Path(cand).exists():
            return str(cand)
    return None


# ------------------------------------------------------------------ metrics
def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    from src.envs.base_env import make_env_factory
    from src.envs.stage_config import StageConfig
    from src.training.checkpoint_manager import CheckpointManager
    from src.training.curriculum import episode_success
    from src.utils import load_config, load_models
    from src.utils.logger import get_logger

    log = get_logger("evaluate")

    cfg = load_config(args.config)
    if args.data_dir:
        cfg.global_cfg["data_dir"] = args.data_dir
    models = load_models(args.models)
    cfg.bind_models(models, model_source=str(args.models))

    model_id = args.model_id
    if model_id not in models:
        log.error("model id %r is not defined in %s (have: %s)",
                  model_id, args.models, ", ".join(sorted(models)))
        return 2
    mdef = models[model_id]
    algo = mdef.algo

    data_dir = Path(cfg.data_dir)
    stage: StageConfig = cfg.stage(args.stage)

    result = _load_train_result(args.run, args.model)
    run_dir = _resolve_run_dir(args.run) if args.run else (data_dir / "runs" / "unresolved")

    model_path = args.model
    if not model_path:
        model_path = _find_model_path(result, model_id, run_dir, algo, stage)
    if not model_path:
        log.error("no model found for %s stage %s -- pass --model or --run", model_id, args.stage)
        return 2
    log.info("loading model %s (algo %s, model id %s)", model_path, algo, model_id)

    from src.training.orchestrator import _algo

    vec = None
    try:
        factory = make_env_factory(
            stage,
            history_frames=mdef.obs_frames,
            history_skip=mdef.obs_skip,
            target_alt=cfg.target_alt_range,
            future_samples=cfg.observation.future_samples,
            future_skip=cfg.observation.future_skip,
            predictor=cfg.observation.predictor,
            future_source=cfg.observation.future_source,
            privileged_fields=tuple(mdef.privileged_critic or ()),
            seed=args.seed,
            monitor_dir=None,
        )
        from stable_baselines3.common.vec_env import DummyVecEnv

        vec = DummyVecEnv([factory])
        model = _algo(algo).load(model_path, env=vec, device=args.device)

        max_steps = stage.max_episode_steps
        rows = []
        infos = [{}]
        for ep in range(max(1, args.episodes)):
            obs = vec.reset()
            ep_reward = 0.0
            done = False
            steps = 0
            while not done and steps <= max_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, dones, infos = vec.step(action)
                ep_reward += float(reward[0])
                steps += 1
                if infos[0].get("episode_stats") is not None:
                    done = bool(dones[0])
            stats = infos[0].get("episode_stats", {}) if done else {}
            rows.append({
                "episode": ep,
                "reward": ep_reward,
                "steps": steps,
                "success": bool(episode_success(stage, stats)) if stats else False,
                "kill": bool(stats.get("kill", False)),
                "final_distance": float(stats.get("final_distance", np.inf)),
                "time_to_intercept_steps": float(stats.get("time_to_intercept", np.nan)),
                "miss_distance": float(stats.get("miss_distance", np.nan)),
            })

        n = len(rows)
        kill_rate = float(np.mean([r["kill"] for r in rows]))
        success_rate = float(np.mean([r["success"] for r in rows]))
        fin_dist = [r["final_distance"] for r in rows if np.isfinite(r["final_distance"])]
        tti = [r["time_to_intercept_steps"] for r in rows
               if np.isfinite(r["time_to_intercept_steps"])]
        miss = [r["miss_distance"] for r in rows if np.isfinite(r["miss_distance"])]
        agg = {
            "model_id": model_id,
            "algo": algo,
            "stage": args.stage,
            "model": str(model_path),
            "run_dir": str(run_dir),
            "episodes": n,
            "seed": args.seed,
            "kill_rate": kill_rate,
            "success_rate": success_rate,
            "required_success_rate": stage.success_rate,
            "mean_episode_reward": float(np.mean([r["reward"] for r in rows])),
            "mean_final_distance": float(np.mean(fin_dist)) if fin_dist else None,
            "mean_time_to_intercept_steps": float(np.mean(tti)) if tti else None,
            "mean_time_to_intercept_s": (float(np.mean(tti)) * 0.01) if tti else None,
            "mean_miss_distance": float(np.mean(miss)) if miss else None,
        }

        print(f"\n=== Evaluate {model_id} stage {args.stage} ({n} eps) ===")
        print(f"kill_rate              : {kill_rate:.3f}   (required {stage.success_rate})")
        print(f"success_rate           : {success_rate:.3f}   (required {stage.success_rate})")
        print(f"mean episode reward    : {agg['mean_episode_reward']:.2f}")
        if fin_dist:
            print(f"mean final distance    : {agg['mean_final_distance']:.2f} m"
                  f"  (threshold {stage.threshold})")
        if tti:
            print(f"mean time_to_intercept : {agg['mean_time_to_intercept_steps']:.1f} steps"
                  f"  ({agg['mean_time_to_intercept_s']:.2f} s)")
        if miss:
            print(f"mean miss_distance     : {agg['mean_miss_distance']:.2f} m")

        out_path = Path(args.out) if args.out else (
            run_dir / "results" / f"eval_{model_id}_stage_{args.stage}.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(agg, fh, indent=2, default=str)
        log.info("evaluation results written to %s", out_path)
        return 0
    finally:
        if vec is not None:
            vec.close()


if __name__ == "__main__":
    sys.exit(main())
