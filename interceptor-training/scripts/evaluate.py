#!/usr/bin/env python3
"""evaluate.py -- run a trained model for N episodes on a stage and report
aggregated metrics (plan §10 success criteria + kill/miss statistics).

Usage:
    python scripts/evaluate.py --config <run.yaml> --config-name ppo_baseline \
        --stage 3 --model <path.zip> [--episodes 50] [--data-dir DIR]

When ``--model`` is omitted the most relevant persisted model is used: the
stage's own final checkpoint, else its latest periodic checkpoint, else the
config-wide final model.  The stage-specific checkpoints are preferred over
the config-wide final model, which is the LAST stage's policy after a full
curriculum run and is not representative of earlier stages.

Results are printed and also persisted to
``<data_dir>/results/eval_<config>_stage_<N>.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate an interceptor model")
    p.add_argument("--config", default="/data/configs/default_run.yaml")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--config-name", default="ppo_baseline")
    p.add_argument("--stage", type=int, default=1)
    p.add_argument("--model", default=None, help="path to a saved .zip model")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=7)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    from src.envs.base_env import make_env_factory
    from src.envs.stage_config import StageConfig
    from src.utils import load_run_config
    from src.utils.logger import get_logger
    from src.training.checkpoint_manager import CheckpointManager
    from src.training.curriculum import episode_success

    log = get_logger("evaluate")
    cfg = load_run_config(args.config)
    if args.data_dir:
        cfg.global_cfg["data_dir"] = args.data_dir
    data_dir = cfg.global_cfg["data_dir"]
    conf = cfg.config(args.config_name)
    algo = conf["algo"]
    history = cfg.config_obs_history(args.config_name)
    # effective stage: built-in defaults + any YAML overrides
    stage: StageConfig = cfg.stage(args.stage)

    cm = CheckpointManager(data_dir, args.config_name, algo)
    model_path = args.model
    if not model_path:
        # prefer the stage's own checkpoints over the config-wide final model
        # (which is the last stage's policy after a full curriculum run).
        candidates = [
            cm.latest_final(stage.id),
            cm.latest_periodic(stage.id),
            cm.config_final_path(),
        ]
        for cand in candidates:
            if cand is not None and Path(cand).exists():
                model_path = str(cand)
                break
    if not model_path:
        log.error("no model found for %s stage %s (pass --model)", args.config_name, args.stage)
        return 2
    log.info("loading model %s (algo %s)", model_path, algo)

    from src.training.orchestrator import _algo

    vec = None
    try:
        factory = make_env_factory(stage, history["frames"], history["skip"],
                                   seed=args.seed, monitor_dir=None)
        from stable_baselines3.common.vec_env import DummyVecEnv

        vec = DummyVecEnv([factory])
        model = _algo(algo).load(model_path, env=vec, device="auto")

        rows = []
        for ep in range(max(1, args.episodes)):
            obs = vec.reset()
            ep_reward = 0.0
            done = False
            steps = 0
            while not done and steps < stage.max_episode_steps + 1:
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
        # aggregate
        n = len(rows)
        kill_rate = float(np.mean([r["kill"] for r in rows]))
        success_rate = float(np.mean([r["success"] for r in rows]))
        fin_dist = [r["final_distance"] for r in rows if np.isfinite(r["final_distance"])]
        tti = [r["time_to_intercept_steps"] for r in rows
               if np.isfinite(r["time_to_intercept_steps"])]
        agg = {
            "config": args.config_name,
            "stage": args.stage,
            "model": str(model_path),
            "episodes": n,
            "seed": args.seed,
            "kill_rate": kill_rate,
            "success_rate": success_rate,
            "required_success_rate": stage.success_rate,
            "mean_episode_reward": float(np.mean([r["reward"] for r in rows])),
            "mean_final_distance": float(np.mean(fin_dist)) if fin_dist else None,
            "mean_time_to_intercept_steps": float(np.mean(tti)) if tti else None,
            "mean_time_to_intercept_s": (float(np.mean(tti)) * 0.01) if tti else None,
            "mean_miss_distance": float(np.nanmean([r["miss_distance"] for r in rows])),
        }
        print(f"\n=== Evaluate {args.config_name} stage {args.stage} ({n} eps) ===")
        print(f"kill_rate            : {kill_rate:.3f}   (required {stage.success_rate})")
        print(f"success_rate         : {success_rate:.3f}   (required {stage.success_rate})")
        print(f"mean episode reward  : {agg['mean_episode_reward']:.2f}")
        if fin_dist:
            print(f"mean final distance  : {np.mean(fin_dist):.2f} m  (threshold {stage.threshold})")
        if tti:
            print(f"mean time_to_intercept: {np.mean(tti):.1f} steps "
                  f"({np.mean(tti) * 0.01:.2f} s)")
        print(f"mean miss_distance   : {agg['mean_miss_distance']:.2f} m")
        # persist the aggregate for later comparison (plan evaluation flow)
        out_path = Path(data_dir) / "results" / f"eval_{args.config_name}_stage_{args.stage}.json"
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