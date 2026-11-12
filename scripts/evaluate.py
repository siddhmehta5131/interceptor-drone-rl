"""evaluate.py -- evaluation + metrics + optional video (plan §15).

Usage::

    python scripts/evaluate.py --model data/runs/apn_rl_v1/ppo_interceptor/final.zip \\
        --stage 3 --episodes 100 --render --output-dir data/evaluation/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api import evaluate_model

_METRIC_ORDER = [
    ("kill_rate", "kill rate"),
    ("mean_time_to_kill", "mean time-to-kill (steps)"),
    ("mean_miss_distance", "mean miss distance (m)"),
    ("mean_energy", "mean energy (int |a|^2 dt)"),
    ("mean_rl_fraction", "mean RL contribution"),
    ("mean_terminal_zem", "terminal ZEM (m)"),
    ("mean_efficiency", "trajectory efficiency"),
    ("mean_reward", "mean reward"),
    ("mean_episode_length", "mean episode length (steps)"),
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained interceptor policy.")
    parser.add_argument("--model", required=True, help="path to a saved SB3 .zip policy")
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model-yaml", default="configs/model.yaml")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--stochastic", action="store_true", help="sample actions (default deterministic)")
    parser.add_argument("--render", action="store_true", help="also write an MP4 of one episode")
    parser.add_argument("--output-dir", default="data/evaluation")
    parser.add_argument("--camera", default=None)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args(argv)

    metrics = evaluate_model(
        args.model,
        config=args.config,
        model_yaml=args.model_yaml,
        stage=args.stage,
        episodes=args.episodes,
        deterministic=not args.stochastic,
    )

    print(f"=== stage {args.stage}: {metrics.get('episodes', 0)} episodes ===")
    for key, label in _METRIC_ORDER:
        if key in metrics:
            value = metrics[key]
            print(f"  {label:32s} {value:.4f}" if isinstance(value, float) else f"  {label:32s} {value}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"stage{args.stage}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    print(f"metrics -> {metrics_path}")

    if args.render:
        try:
            from visualiser.cli import generate_trajectory
            from visualiser.animator import InterceptAnimator

            rollout = generate_trajectory(
                args.stage, model_path=args.model,
                config_path=args.config, model_yaml=args.model_yaml, seed=args.seed,
            )
            animator = InterceptAnimator(
                model_name=Path(args.model).stem, stage=args.stage,
                trajectory_data=rollout["trajectory"], camera=args.camera,
                kill_radius=rollout["kill_radius"],
            )
            video = str(out_dir / f"stage{args.stage}_intercept.mp4")
            animator.render(video, fps=args.fps)
            print(f"video   -> {video}")
        except Exception as exc:  # pragma: no cover - depends on pyvista
            print(f"render skipped: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
