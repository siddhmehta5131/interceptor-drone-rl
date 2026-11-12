"""run.py -- worked examples + CLI for the APN+RL interceptor (plan B-3).

This file is documentation that executes.  It is *not* imported by the
training pipeline; copy the parts you need into your own script, or run this
file directly to drive a run from the command line::

    python run.py --stages 1 --name demo --steps 20000
    python run.py --stages 1,2,3 --name apn_rl_v1

The canonical three-stage curriculum is exposed by :func:`full_curriculum`
below; the smoke test trains stage 1 for a few thousand steps so you can verify
the whole stack in a couple of minutes.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Sequence

from src.api import train_model
from src.results import TrainResult

__all__ = ["smoke_example", "full_curriculum", "format_summary", "main"]


def smoke_example(
    *,
    name: str = "smoke",
    model_file: str = "configs/model.yaml",
    config_file: str = "configs/config.yaml",
    steps: int = 20000,
) -> TrainResult:
    """Train stage 1 for a small budget -- fast end-to-end verification."""
    return train_model(
        source=model_file,
        stages=[1],
        name=name,
        config=config_file,
        max_steps_per_stage=steps,
    )


def full_curriculum(
    *,
    name: str = "apn_rl_v1",
    model_file: str = "configs/model.yaml",
    config_file: str = "configs/config.yaml",
) -> TrainResult:
    """Train the full static -> polynomial -> evasive curriculum from YAML."""
    return train_model(
        source=model_file,
        stages=[1, 2, 3],
        name=name,
        config=config_file,
    )


def format_summary(result: TrainResult) -> str:
    """Human-readable one-block summary of a :class:`TrainResult`."""
    lines = [
        f"run {result.run_dir}  status={result.status}  "
        f"hash={result.config_hash[:12]}  {result.duration_s:.1f}s",
    ]
    for model_id in result.model_ids():
        model = getattr(result, model_id)
        stages = ", ".join(
            f"{o.stage}:{o.status}({o.kill_rate:.2f}/{o.steps_trained} steps)"
            for o in model.stages
        )
        line = f"  {model_id} [{model.status}] {stages}"
        if model.final_path:
            line += f"  -> {model.final_path}"
        lines.append(line)
    return "\n".join(lines)


def _parse_stages(text: str) -> List[int]:
    out: List[int] = []
    for chunk in str(text).replace(" ", "").split(","):
        if not chunk:
            continue
        try:
            out.append(int(chunk))
        except ValueError:
            raise SystemExit(f"--stages: {chunk!r} is not an integer")
    if not out:
        raise SystemExit("--stages: give at least one stage, e.g. --stages 1,2")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Train APN+RL interceptor policies through the curriculum.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source", default="configs/model.yaml",
        help="model YAML defining the policy to train",
    )
    parser.add_argument("--stages", default="1,2,3", help="comma-separated stage numbers")
    parser.add_argument("--name", required=True, help="run name (folder under <data_dir>)")
    parser.add_argument("--config", default="configs/config.yaml", help="training settings YAML")
    parser.add_argument("--data-dir", default=None, help="override global.data_dir")
    parser.add_argument("--device", default=None, help="override global.device (auto/cpu/cuda)")
    parser.add_argument(
        "--steps", type=int, default=None,
        help="cap training steps per stage (smoke tests)",
    )
    parser.add_argument("--quiet", action="store_true", help="silence per-stage progress")
    args = parser.parse_args(argv)

    result = train_model(
        source=args.source,
        stages=_parse_stages(args.stages),
        name=args.name,
        config=args.config,
        data_dir=args.data_dir,
        device=args.device,
        max_steps_per_stage=args.steps,
        verbose=0 if args.quiet else 1,
    )
    print(format_summary(result))

    return 0 if result.status == "complete" else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
