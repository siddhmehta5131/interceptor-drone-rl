"""run.py -- worked examples for the public training API (plan B-3).

This file is documentation that executes.  It is *not* imported by the
training pipeline; copy the parts you need into your own script, or run this
file directly to drive a run from the command line::

    python run.py --stages 1,2 --name demo --config configs/config.yaml
    python run.py --continue-from <run_dir>/results/run_summary.json --stages 3,4 --name demo2

The canonical two-phase pattern -- train the easy half of the curriculum from
YAML, then continue from that result with the hard half -- is
:func:`two_phase_example` below.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from src.api import DEFAULT_CONFIG, train_model
from src.results import ModelResult, TrainResult

__all__ = ["two_phase_example", "format_summary", "main"]


def two_phase_example(
    *,
    model_file: str = "configs/model.yaml",
    config_file: str = DEFAULT_CONFIG,
    easy_stages: Sequence[int] = (1, 2, 3, 4),
    hard_stages: Sequence[int] = (5, 6, 7),
    first_name: str = "experiment_1",
    second_name: str = "experiment_2",
    save_to: Optional[str] = "my_model/",
    model_ids: Optional[Sequence[str]] = None,
) -> TrainResult:
    """Train stages 1-4 from YAML, then continue that result through 5-7.

    The second call needs no ``config=``: the settings recorded by the first
    run are reused, and passing a ``config.yaml`` that disagrees with them
    raises instead of silently changing the model.
    """
    model1 = train_model(
        source=model_file,
        stages=list(easy_stages),
        name=first_name,
        config=config_file,
        model_ids=list(model_ids) if model_ids else None,
    )
    print(format_summary(model1))

    model2 = train_model(
        source=model1,
        stages=list(hard_stages),
        name=second_name,
    )
    print(format_summary(model2))

    for model_id in model2.models:
        if save_to:
            path = model2[model_id].save(save_to)
            print(f"saved {model_id} -> {path}")
    return model2


def format_summary(result: TrainResult) -> str:
    """Human-readable one-block summary of a :class:`TrainResult`."""
    lines = [
        f"run {result.name!r}  stages={result.stages}  "
        f"hash={result.config_hash[:12]}  {result.seconds:.1f}s",
    ]
    if result.run_dir:
        lines.append(f"  artifacts: {result.run_dir}")
    for model_id, model in result.models.items():
        stages = ", ".join(
            f"{o.stage}:{o.result}({o.success_rate:.2f}/{o.steps} steps)"
            for o in model.stage_outcomes
        )
        line = f"  {model_id} [{model.status}] {stages}"
        if model.reason:
            line += f"  -- {model.reason}"
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


def _parse_models(text: str) -> List[str]:
    return [c for c in str(text).replace(" ", "").split(",") if c]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Train interceptor policies through the curriculum.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--source", default="configs/model.yaml",
        help="model YAML for a fresh run",
    )
    src.add_argument(
        "--continue-from",
        help="path to a run_summary.json / *.result.json / saved-model directory "
             "produced by an earlier train_model() call (D-28)",
    )
    parser.add_argument("--stages", default="1,2", help="comma-separated stage numbers")
    parser.add_argument("--name", required=True, help="run name (folders under <data_dir>/runs)")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="training settings YAML")
    parser.add_argument("--models", default="", help="comma-separated model ids to train")
    parser.add_argument("--seed", type=int, default=None, help="override global.seed")
    parser.add_argument("--save", default="", help="also save final models into this directory")
    args = parser.parse_args(argv)

    stages = _parse_stages(args.stages)
    models = _parse_models(args.models) or None

    if args.continue_from:
        source: object = TrainResult.from_dict(_load_summary(args.continue_from))
    else:
        source = args.source

    result = train_model(
        source=source,
        stages=stages,
        name=args.name,
        config=args.config,
        model_ids=models,
        seed=args.seed,
    )
    print(format_summary(result))

    if args.save:
        for model_id in result.models:
            print(f"saved {model_id} -> {result[model_id].save(args.save)}")

    return 0 if result.completed else 1


def _load_summary(path: str) -> dict:
    """Read a saved run descriptor back into a plain dict for ``from_dict``."""
    import json

    p = Path(path)
    if p.is_dir():
        candidates = sorted(p.glob("*.result.json")) or sorted(p.glob("run_summary.json"))
        if not candidates:
            raise SystemExit(f"{path}: no run_summary.json or *.result.json found")
        p = candidates[0]
    elif p.suffix == ".zip":
        p = p.with_name(p.stem + ".result.json")
    if not p.exists():
        raise SystemExit(f"{path}: not found")
    data = json.loads(p.read_text(encoding="utf-8"))
    if "models" not in data:
        raise SystemExit(f"{p}: does not look like a saved run descriptor")
    return data


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
