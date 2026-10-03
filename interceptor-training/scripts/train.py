#!/usr/bin/env python3
"""train.py -- container entry point around :func:`src.api.train_model` (B-4).

The container image keeps this file as its ENTRYPOINT so ``docker compose up``
still trains, but the CLI is now a thin wrapper over the public API.  All the
curriculum logic lives in ``src/training/orchestrator.py`` and every argument
here maps onto a ``train_model`` parameter.

Usage::

    # full curriculum from the shipped model file
    python scripts/train.py --source configs/model.yaml --stages 1,2,3,4 \
        --name experiment_1 --config configs/config.yaml

    # fast integration check (tiny budgets, no rollback, capped -> keep going)
    python scripts/train.py --smoke --smoke-stages 1,2,3 --smoke-steps 1000

    # continue an earlier run
    python scripts/train.py --continue-from /data/runs/experiment_1 \
        --stages 5,6,7 --name experiment_2

Exit code is ``0`` when every requested model finished (``completed`` or
``capped``) and ``1`` otherwise, so CI and Docker can gate on it.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

#: Where docker-compose mounts the shipped configs.
DEFAULT_CONFIG = os.environ.get("INTERCEPTOR_CONFIG", "/data/configs/config.yaml")
DEFAULT_MODEL = os.environ.get("INTERCEPTOR_MODEL", "/data/configs/model.yaml")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train.py",
        description="Interceptor curriculum training (wraps src.api.train_model)",
    )
    src = p.add_argument_group("source")
    g = src.add_mutually_exclusive_group()
    g.add_argument("--source", default=None,
                   help=f"model definition YAML (default: {DEFAULT_MODEL})")
    g.add_argument("--continue-from", default=None,
                   help="run summary / result dir / model zip of an earlier run")
    p.add_argument("--config", default=DEFAULT_CONFIG,
                   help="training settings YAML (ignored when --continue-from "
                        "supplies its own)")
    p.add_argument("--stages", default="1,2",
                   help="comma-separated stage numbers to train")
    p.add_argument("--name", default=None,
                   help="run name; artifacts go to <data_dir>/runs/<name>/ "
                        "(default: 'run_<timestamp>')")
    p.add_argument("--models", default=None,
                   help="comma-separated model ids to train (default: all)")
    p.add_argument("--data-dir", default=None, help="override data_dir")
    p.add_argument("--seed", type=int, default=None, help="override the global seed")
    p.add_argument("--save", default=None, help="also copy each model here")
    p.add_argument("--vecenv", default="auto", choices=["auto", "subproc", "dummy"])
    p.add_argument("--no-resume", action="store_true",
                   help="ignore run_state.json and start each stage fresh")

    p.add_argument("--smoke", action="store_true",
                   help="integration smoke test: tiny budgets, capped stages advance")
    p.add_argument("--smoke-model", default=DEFAULT_MODEL,
                   help="model YAML used by --smoke")
    p.add_argument("--smoke-config", default=DEFAULT_CONFIG,
                   help="settings YAML used by --smoke")
    p.add_argument("--smoke-stages", default="1,2,3")
    p.add_argument("--smoke-steps", type=int, default=1000)
    p.add_argument("--smoke-models", default=None,
                   help="comma-separated model ids for --smoke (default: all)")
    return p


def _parse_int_list(text: str) -> list:
    out = [int(part) for part in str(text).replace(" ", "").split(",") if part]
    if not out:
        raise SystemExit("error: --stages must list at least one stage number")
    return out


def _smoke_config(path: str, stages, steps: int, data_dir=None):
    """A settings object with a tiny budget per stage (no temp files)."""
    from src.utils.config_loader import load_config

    cfg = load_config(path)
    for n in stages:
        key = f"stage_{int(n)}"
        overrides = dict(cfg.stage_overrides.get(key, {}))
        overrides["max_training_steps"] = max(1, int(steps))
        # A smoke run must march forward even when a stage cannot reach its
        # success target, and it must not spend budget on rollback retries.
        cfg.stage_overrides[key] = overrides
    if data_dir:
        cfg.global_cfg["data_dir"] = data_dir
    cfg.on_capped = "continue"
    cfg.rollback = {**dict(cfg.rollback), "max_attempts": 0}
    cfg.recompute_hash()
    return cfg


def main(argv: object = None) -> int:
    args = build_parser().parse_args(argv)

    from src.api import train_model
    from src.utils.logger import get_logger

    log = get_logger("train")
    name = args.name or f"run_{time.strftime('%Y%m%d-%H%M%S')}"
    resume = not args.no_resume

    if args.smoke:
        stages = _parse_int_list(args.smoke_stages)
        source = args.source or args.smoke_model
        config = _smoke_config(args.smoke_config, stages, args.smoke_steps, args.data_dir)
        result = train_model(
            source=source,
            stages=stages,
            name=name,
            config=config,
            model_ids=(args.smoke_models.split(",") if args.smoke_models else None),
            seed=args.seed,
            vecenv=args.vecenv,
            resume=resume,
        )
    else:
        if args.continue_from:
            source: object = args.continue_from
        else:
            source = args.source or DEFAULT_MODEL
        config: object = args.config
        result = train_model(
            source=source,
            stages=_parse_int_list(args.stages),
            name=name,
            config=config,
            model_ids=(args.models.split(",") if args.models else None),
            seed=args.seed,
            vecenv=args.vecenv,
            resume=resume,
        )

    from run import format_summary

    log.info("=== TRAINING SUMMARY ===")
    print(format_summary(result))

    if args.save:
        for model_id, model_result in result.models.items():
            try:
                model_result.save(args.save)
            except Exception as exc:  # pragma: no cover - best effort
                log.warning("could not save %s to %s: %s", model_id, args.save, exc)
    return 0 if result.completed else 1


if __name__ == "__main__":
    sys.exit(main())
