#!/usr/bin/env python3
"""train.py -- training entry point (container ENTRYPOINT).

Usage:
    python scripts/train.py [--config CONFIG] [--data-dir DIR]
    python scripts/train.py --smoke [--smoke-config ppo_baseline]
                            [--smoke-stages 1,2,3] [--smoke-steps 1000]

``--smoke`` runs the plan §12 integration smoke test: one config, a subset of
stages, a small step budget per stage (advance is disabled, the hard cap
drives stage progression).  The default config path is ``/data/configs/default_run.yaml``
(host-mounted by docker-compose) and can be overridden per the environment
variable ``INTERCEPTOR_CONFIG``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Interceptor curriculum training")
    p.add_argument("--config", default=os.environ.get("INTERCEPTOR_CONFIG", "/data/configs/default_run.yaml"))
    p.add_argument("--data-dir", default=None, help="override data_dir (host volume /data)")
    p.add_argument("--vecenv", default="auto", choices=["auto", "subproc", "dummy"])
    p.add_argument("--smoke", action="store_true", help="run the integration smoke test")
    p.add_argument("--smoke-config", default="ppo_baseline")
    p.add_argument("--smoke-stages", default="1,2,3")
    p.add_argument("--smoke-steps", type=int, default=1000)
    return p


def main(argv: object = None) -> int:
    args = build_parser().parse_args(argv)

    from src.utils import load_run_config
    from src.utils.logger import configure_file_logging, get_logger

    cfg = load_run_config(args.config)
    if args.data_dir:
        cfg.global_cfg["data_dir"] = args.data_dir
    if args.smoke:
        cfg.global_cfg["smoke"] = {
            "config": args.smoke_config,
            "stages": [int(s.strip()) for s in args.smoke_stages.split(",")],
            "steps_per_stage": max(1, int(args.smoke_steps)),
        }

    data_dir = Path(cfg.global_cfg["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    configure_file_logging(str(data_dir / "results" / "logs"))
    log = get_logger("train")
    log.info("config=%s data_dir=%s", cfg.source, data_dir)

    from src.training.orchestrator import Orchestrator

    orch = Orchestrator(cfg, data_dir=str(data_dir), vecenv=args.vecenv)
    summary = orch.run()
    log.info("=== TRAINING SUMMARY ===")
    log.info(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())