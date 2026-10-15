"""``python -m stage_runtime`` -- render one or more stages to video.

Examples
--------
Render the whole shipped set (one MP4 per stage) into ``./out``::

    python -m stage_runtime --all --out ./out

Render a single stage, or a range of them::

    python -m stage_runtime --stage 3 --out ./out
    python -m stage_runtime --stages 5,6,7 --out ./out

Drive a stage from an uploaded checkpoint (bind-mounted, or a host path)::

    python -m stage_runtime --stage 8 --model /data/checkpoints/ppo_final.zip \\
        --algo PPO --out ./out

Fly a stage without writing any video (sanity check / CI)::

    python -m stage_runtime --all --no-render --out ./out
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Sequence

from . import __version__, bootstrap_path

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "stage"


def _stage_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stage", type=int, default=None,
                        help="single stage number (1..8)")
    parser.add_argument("--stages", type=str, default=None,
                        help="comma-separated stage numbers, e.g. 1,2,3")
    parser.add_argument("--all", action="store_true",
                        help="render every stage_*.json found in --config-dir")
    parser.add_argument("--config-dir", type=Path, default=None,
                        help=f"directory of stage JSONs (default: {DEFAULT_CONFIG_DIR})")
    parser.add_argument("--config", type=Path, action="append", default=[],
                        help="explicit stage JSON file (repeatable)")


def _output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", "-o", type=Path,
                        default=Path(os.environ.get("STAGE_OUT_DIR", "./out")),
                        help="output directory (env STAGE_OUT_DIR)")
    parser.add_argument("--format", dest="fmt", default=None,
                        choices=("mp4", "gif", "png"),
                        help="override the video format from the JSON")
    parser.add_argument("--fps", type=int, default=None, help="override video fps")
    parser.add_argument("--segments", type=int, default=None,
                        help="override the number of segments per stage")
    parser.add_argument("--segment-seconds", type=float, default=None,
                        help="override the seconds per segment (default 3)")
    parser.add_argument("--no-render", action="store_true",
                        help="fly the stage(s) but write no video/frames")
    parser.add_argument("--no-sidecar", action="store_true",
                        help="do not write the per-stage JSON report")
    parser.add_argument("--quiet", "-q", action="store_true")


def _control_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", type=Path, default=None,
                        help="checkpoint to fly the stage with "
                             "(overrides the JSON; implies control=model)")
    parser.add_argument("--algo", type=str, default=None,
                        help="algorithm of --model (PPO, SAC, TD3, ...)")
    parser.add_argument("--data-root", type=Path, action="append", default=[],
                        help="extra directory searched for --model "
                             "(env STAGE_DATA_ROOTS, colon separated)")
    parser.add_argument("--stochastic", action="store_true",
                        help="sample actions instead of deterministic ones")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m stage_runtime",
        description="Stage-mode runtime: fly one curriculum stage with a fixed "
                    "instruction set or with an uploaded checkpoint, and render "
                    "the result to video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version",
                        version=f"stage_runtime {__version__}")
    _stage_args(parser)
    _output_args(parser)
    _control_args(parser)
    return parser


def resolve_configs(args: argparse.Namespace) -> List[Path]:
    if args.config:
        return [Path(c).expanduser() for c in args.config]
    config_dir = Path(args.config_dir) if args.config_dir else DEFAULT_CONFIG_DIR
    if args.stage is not None:
        candidate = config_dir / f"stage_{args.stage:02d}.json"
        if candidate.is_file():
            return [candidate]
        matches = sorted(config_dir.glob(f"stage_{args.stage:02d}*.json"))
        if matches:
            return [matches[0]]
        raise SystemExit(f"no stage config for stage {args.stage} in {config_dir}")
    if args.stages:
        wanted = [int(s) for s in str(args.stages).replace(" ", "").split(",") if s]
        out: List[Path] = []
        for stage in wanted:
            candidate = config_dir / f"stage_{stage:02d}.json"
            if not candidate.is_file():
                raise SystemExit(f"missing stage config {candidate}")
            out.append(candidate)
        return out
    from .stage_config import discover_stage_configs
    return discover_stage_configs(config_dir)


def apply_overrides(cfg, args: argparse.Namespace):
    """Apply CLI overrides on top of the stage JSON."""
    from dataclasses import replace

    video = cfg.video
    if args.fmt:
        video = replace(video, fmt=args.fmt)
    if args.fps:
        video = replace(video, fps=int(args.fps))
    control = cfg.control
    model = cfg.model
    if args.model:
        control = "model"
        model = replace(model, path=Path(args.model), algo=(args.algo or model.algo))
    if args.algo:
        model = replace(model, algo=args.algo)
    if args.stochastic:
        model = replace(model, deterministic=False)
    out = replace(cfg, video=video, control=control, model=model)
    if args.segments:
        out = replace(out, segments=int(args.segments))
    if args.segment_seconds:
        out = replace(out, segment_seconds=float(args.segment_seconds))
    return out


def data_roots(args: argparse.Namespace) -> List[Path]:
    roots = [Path(r) for r in args.data_root]
    env = os.environ.get("STAGE_DATA_ROOTS", "")
    roots += [Path(p) for p in env.split(os.pathsep) if p]
    return roots


def main(argv: Sequence[str] | None = None) -> int:
    bootstrap_path()
    args = build_parser().parse_args(argv)
    try:
        configs = resolve_configs(args)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    from .runner import RunError, run_stage
    from .stage_config import load_stage_config

    roots = data_roots(args)
    out_dir = Path(args.out).expanduser()
    reports = []
    failures = []
    for path in configs:
        try:
            cfg = apply_overrides(load_stage_config(path), args)
            report = run_stage(cfg, out_dir=out_dir, data_roots=roots,
                               render=not args.no_render,
                               write_sidecar=not args.no_sidecar,
                               verbose=not args.quiet)
            reports.append(report)
            if not args.quiet:
                _print_summary(report)
        except (RunError, ValueError, OSError) as exc:
            failures.append((path, str(exc)))
            print(f"error: {path.name}: {exc}", file=sys.stderr)
    if args.no_render:
        print(f"flew {len(reports)} stage(s), no video written (--no-render)")
    elif len(reports) > 1:
        total = sum(r["video"].get("bytes", 0) for r in reports)
        print(f"rendered {len(reports)} stage(s), {total} bytes total")
    return 1 if failures else 0


def _print_summary(report: dict) -> None:
    video = report["video"]
    print(f"  -> {video['path']}  ({video['format']}, {video['frames']} frames, "
          f"{video['duration_s']} s, {video.get('bytes', 0)} bytes)")
    if report.get("sidecar"):
        print(f"  -> {report['sidecar']}")


if __name__ == "__main__":
    sys.exit(main())