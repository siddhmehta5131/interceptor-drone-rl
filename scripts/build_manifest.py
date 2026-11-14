"""Build and verify ``results/manifest.json`` (plan P7.5).

Usage::

    python scripts/build_manifest.py --runs runs --output results/manifest.json --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.manifest import build_manifest, verify_manifest

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default=str(ROOT / "runs"))
    parser.add_argument("--output", default=str(ROOT / "results" / "manifest.json"))
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--load", action="store_true", help="also load each checkpoint")
    args = parser.parse_args(argv)

    manifest = build_manifest(Path(args.runs), Path(args.output))
    print(f"wrote {args.output} with {len(manifest['runs'])} runs")
    if args.verify:
        problems = verify_manifest(manifest, Path(args.runs), load=args.load)
        if problems:
            for p in problems:
                print("FAIL", p)
            return 1
        print("manifest verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
