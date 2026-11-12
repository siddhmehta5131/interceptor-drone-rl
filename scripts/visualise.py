"""visualise.py -- generate demo videos for stages (plan §17.1 file 32).

Thin wrapper over :mod:`visualiser.cli`.  Examples::

    python scripts/visualise.py --model apn --stage all --output-dir data/videos
    python scripts/visualise.py --model data/runs/apn_rl_v1/ppo_interceptor/final.zip \\
        --stage 2 --output data/videos/stage2.mp4
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visualiser.cli import main as visualiser_main


def main(argv=None) -> int:
    return visualiser_main(argv)


if __name__ == "__main__":
    sys.exit(main())
