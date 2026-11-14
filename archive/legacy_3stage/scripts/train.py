"""train.py -- training entry point (plan §17.1 file 30).

Thin CLI over :func:`interceptor.api.train_model`.  Example::

    python scripts/train.py --stages 1,2,3 --name apn_rl_v1
    python scripts/train.py --stages 1 --name smoke --steps 20000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interceptor.api import train_model


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train the APN+RL interceptor policy.")
    parser.add_argument("--model", default="configs/model.yaml", help="model definition YAML")
    parser.add_argument("--config", default="configs/config.yaml", help="training settings YAML")
    parser.add_argument("--stages", default="1,2,3", help="comma-separated stage numbers")
    parser.add_argument("--name", default="apn_rl_v1", help="run name under <data_dir>")
    parser.add_argument("--steps", type=int, default=None, help="cap steps per stage (smoke tests)")
    parser.add_argument("--data-dir", default=None, help="override global.data_dir")
    parser.add_argument("--device", default=None, help="override global.device (auto/cpu/cuda)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    stages = [int(c) for c in str(args.stages).replace(" ", "").split(",") if c]
    result = train_model(
        source=args.model,
        stages=stages,
        name=args.name,
        config=args.config,
        data_dir=args.data_dir,
        device=args.device,
        max_steps_per_stage=args.steps,
        verbose=0 if args.quiet else 1,
    )
    print(result.summary())
    return 0 if result.status == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
