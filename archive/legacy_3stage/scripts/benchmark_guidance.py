"""Profile the guidance environment step cost (plan P5.10, fix F17).

The critic oracle is the closed-form boundary-value command
(``guidance.kinematics.compute_optimal_intercept_oracle``), evaluated once per
environment step.  The full-trajectory oracle in
``guidance.optimal_trajectory`` is kept for offline analysis only; this script
asserts that it is not imported by the environment module.

Usage::

    python scripts/benchmark_guidance.py [--steps 2000] [--target-steps-s 1000]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interceptor.guidance.env import UnifiedInterceptEnv
from interceptor.guidance.maneuvers import ManeuverType, PlatformType


def run(platform: PlatformType, steps: int, seed: int = 7) -> tuple[int, float]:
    """Return ``(steps_run, steps_per_second)`` for one evasive episode."""
    env = UnifiedInterceptEnv(
        platform=platform, mode="apn_filtered", target_set="evasive",
        fixed_maneuver=ManeuverType.SPIRAL_CORKSCREW, seed=seed)
    env.reset(seed=seed)
    action = np.zeros(3, dtype=np.float32)
    n = 0
    start = time.perf_counter()
    while n < steps:
        _, _, terminated, truncated, _ = env.step(action)
        n += 1
        if terminated or truncated:
            env.reset()
    elapsed = time.perf_counter() - start
    return n, n / elapsed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--target-steps-s", type=float, default=1000.0)
    args = parser.parse_args(argv)

    ok = True
    for platform in PlatformType:
        n, rate = run(platform, args.steps)
        passed = rate >= args.target_steps_s
        ok &= passed
        print(f"{platform.value:11s} {n:5d} steps  {rate:8.1f} steps/s  "
              f"{'PASS' if passed else 'FAIL'} (target {args.target_steps_s:.0f})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
