"""Single-process simulator throughput (plan P3.10).

Reports control steps per second at the legacy 100 Hz control rate and at the
S8 rates (500 Hz controller, 1000 Hz physics).  The guidance environment is
benchmarked separately in P5.10.

Usage: ``python scripts/benchmark_sim.py [--seconds 2.0]``
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interceptor.sim import constants, pipeline  # noqa: E402

CMD = np.array([0.45, 0.15, -0.08, 0.04])


def bench(control_dt: float, physics_dt: float, seconds: float) -> float:
    params = pipeline.PhysicsParams()
    state = (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3),
             np.zeros(3), np.full(4, constants.PH_OMEGA_HOVER))
    n = int(round(seconds / control_dt))
    start = time.perf_counter()
    for _ in range(n):
        state = pipeline.control_step(state, CMD, 4.2, params,
                                      control_dt=control_dt,
                                      physics_dt=physics_dt)
    elapsed = time.perf_counter() - start
    return n / elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=2.0)
    args = parser.parse_args()

    rows = [
        ("legacy 100 Hz control", 0.01, 0.01),
        ("S8 500 Hz control / 1000 Hz physics", 0.002, 0.001),
        ("1000 Hz control / 1000 Hz physics", 0.001, 0.001),
    ]
    print(f"{'configuration':44s} {'control steps/s':>16s} {'physics steps/s':>16s}")
    for name, cdt, pdt in rows:
        rate = bench(cdt, pdt, args.seconds)
        print(f"{name:44s} {rate:16.0f} {rate * cdt / pdt:16.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
