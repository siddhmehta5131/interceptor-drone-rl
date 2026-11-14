"""Throughput benchmark for training budgets (plan P7.2).

Measures environment steps per second with ``SubprocVecEnv`` for the
flight-controller and quad-guidance tasks.  The output is consumed by
``configs/training.yaml``; the Colab run should repeat it and replace the
laptop numbers, because budgets are hardware-dependent.

Usage::

    python scripts/benchmark_training.py --n-envs 4 --steps 2000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "tables" / "training_throughput.json"


def _make_fc(seed: int):
    from interceptor.fc.env import FlightControllerEnv
    return FlightControllerEnv(seed=seed, domain_randomization=True, wind=True)


def _make_guidance(seed: int):
    from interceptor.guidance.env import UnifiedInterceptEnv
    from interceptor.guidance.maneuvers import PlatformType
    return UnifiedInterceptEnv(platform=PlatformType.QUADCOPTER, mode="apn_rl",
                               target_set="evasive", N_nav=5.0, seed=seed)


def bench(name, factory, n_envs: int, steps: int) -> dict:
    from stable_baselines3.common.vec_env import SubprocVecEnv
    venv = SubprocVecEnv([partial(factory, i) for i in range(n_envs)])
    try:
        obs = venv.reset()
        action = np.zeros((n_envs, venv.action_space.shape[0]), dtype=np.float32)
        for _ in range(50):
            venv.step(action)
        start = time.perf_counter()
        for _ in range(steps):
            venv.step(action)
        elapsed = time.perf_counter() - start
    finally:
        venv.close()
    total = n_envs * steps
    return {"name": name, "n_envs": n_envs, "steps": total,
            "seconds": elapsed, "steps_per_s": total / elapsed}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args(argv)

    rows = [bench("fc", _make_fc, args.n_envs, args.steps),
            bench("guidance", _make_guidance, args.n_envs, args.steps)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"hardware": "local laptop (Colab rerun pending)",
                               "rows": rows}, indent=2), encoding="utf-8")
    for row in rows:
        print(f"{row['name']:9s} {row['steps_per_s']:10.1f} steps/s "
              f"({row['n_envs']} envs, {row['seconds']:.2f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
