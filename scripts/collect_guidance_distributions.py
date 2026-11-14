"""Per-episode miss-distance samples for P8.3 distribution plots.

Runs the same seeds as ``evaluate_guidance.py`` (10000 + k) for the evasive
set and writes raw samples, so figures can show empirical distributions rather
than only medians.  Default: quadcopter, A1 and A2.

    python scripts/collect_guidance_distributions.py --workers 6
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interceptor.guidance.config import resolve_n_prime
from interceptor.guidance.env import UnifiedInterceptEnv
from interceptor.guidance.maneuvers import ManeuverType, PlatformType

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "tables"
EVAL_SEED0 = 10000


def _run(job):
    platform, arm, behavior, episodes, seed0 = job
    out = []
    for k in range(episodes):
        env = UnifiedInterceptEnv(
            platform=PlatformType(platform), mode=arm, target_set="evasive",
            fixed_maneuver=ManeuverType(behavior), noise_scale=1.0,
            N_nav=resolve_n_prime(platform, arm), seed=seed0 + k)
        env.reset()
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(np.zeros(3))
            done = terminated or truncated
        out.append((platform, arm, behavior, seed0 + k, k,
                    bool(env.last_info["killed"]), float(env.min_miss_distance),
                    float(env.step_count * env.dt)))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platforms", nargs="*", default=["quadcopter"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)
    jobs = [(p, arm, m.value, args.episodes, EVAL_SEED0)
            for p in args.platforms for arm in ("apn_raw", "apn_filtered")
            for m in ManeuverType]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for chunk in pool.map(_run, jobs):
            rows.extend(chunk)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "guidance_distributions.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["platform", "arm", "behavior", "seed", "episode",
                         "killed", "miss", "time"])
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} episodes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
