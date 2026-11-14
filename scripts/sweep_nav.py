"""Sweep the navigation constant N' for the classical arms (plan P5.11).

For every platform and arm (A1 raw, A2 filtered) the admissible range
3 <= N' <= 5 is swept on the evasive target set.  The winner is chosen by kill
rate first and median miss distance second, and written to
``results/tables/nav_sweep_<platform>.csv``.

Usage::

    python scripts/sweep_nav.py --episodes 100 --workers 4
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interceptor.guidance.env import UnifiedInterceptEnv
from interceptor.guidance.maneuvers import ManeuverType, PlatformType

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "tables"
N_GRID = (3.0, 3.5, 4.0, 4.5, 5.0)


def _run_cell(job) -> dict:
    platform_name, mode, n_nav, episodes, seed0 = job
    platform = PlatformType(platform_name)
    maneuvers = list(ManeuverType)
    kills: list[bool] = []
    misses: list[float] = []
    times: list[float] = []
    for k in range(episodes):
        seed = int(seed0 + k)
        env = UnifiedInterceptEnv(
            platform=platform, mode=mode, target_set="evasive",
            fixed_maneuver=maneuvers[k % len(maneuvers)],
            N_nav=float(n_nav), seed=seed)
        env.reset()
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(np.zeros(3))
            done = terminated or truncated
        kills.append(bool(env.last_info["killed"]))
        misses.append(float(env.min_miss_distance))
        times.append(float(env.step_count * env.dt))
    return {
        "platform": platform_name, "mode": mode, "N_prime": float(n_nav),
        "episodes": episodes, "kill_rate": float(np.mean(kills)),
        "median_miss": float(np.median(misses)),
        "iqr_miss": float(np.percentile(misses, 75) - np.percentile(misses, 25)),
        "median_time": float(np.median(times)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=5000)
    args = parser.parse_args(argv)

    jobs = [(p.value, mode, n, args.episodes, args.seed)
            for p in PlatformType for mode in ("apn_raw", "apn_filtered")
            for n in N_GRID]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_run_cell, jobs))

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "nav_sweep.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    best = {}
    for platform in PlatformType:
        best[platform.value] = {}
        for mode in ("apn_raw", "apn_filtered"):
            cells = [r for r in rows
                     if r["platform"] == platform.value and r["mode"] == mode]
            cells.sort(key=lambda r: (-r["kill_rate"], r["median_miss"]))
            best[platform.value][mode] = cells[0]

    (OUT / "nav_sweep_best.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8")
    for platform, arms in best.items():
        for mode, row in arms.items():
            print(f"{platform:11s} {mode:12s} N'={row['N_prime']:.1f} "
                  f"kill={row['kill_rate']:.3f} median_miss={row['median_miss']:.2f} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
