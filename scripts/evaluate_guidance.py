"""Evaluate A1/A2 with the full protocol (plan P5.13).

One table per platform and target class:
  * evasive: all 16 manoeuvres (including ``NONE``);
  * non-evasive: static plus polynomial orders 1-3.

Every cell has 100 episodes with fixed evaluation seeds.  A1 (raw seeker) and
A2 (filtered seeker + axial term) share seeds, so the rows are paired.  Kill
rate carries a Wilson 95% interval; miss distance is median and IQR.

Usage::

    python scripts/evaluate_guidance.py --episodes 100 --workers 6
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interceptor.eval.stats import KillRate, median_iqr
from interceptor.guidance.config import resolve_n_prime
from interceptor.guidance.env import UnifiedInterceptEnv
from interceptor.guidance.maneuvers import ManeuverType, PlatformType

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "tables"
ARMS = ("apn_raw", "apn_filtered")
EVAL_SEED0 = 10000


def _run_cell(job) -> dict:
    platform_name, arm, target_set, behavior, episodes, seed0 = job
    platform = PlatformType(platform_name)
    n_prime = resolve_n_prime(platform, arm)
    kills, misses, times = [], [], []
    for k in range(episodes):
        seed = int(seed0 + k)
        kwargs = {}
        if target_set == "evasive":
            kwargs["fixed_maneuver"] = ManeuverType(behavior)
        else:
            kwargs["polynomial_order"] = int(behavior) if behavior != "static" else 1
        env = UnifiedInterceptEnv(
            platform=platform, mode=arm, target_set=target_set, noise_scale=1.0,
            N_nav=n_prime, seed=seed, **kwargs)
        env.reset()
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(np.zeros(3))
            done = terminated or truncated
        kills.append(bool(env.last_info["killed"]))
        misses.append(float(env.min_miss_distance))
        times.append(float(env.step_count * env.dt))
    kr = KillRate.from_outcomes(kills)
    mi = median_iqr(misses)
    return {
        "platform": platform_name, "arm": arm, "target_set": target_set,
        "behavior": behavior, "N_prime": n_prime, "episodes": episodes,
        "kills": kr.kills, "kill_rate": kr.rate,
        "kill_lo": kr.lo, "kill_hi": kr.hi,
        "median_miss": mi["median"], "iqr_miss": mi["iqr"],
        "median_time": float(np.median(times)), "seed_start": seed0,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--platforms", nargs="*", default=[p.value for p in PlatformType])
    parser.add_argument("--seed", type=int, default=EVAL_SEED0)
    args = parser.parse_args(argv)

    jobs = []
    for p in args.platforms:
        for arm in ARMS:
            for maneuver in ManeuverType:
                jobs.append((p, arm, "evasive", maneuver.value, args.episodes, args.seed))
            for behavior in ("static", "1", "2", "3"):
                jobs.append((p, arm, "polynomial" if behavior != "static" else "static",
                             behavior, args.episodes, args.seed))

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = []
        for i, row in enumerate(pool.map(_run_cell, jobs), 1):
            rows.append(row)
            if i % 10 == 0 or i == len(jobs):
                print(f"completed {i}/{len(jobs)} cells", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    for platform in args.platforms:
        for target_set, suffix in (("evasive", "evasive"), ("non", "nonevasive")):
            subset = [r for r in rows if r["platform"] == platform and
                      (r["target_set"] == "evasive") == (target_set == "evasive")]
            path = OUT / f"guidance_{platform}_{suffix}.csv"
            with path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                writer.writerows(subset)
            print(f"wrote {path} ({len(subset)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
