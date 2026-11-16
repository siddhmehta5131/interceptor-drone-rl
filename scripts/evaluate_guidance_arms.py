"""Evaluate the learned guidance arms A3/A4 (plan P8.3).

A1/A2 are covered by ``evaluate_guidance.py``.  This script runs the RL arms on
the same evaluation seeds with a checkpoint per (arm, platform):

    python scripts/evaluate_guidance_arms.py \
        --policy a3_quadcopter=runs/p7_quadcopter_a3_s0/checkpoint.zip \
        --policy a4_quadcopter=runs/p7_quadcopter_a4_s0/checkpoint.zip

Cells without a checkpoint are written to ``guidance_rl_pending.csv`` instead
of being reported as results.
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

from interceptor.eval.stats import KillRate, median_iqr
from interceptor.guidance.env import UnifiedInterceptEnv
from interceptor.guidance.maneuvers import ManeuverType, PlatformType

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "tables"
EVAL_SEED0 = 10000


def _run_cell(job) -> dict:
    platform, arm, mode, target_set, behavior, episodes, seed0, policy_path = job
    from stable_baselines3 import PPO
    policy = PPO.load(policy_path, device="cpu")
    # Evaluate in the environment the policy was trained in: a run trained with a
    # non-default residual authority (P9.3) records it in its config.json.
    extra = {}
    cfg_path = Path(policy_path).parent / "config.json"
    if cfg_path.exists():
        import json
        auth = (json.loads(cfg_path.read_text(encoding="utf-8")).get("guidance") or {}).get("residual_authority")
        if auth is not None and mode == "apn_rl":
            extra["residual_authority"] = float(auth)
    kills, misses, times = [], [], []
    for k in range(episodes):
        kwargs = {"fixed_maneuver": ManeuverType(behavior)} if target_set == "evasive" else {}
        if target_set != "evasive":
            kwargs["polynomial_order"] = int(behavior) if behavior != "static" else 1
        env = UnifiedInterceptEnv(platform=PlatformType(platform), mode=mode,
                                  target_set=target_set, noise_scale=1.0,
                                  seed=seed0 + k, **kwargs, **extra)
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        kills.append(bool(env.last_info["killed"]))
        misses.append(float(env.min_miss_distance))
        times.append(float(env.step_count * env.dt))
    kr = KillRate.from_outcomes(kills)
    mi = median_iqr(misses)
    return {"platform": platform, "arm": arm, "target_set": target_set,
            "behavior": behavior, "episodes": episodes, "kills": kr.kills,
            "kill_rate": kr.rate, "kill_lo": kr.lo, "kill_hi": kr.hi,
            "median_miss": mi["median"], "iqr_miss": mi["iqr"],
            "median_time": float(np.median(times)), "seed_start": seed0}


def _parse(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("use arm_platform=path")
    name, path = spec.split("=", 1)
    return name, Path(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", action="append", type=_parse, default=[])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=EVAL_SEED0)
    args = parser.parse_args(argv)

    jobs, pending = [], []
    for name, path in args.policy:
        arm, platform = name.split("_", 1)
        if not path.exists():
            pending.append({"arm": arm, "platform": platform, "reason": f"missing {path}"})
            continue
        mode = "apn_rl" if arm == "a3" else "pure_rl"
        for maneuver in ManeuverType:
            jobs.append((platform, arm, mode, "evasive", maneuver.value,
                         args.episodes, args.seed, str(path)))
        for behavior in ("static", "1", "2", "3"):
            ts = "static" if behavior == "static" else "polynomial"
            jobs.append((platform, arm, mode, ts, behavior, args.episodes,
                         args.seed, str(path)))

    rows = []
    if jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(_run_cell, jobs))

    OUT.mkdir(parents=True, exist_ok=True)
    if rows:
        path = OUT / "guidance_rl_arms.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path} ({len(rows)} rows)")
    if pending:
        with (OUT / "guidance_rl_pending.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["arm", "platform", "reason"])
            writer.writeheader()
            writer.writerows(pending)
        print(json.dumps(pending, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
