"""Initial exploration standard deviation of the residual arm A3 (BUGS #38).

PPO in Stable-Baselines3 starts with ``log_std_init = 0`` (std 1).  For the
residual arm that adds ``N(0, 1) x residual_authority x a_n_max`` to the APN
command at every step, so the behaviour policy starts far from the APN baseline
even though its mean starts at zero.  This script trains A3 with smaller initial
standard deviations and compares them, the std-1 policy and the zero-residual
baseline on VALIDATION seeds (20000 + k), which are disjoint from the training
seeds and from the evaluation seeds (10000 + k) of the result tables.

Selection rule (fixed before the runs): highest validation kill rate; ties are
broken by the lower median miss distance.

    python scripts/sweep_explore_std.py --train --workers 4
    python scripts/sweep_explore_std.py --evaluate --workers 4

Output: ``results/tables/explore_std_sweep.csv``.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]

SIGMAS = (0.3, 0.1)                 # std 1.0 is the existing p7_<platform>_a3_s0 run
PLATFORMS = ("quadcopter", "fixed_wing")
STEPS = 300_000                     # same budget as the std-1 runs it is compared with
VAL_SEED0 = 20000
VAL_EPISODES_PER_MANEUVER = 4


def run_id(platform: str, sigma: float) -> str:
    return f"p9e_{platform}_sigma{sigma:g}_s0"


def _train(job):
    platform, sigma, steps = job
    os.environ["OMP_NUM_THREADS"] = "1"
    import warnings
    import torch
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")
    from training.runner import train
    out = train(run_id(platform, sigma), {
        "task": "guidance", "algo": "ppo", "seed": 0, "total_steps": steps,
        "checkpoint_every": 25000, "asymmetric": True, "actor_obs_dim": 85,
        "log_std_init": math.log(sigma),
        "guidance": {"platform": platform, "mode": "apn_rl", "target_set": "evasive",
                     "noise_scale": 1.0}})
    return run_id(platform, sigma), int(out["num_timesteps"])


def _evaluate(job):
    platform, label, ckpt = job
    steps = 0
    if ckpt is not None:
        import json
        steps = int(json.loads((Path(ckpt).parent / "DONE").read_text())["num_timesteps"])
    os.environ["OMP_NUM_THREADS"] = "1"
    import warnings
    import torch
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")
    from interceptor.guidance.env import UnifiedInterceptEnv
    from interceptor.guidance.maneuvers import ManeuverType, PlatformType
    model = None
    if ckpt is not None:
        from stable_baselines3 import PPO
        model = PPO.load(str(ckpt), device="cpu")
    kills, misses = [], []
    maneuvers = list(ManeuverType)
    for i, man in enumerate(maneuvers):
        for k in range(VAL_EPISODES_PER_MANEUVER):
            seed = VAL_SEED0 + 100 * i + k
            env = UnifiedInterceptEnv(platform=PlatformType(platform), mode="apn_rl",
                                      target_set="evasive", fixed_maneuver=man, seed=seed)
            obs, _ = env.reset()
            done = False
            while not done:
                act = np.zeros(3) if model is None else model.predict(obs, deterministic=True)[0]
                obs, _, term, trunc, _ = env.step(act)
                done = term or trunc
            kills.append(bool(env.last_info["killed"]))
            misses.append(float(env.min_miss_distance))
    return {"platform": platform, "policy": label, "episodes": len(kills),
            "kills": int(np.sum(kills)), "kill_rate": float(np.mean(kills)),
            "median_miss": float(np.median(misses)),
            "seed_start": VAL_SEED0, "train_steps": steps}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--platforms", nargs="*", default=list(PLATFORMS))
    ap.add_argument("--sigmas", nargs="*", type=float, default=list(SIGMAS))
    ap.add_argument("--steps", type=int, default=STEPS,
                    help="training budget; a larger value resumes the same run")
    ap.add_argument("--out", default="explore_std_sweep.csv")
    args = ap.parse_args(argv)

    if args.train:
        jobs = [(p, s, args.steps) for p in args.platforms for s in args.sigmas]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for rid, n in ex.map(_train, jobs):
                print(rid, n, flush=True)
    if args.evaluate:
        jobs = []
        for p in args.platforms:
            jobs.append((p, "zero residual (= A2 command)", None))
            jobs.append((p, "sigma0 = 1 (SB3 default)", ROOT / "runs" / f"p7_{p}_a3_s0" / "checkpoint.zip"))
            for s in args.sigmas:
                jobs.append((p, f"sigma0 = {s:g}", ROOT / "runs" / run_id(p, s) / "checkpoint.zip"))
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            rows = list(ex.map(_evaluate, jobs))
        out = ROOT / "results" / "tables" / args.out
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        for r in rows:
            print(f"{r['platform']:11s} {r['policy']:30s} kills {r['kills']:3d}/{r['episodes']} "
                  f"({r['kill_rate']:.3f}) median miss {r['median_miss']:.2f} m")
        print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
