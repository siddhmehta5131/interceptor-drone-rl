"""Bridge body-rate-limit sweep on the closed-loop quad (evaluation only).

The acceleration bridge clips the commanded body rate at ``rate_limit``
(nominal 12 rad/s, a PROJECT_CHOICE).  This sweep runs the closed loop
(guidance -> bridge -> B1 inner rate loop -> 1000 Hz physics, Dryden wind on)
at the nominal limit and at 1/2, 1/4 and 1/8 of it on the same evaluation seeds,
so the sensitivity of the headline closed-loop result to the choice is reported.

Only lower values are swept because higher ones cannot act: the rate command is
``2 Kp q_err`` (EQ-BRIDGE-4) with Kp = 6 1/s and |q_err| <= 1, so it never
exceeds 12 rad/s.  A limit of 24 rad/s gives bit-identical episodes to 12
(checked 2026-11-15), and 6 rad/s binds only above a 60 deg attitude error.

    python scripts/sweep_bridge_rate.py --episodes 5 --workers 2
    python scripts/sweep_bridge_rate.py --guidance-policy runs/p7_quadcopter_a3_s0/checkpoint.zip

``--episodes`` is episodes per grid value per arm.  A2 (apn_filtered) always
runs; A3 (apn_rl) runs when ``--guidance-policy`` is given.  Writes
results/tables/sweep_bridge_rate.csv and results/figures/sweep_bridge_rate.png.
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from interceptor.bridge.closed_loop import run_closed_loop
from interceptor.eval.stats import KillRate, median_iqr
from interceptor.guidance.config import resolve_n_prime
from interceptor.guidance.maneuvers import ManeuverType, PlatformType

ROOT = Path(__file__).resolve().parents[1]
EVAL_SEED0 = 10000                      # same seeds as scripts/evaluate_closed_loop.py
RATE_LIMITS = (1.5, 3.0, 6.0, 12.0)     # rad/s: x1/8, x1/4, x1/2, nominal (= 2 Kp, the law's maximum)


def _run(job) -> dict:
    arm, rate, seed, maneuver, policy_path = job
    policy = None
    if policy_path:
        from stable_baselines3 import PPO
        policy = PPO.load(policy_path, device="cpu")
    r = run_closed_loop(
        platform=PlatformType.QUADCOPTER, mode=arm, seed=seed, inner="b1", policy=policy,
        target_set="evasive", fixed_maneuver=ManeuverType(maneuver),
        N_nav=resolve_n_prime("quadcopter", arm), wind=True, bridge_rate_limit=rate)
    return {"arm": arm, "rate": rate, "killed": r.killed, "miss": r.miss_distance,
            "time": r.time, "accel_rms": r.accel_tracking_rms}


def _aggregate(rows: list[dict], arm: str, rate: float) -> dict:
    kr = KillRate.from_outcomes([r["killed"] for r in rows])
    mi = median_iqr([r["miss"] for r in rows])
    return {"platform": "quadcopter", "arm": arm, "inner": "b1",
            "bridge_rate_limit": rate, "episodes": len(rows), "kills": kr.kills,
            "kill_rate": kr.rate, "kill_lo": kr.lo, "kill_hi": kr.hi,
            "median_miss": mi["median"], "iqr_miss": mi["iqr"],
            "median_time": float(np.median([r["time"] for r in rows])),
            "accel_rms": float(np.median([r["accel_rms"] for r in rows]))}


def _plot(rows: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    for arm in dict.fromkeys(r["arm"] for r in rows):
        rr = [r for r in rows if r["arm"] == arm]
        x = [r["bridge_rate_limit"] for r in rr]
        ax[0].errorbar(x, [r["kill_rate"] for r in rr],
                       yerr=[[r["kill_rate"] - r["kill_lo"] for r in rr],
                             [r["kill_hi"] - r["kill_rate"] for r in rr]],
                       marker="o", label=arm)
        ax[1].plot(x, [r["median_miss"] for r in rr], marker="o", label=arm)
        ax[2].plot(x, [r["accel_rms"] for r in rr], marker="o", label=arm)
    ax[0].set(ylabel="kill rate (95% Wilson)", ylim=(0, 1.02))
    ax[1].set(ylabel="median miss [m]")
    ax[2].set(ylabel="median accel tracking RMS [m/s^2]")
    for a in ax:
        a.set(xlabel="bridge rate limit [rad/s]", xscale="log", xticks=list(RATE_LIMITS),
              xticklabels=[f"{v:g}" for v in RATE_LIMITS])
        a.grid(alpha=.25)
    ax[0].legend()
    fig.suptitle("Closed loop (quad, B1 inner loop): bridge rate-limit sensitivity")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=5, help="episodes per grid value per arm")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--guidance-policy", default=None, help="checkpoint for arm A3 (apn_rl)")
    ap.add_argument("--rates", type=float, nargs="*", default=list(RATE_LIMITS))
    args = ap.parse_args(argv)

    maneuvers = list(ManeuverType)
    arms = [("apn_filtered", None)]
    if args.guidance_policy:
        arms.append(("apn_rl", args.guidance_policy))
    jobs = [(arm, rate, EVAL_SEED0 + k, maneuvers[k % len(maneuvers)].value, pol)
            for arm, pol in arms for rate in args.rates for k in range(args.episodes)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_run, jobs))

    out_rows = [_aggregate([r for r in rows if r["arm"] == arm and r["rate"] == rate], arm, rate)
                for arm, _ in arms for rate in args.rates]
    tables = ROOT / "results" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    with (tables / "sweep_bridge_rate.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    for row in out_rows:
        print(row)
    _plot(out_rows, ROOT / "results" / "figures" / "sweep_bridge_rate.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
