"""Closed-loop evaluation on the quad (plan P8.6).

Arms A2 (APN + axial) and A3 (APN + residual RL) are run through the bridge,
the inner rate loop and the 1000 Hz physics on the same evaluation seeds as the
kinematic quad runs.  With no trained policy, A3 is skipped and recorded as
pending; the RL inner controller is likewise optional.

Usage::

    python scripts/evaluate_closed_loop.py --episodes 100 --workers 6 \
        --guidance-policy runs/g3/checkpoint.zip --inner-policy runs/fc/checkpoint.zip
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interceptor.bridge.closed_loop import run_closed_loop
from interceptor.eval.stats import KillRate, median_iqr
from interceptor.guidance.config import resolve_n_prime
from interceptor.guidance.maneuvers import ManeuverType, PlatformType

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "tables"
EVAL_SEED0 = 10000


def _run(job) -> dict:
    arm, inner, seed, maneuver, policy_path, inner_policy_path = job
    policy = None
    if policy_path:
        from stable_baselines3 import PPO
        policy = PPO.load(policy_path, device="cpu")
    inner_policy = None
    if inner_policy_path:
        from stable_baselines3 import PPO
        inner_policy = PPO.load(inner_policy_path, device="cpu")
    r = run_closed_loop(
        platform=PlatformType.QUADCOPTER, mode=arm, seed=seed, inner=inner, policy=policy,
        inner_policy=inner_policy,
        target_set="evasive", fixed_maneuver=ManeuverType(maneuver),
        N_nav=resolve_n_prime("quadcopter", arm), wind=True)
    return {"arm": arm, "inner": inner, "seed": seed, "maneuver": maneuver,
            "killed": r.killed, "miss": r.miss_distance, "time": r.time,
            "accel_rms": r.accel_tracking_rms}


def _aggregate(rows: list[dict], arm: str, inner: str) -> dict:
    outcomes = [r["killed"] for r in rows]
    kr = KillRate.from_outcomes(outcomes)
    mi = median_iqr([r["miss"] for r in rows])
    return {"platform": "quadcopter", "arm": arm, "inner": inner,
            "episodes": len(rows), "kills": kr.kills, "kill_rate": kr.rate,
            "kill_lo": kr.lo, "kill_hi": kr.hi,
            "median_miss": mi["median"], "iqr_miss": mi["iqr"],
            "median_time": float(np.median([r["time"] for r in rows])),
            "accel_rms": float(np.median([r["accel_rms"] for r in rows]))}


def _plot(rows: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [f"{r['arm']}\n{r['inner']}" for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(x - .2, [r["kill_rate"] for r in rows], .4, label="physics")
    axes[0].bar(x + .2, [r["kinematic_kill_rate"] for r in rows], .4, label="kinematic")
    axes[0].set(ylabel="kill rate", title="Kill rate", xticks=x, xticklabels=labels)
    axes[1].bar(x - .2, [r["median_miss"] for r in rows], .4, label="physics")
    axes[1].bar(x + .2, [r["kinematic_median_miss"] for r in rows], .4, label="kinematic")
    axes[1].set(ylabel="median miss (m)", title="Miss distance", xticks=x, xticklabels=labels)
    for ax in axes:
        ax.grid(axis="y", alpha=.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _kinematic_reference() -> dict:
    path = OUT / "guidance_quadcopter_evasive.csv"
    if not path.exists():
        return {}
    out = {}
    with path.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.setdefault(row["arm"], {"kills": 0, "episodes": 0, "miss": []})
            out[row["arm"]]["kills"] += int(row["kills"])
            out[row["arm"]]["episodes"] += int(row["episodes"])
            out[row["arm"]]["miss"].append(float(row["median_miss"]))
    return {k: {"kill_rate": v["kills"] / max(v["episodes"], 1),
                "median_miss": float(np.median(v["miss"]))}
            for k, v in out.items()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--inner", default="b1", choices=["b1", "b2", "rl"])
    parser.add_argument("--guidance-policy", default=None)
    parser.add_argument("--inner-policy", default=None)
    args = parser.parse_args(argv)

    maneuvers = list(ManeuverType)
    jobs = []
    arms = ["apn_filtered"]
    if args.guidance_policy:
        arms.append("apn_rl")
    for arm in arms:
        for k in range(args.episodes):
            jobs.append((arm, args.inner, EVAL_SEED0 + k,
                         maneuvers[k % len(maneuvers)].value,
                         args.guidance_policy, args.inner_policy))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_run, jobs))

    ref = _kinematic_reference()
    out_rows = []
    for arm in arms:
        agg = _aggregate([r for r in rows if r["arm"] == arm], arm, args.inner)
        kin = ref.get(arm, {})
        agg["kinematic_kill_rate"] = kin.get("kill_rate", float("nan"))
        agg["kinematic_median_miss"] = kin.get("median_miss", float("nan"))
        out_rows.append(agg)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "closed_loop.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)
    for row in out_rows:
        print(row)
    _plot(out_rows, OUT.parent / "figures" / "closed_loop.png")
    if not args.guidance_policy:
        with (OUT / "closed_loop_pending.csv").open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows([
                ["arm", "status", "reason"],
                ["apn_rl (A3)", "pending", "no guidance policy checkpoint"],
                ["pure_rl (A4)", "pending", "no guidance policy checkpoint"],
            ])
    return 0


if __name__ == "__main__":
    sys.exit(main())
