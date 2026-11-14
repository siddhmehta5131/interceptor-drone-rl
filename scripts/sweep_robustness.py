"""P9.1 robustness sweeps and P9.6 kill-radius sensitivity (evaluation only).

    python scripts/sweep_robustness.py --episodes 5 --workers 4
    python scripts/sweep_robustness.py --policy quadcopter=runs/p7_quadcopter_a3_s0/checkpoint.zip

A2 (apn_filtered) always runs; A3 (apn_rl) runs for platforms given a --policy.
Writes results/tables/sweep_<sweep>.csv, results/figures/sweep_<sweep>_<platform>.png
and results/figures/killradius_<platform>.png (from the nominal-noise miss distances).
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from interceptor.eval import sweeps
from interceptor.guidance.maneuvers import PlatformType
from interceptor.guidance.platform import PLATFORMS

ROOT = Path(__file__).resolve().parents[1]


def _job(a):
    sweep, platform, mode, episodes, policy_path = a
    policy = None
    if policy_path:
        from stable_baselines3 import PPO
        policy = PPO.load(policy_path, device="cpu")
    return sweeps.run_sweep(sweep, PlatformType(platform), mode, episodes=episodes, policy=policy)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=5, help="episodes per behaviour per grid value")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--policy", action="append", default=[], help="platform=path for arm A3")
    ap.add_argument("--platforms", nargs="*", default=[p.value for p in PlatformType])
    args = ap.parse_args(argv)
    pol = dict(p.split("=", 1) for p in args.policy)
    jobs = []
    for sweep in sweeps.SWEEPS:
        for plat in args.platforms:
            jobs.append((sweep, plat, "apn_filtered", args.episodes, None))
            if plat in pol:
                jobs.append((sweep, plat, "apn_rl", args.episodes, pol[plat]))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(_job, jobs))

    tables, figs = ROOT / "results" / "tables", ROOT / "results" / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for sweep, (label, _) in sweeps.SWEEPS.items():
        rows = [r for j, res in zip(jobs, results) if j[0] == sweep for r in res]
        with (tables / f"sweep_{sweep}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=[k for k in rows[0] if k != "misses"])
            w.writeheader()
            w.writerows({k: v for k, v in r.items() if k != "misses"} for r in rows)
        for plat in args.platforms:
            fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
            for mode in ("apn_filtered", "apn_rl"):
                rr = [r for r in rows if r["platform"] == plat and r["mode"] == mode]
                if not rr:
                    continue
                x = [r["value"] for r in rr]
                ax[0].errorbar(x, [r["kill_rate"] for r in rr],
                               yerr=[[r["kill_rate"] - r["kill_lo"] for r in rr],
                                     [r["kill_hi"] - r["kill_rate"] for r in rr]],
                               marker="o", label=mode)
                ax[1].plot(x, [r["median_miss"] for r in rr], marker="o", label=mode)
            ax[0].set(xlabel=label, ylabel="kill rate (95% Wilson)", ylim=(0, 1.02))
            ax[1].set(xlabel=label, ylabel="median miss [m]")
            ax[0].legend()
            fig.suptitle(f"{plat}: {sweep}")
            fig.tight_layout()
            fig.savefig(figs / f"sweep_{sweep}_{plat}.png", dpi=130)
            plt.close(fig)

    # P9.6: kill-radius sensitivity from the nominal-noise miss distances
    for plat in args.platforms:
        nominal = [r for j, res in zip(jobs, results) if j[0] == "noise_scale" and j[1] == plat
                   for r in res if r["value"] == 1.0]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        k = PLATFORMS[PlatformType(plat)].kill_radius
        radii = np.linspace(0.25 * k, 3 * k, 25)
        for r in nominal:
            kr = sweeps.kill_rate_vs_radius(r["misses"], radii)
            ax.plot(radii, [q["kill_rate"] for q in kr], label=r["mode"])
        ax.axvline(k, color="k", ls=":", label="nominal radius")
        ax.set(xlabel="kill radius [m]", ylabel="kill rate", title=plat)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figs / f"killradius_{plat}.png", dpi=130)
        plt.close(fig)
    print("wrote sweep tables and figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
