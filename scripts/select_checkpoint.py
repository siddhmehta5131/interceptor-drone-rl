"""Best-checkpoint selection on validation data (early stopping).

PPO returns are not monotone in training steps: the flight-controller run
``p7_fc_asym_s0`` peaked near 360k steps (mean episode reward -344) and fell to
-1958 at 1M (``runs/p7_fc_asym_s0/curves.csv``).  The runner therefore keeps a
snapshot at every checkpoint (``runs/<id>/snapshots/step_*.zip``) and this script
picks one of them on VALIDATION data that is disjoint from training and from the
evaluation seeds of the result tables.

Flight controller (``--task fc``): 10 setpoint sequences of each kind, nominal
physics, seeds 9000 + 100 * kind + i (evaluation uses 7000 + ...).  Score = mean
over the six kinds of the median rate-tracking RMS error (rad/s); lowest wins.

Guidance (``--task guidance``): evasive set, 4 episodes per manoeuvre, seeds
20000 + 100 * manoeuvre + k (evaluation uses 10000 + k).  Highest kill rate
wins; ties are broken by the lower median miss distance.

The rule is fixed here before any snapshot is scored.  The chosen snapshot is
copied to ``runs/<id>/best.zip`` and the scores of all snapshots are written to
``results/tables/checkpoint_selection_<id>.csv``.

    python scripts/select_checkpoint.py --task fc --run p8_fc_asym_s0
    python scripts/select_checkpoint.py --task guidance --run p9e_quadcopter_sigma0.1_s0 --platform quadcopter
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]

FC_VAL_SEED0 = 9000
FC_VAL_SEQUENCES = 10
G_VAL_SEED0 = 20000
G_VAL_EPISODES = 4


def _quiet():
    os.environ["OMP_NUM_THREADS"] = "1"
    import warnings
    import torch
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")


def _score_fc(job):
    path, _ = job
    _quiet()
    from stable_baselines3 import PPO
    from interceptor.bridge.closed_loop import PolicyRateController
    from interceptor.fc import metrics, rollout, setpoints
    from interceptor.sim import pipeline
    model = PPO.load(str(path), device="cpu")
    dt, n_steps = 0.002, 750
    per_kind = []
    for ki, kind in enumerate(setpoints.KINDS):
        vals = []
        for i in range(FC_VAL_SEQUENCES):
            rng = np.random.default_rng(FC_VAL_SEED0 + 100 * ki + i)
            sp = setpoints.sample(kind, n_steps, dt, rng, 12.0)
            log = rollout.rollout(PolicyRateController(model, rate_max=12.0), sp,
                                  params=pipeline.PhysicsParams.rotor_drag(), dt=dt)
            m = metrics.tracking_metrics(log["t"], log["setpoint"][:, :3], log["omega"], log["cmd"])
            vals.append(m["rms"])
        per_kind.append(float(np.median(vals)))
    return {"snapshot": Path(path).name, "score_rms": float(np.mean(per_kind)),
            **{f"rms_{k}": v for k, v in zip(setpoints.KINDS, per_kind)}}


def _score_guidance(job):
    path, platform = job
    _quiet()
    from stable_baselines3 import PPO
    from interceptor.guidance.env import UnifiedInterceptEnv
    from interceptor.guidance.maneuvers import ManeuverType, PlatformType
    model = PPO.load(str(path), device="cpu")
    kills, misses = [], []
    for i, man in enumerate(ManeuverType):
        for k in range(G_VAL_EPISODES):
            env = UnifiedInterceptEnv(platform=PlatformType(platform), mode="apn_rl",
                                      target_set="evasive", fixed_maneuver=man,
                                      seed=G_VAL_SEED0 + 100 * i + k)
            obs, _ = env.reset()
            done = False
            while not done:
                obs, _, term, trunc, _ = env.step(model.predict(obs, deterministic=True)[0])
                done = term or trunc
            kills.append(bool(env.last_info["killed"]))
            misses.append(float(env.min_miss_distance))
    return {"snapshot": Path(path).name, "episodes": len(kills), "kills": int(np.sum(kills)),
            "kill_rate": float(np.mean(kills)), "median_miss": float(np.median(misses))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=["fc", "guidance"], required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--platform", default="quadcopter")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--min-step", type=int, default=0)
    args = ap.parse_args(argv)

    run = ROOT / "runs" / args.run
    snaps = sorted((run / "snapshots").glob("step_*.zip"))
    snaps = [s for s in snaps if int(s.stem.split("_")[1]) >= args.min_step]
    if not snaps:
        raise SystemExit(f"no snapshots in {run / 'snapshots'}")
    fn = _score_fc if args.task == "fc" else _score_guidance
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(fn, [(s, args.platform) for s in snaps]))
    if args.task == "fc":
        best = min(rows, key=lambda r: r["score_rms"])
    else:
        best = min(rows, key=lambda r: (-r["kill_rate"], r["median_miss"]))
    for r in rows:
        r["selected"] = int(r is best)
    out = ROOT / "results" / "tables" / f"checkpoint_selection_{args.run}.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    shutil.copyfile(run / "snapshots" / best["snapshot"], run / "best.zip")
    for r in rows:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    print("selected", best["snapshot"], "->", run / "best.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
