"""Run every training job of notebooks 02-05 locally at reduced budgets.

The run specifications are copied from the notebooks (02 algorithm selection,
03 flight controller, 04 guidance, 05 ablations); only ``total_steps`` is
smaller, because the full budgets of ``configs/training.yaml`` need about 30
laptop CPU-hours.  The budgets used are written to
``results/tables/reduced_budget_run.json`` so every table built from these
checkpoints can say it is a reduced-budget result.  The Colab notebooks remain
the full-budget procedure.

    python scripts/run_reduced_budget.py --phase fc p6 --workers 2
    python scripts/run_reduced_budget.py --phase guidance --only fixed_wing
    python scripts/run_reduced_budget.py --phase ablations --shard 1/3

``--only`` keeps jobs whose run id contains the text and ``--shard k/n`` keeps
every n-th job, so one queue can be split across several Colab sessions
(``notebooks/00_colab_parallel.ipynb``).  Each worker process runs torch with one
thread, so ``--workers`` (default: CPU count) processes do not oversubscribe.

Jobs already finished (``DONE`` file with enough steps) are skipped by
``training.runner.train``, so the script can be restarted.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]

# Reduced budgets (full budgets in configs/training.yaml: 200k / 1M / 5M).
BUDGETS = {
    "algorithm_selection_steps": 30_000,
    "fc_train_steps": 200_000,
    "guidance_train_steps": 300_000,
    "ablation_guidance_steps": 100_000,
    "ablation_fc_steps": 100_000,
}
SEEDS = (0, 1, 2)


def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def jobs_fc():
    return [(f"p7_fc_asym_s{s}", {
        "task": "fc", "algo": "ppo", "seed": s, "total_steps": BUDGETS["fc_train_steps"],
        "checkpoint_every": 20000, "asymmetric": True, "actor_obs_dim": 33,
        "fc": {"domain_randomization": True, "wind": True}}) for s in SEEDS]


def jobs_p6():
    # Notebook 02 passed N_nav = 5.0 (the quad A2 winner at the time); since BUGS #35 the
    # env resolves the current sweep winner itself, so it is left out here.
    tasks = {
        "fc": {"task": "fc", "fc": {"domain_randomization": True, "wind": True}},
        "guidance": {"task": "guidance", "guidance": {
            "platform": "quadcopter", "mode": "apn_rl", "target_set": "evasive"}},
    }
    out = []
    for algo in ("ppo", "sac", "td3"):
        for task, extra in tasks.items():
            for s in SEEDS:
                out.append((f"p6_{algo}_{task}_s{s}", {
                    "algo": algo, "seed": s, "total_steps": BUDGETS["algorithm_selection_steps"],
                    "checkpoint_every": 10000, **copy.deepcopy(extra)}))
    return out


def jobs_guidance():
    out = []
    for platform in ("quadcopter", "fixed_wing", "missile"):
        for arm, mode in (("a3", "apn_rl"), ("a4", "pure_rl")):
            for s in SEEDS:
                out.append((f"p7_{platform}_{arm}_s{s}", {
                    "task": "guidance", "algo": "ppo", "seed": s,
                    "total_steps": BUDGETS["guidance_train_steps"], "checkpoint_every": 25000,
                    "asymmetric": True, "actor_obs_dim": 85,
                    "guidance": {"platform": platform, "mode": mode,
                                 "target_set": "evasive", "noise_scale": 1.0}}))
    return out


def jobs_ablations():
    import yaml
    abl = yaml.safe_load((ROOT / "configs" / "ablations.yaml").read_text(encoding="utf-8"))

    def base(s):
        return {"task": "guidance", "algo": "ppo", "seed": s,
                "total_steps": BUDGETS["ablation_guidance_steps"], "checkpoint_every": 25000,
                "asymmetric": True, "actor_obs_dim": 85,
                "guidance": {"platform": "quadcopter", "mode": "apn_rl", "target_set": "evasive"}}

    out = []
    for group, prefix in (("guidance_ablations", "p9a"), ("sensitivity", "p9s"),
                          ("reward_weights", "p9r"), ("network", "p9n")):
        for name, delta in abl[group].items():
            for s in SEEDS:
                out.append((f"{prefix}_{name}_s{s}", _merge(base(s), delta)))
    for name, delta in abl["fc_randomisation"].items():
        for s in SEEDS:
            out.append((f"p9f_{name}_s{s}", {
                "task": "fc", "algo": "ppo", "seed": s,
                "total_steps": BUDGETS["ablation_fc_steps"], "checkpoint_every": 25000,
                "asymmetric": True, **copy.deepcopy(delta)}))
    return out


PHASES = {"fc": jobs_fc, "p6": jobs_p6, "guidance": jobs_guidance, "ablations": jobs_ablations}


def _run(run_id, spec):
    import os
    import warnings
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    import torch
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")
    from training.runner import train
    t0 = time.time()
    try:
        out = train(run_id, spec)
        return run_id, "ok", int(out.get("num_timesteps", 0)), time.time() - t0, ""
    except Exception:  # keep the queue going; the failure is logged
        return run_id, "FAILED", 0, time.time() - t0, traceback.format_exc()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", nargs="+", choices=list(PHASES), required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--only", default=None, help="keep run ids containing this text")
    ap.add_argument("--shard", default=None, help="k/n: keep jobs k-1, k-1+n, ... (1-based k)")
    args = ap.parse_args(argv)

    rec_path = ROOT / "results" / "tables" / "reduced_budget_run.json"
    rec = json.loads(rec_path.read_text()) if rec_path.exists() else {}
    rec.update({"budgets": BUDGETS, "seeds": list(SEEDS),
                "note": "Local laptop run at reduced budgets (results due 2026-11-15); "
                        "full budgets are in configs/training.yaml and the Colab notebooks."})
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    rec_path.write_text(json.dumps(rec, indent=2))

    jobs = [j for ph in args.phase for j in PHASES[ph]()]
    if args.only:
        jobs = [j for j in jobs if args.only in j[0]]
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        jobs = jobs[k - 1::n]
    log = ROOT / "results" / "tables" / "reduced_budget_log.csv"
    if not log.exists():
        log.write_text("run_id,status,num_timesteps,seconds\n")
    print(f"{len(jobs)} jobs, {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_run, rid, spec) for rid, spec in jobs]
        for f in as_completed(futs):
            rid, status, n, sec, tb = f.result()
            with log.open("a") as fh:
                fh.write(f"{rid},{status},{n},{sec:.0f}\n")
            print(f"{time.strftime('%H:%M:%S')} {rid} {status} {n} steps {sec/60:.1f} min", flush=True)
            if tb:
                print(tb, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
