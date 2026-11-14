"""Which repo lines, functions and configs does the live v2 pipeline touch? (value audit support)

    python scripts/trace_liveness.py [--out results/tables/liveness.json]

Copies the repository to a temporary folder (so the evaluation scripts cannot
overwrite real result tables), then, inside that copy, runs tiny versions of
every v2 entry point under ``sys.monitoring`` (Python >= 3.12): guidance training
for 3 platforms x {apn_rl, pure_rl} (asymmetric PPO) and SAC/TD3, flight-controller
training, evaluate_guidance, evaluate_closed_loop, evaluate_fc, sweep_nav and
sweep_robustness with 1 episode. Process pools are replaced by thread pools so the
tracer sees worker code. Output: executed (file, line) pairs, started functions
(file, first line) and opened configs/*.yaml files.

One-episode runs give a LOWER bound on liveness (rare manoeuvre branches may not
fire), so only files with no executed line at all count as legacy. Those files were
moved to archive/legacy_3stage/ on 2026-11-14 and are no longer audited. ~10 min on a
laptop CPU.
"""
import argparse, shutil, subprocess, tempfile
import builtins, concurrent.futures, json, os, sys, tempfile, traceback
from pathlib import Path

if len(sys.argv) < 2 or sys.argv[1] != "--inner":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/tables/liveness.json")
    a = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    tmp = Path(tempfile.mkdtemp(prefix="liveness_")) / "repo"
    shutil.copytree(repo, tmp, ignore=shutil.ignore_patterns(
        ".git", "archive", "videos", "runs", "__pycache__", "*.zip", "*.obj"))
    out = (repo / a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    raise SystemExit(subprocess.call([sys.executable, str(tmp / "scripts" / "trace_liveness.py"),
                                      "--inner", str(tmp), str(out)]))
ROOT = Path(sys.argv[2]).resolve()
OUT = Path(sys.argv[3])
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
root_s = str(ROOT).lower()

lines, starts, opened = set(), set(), set()
mon = sys.monitoring
TOOL = mon.PROFILER_ID
mon.use_tool_id(TOOL, "liveness")


def _rel(fn):
    f = os.path.abspath(fn).lower()
    if f.startswith(root_s) and "site-packages" not in f:
        return os.path.relpath(os.path.abspath(fn), ROOT).replace("\\", "/")
    return None


_cache = {}


def rel(code):
    fn = code.co_filename
    if fn not in _cache:
        _cache[fn] = _rel(fn)
    return _cache[fn]


def on_line(code, line):
    r = rel(code)
    if r is None:
        return mon.DISABLE
    lines.add((r, line))
    return mon.DISABLE  # each (code, line) location only needs recording once


def on_start(code, off):
    r = rel(code)
    if r is not None:
        starts.add((r, code.co_firstlineno))
    return mon.DISABLE


mon.register_callback(TOOL, mon.events.LINE, on_line)
mon.register_callback(TOOL, mon.events.PY_START, on_start)
mon.set_events(TOOL, mon.events.LINE | mon.events.PY_START)


def audit(ev, args):
    if ev == "open" and args and isinstance(args[0], (str, os.PathLike)):
        r = _rel(os.fspath(args[0]))
        if r and r.startswith("configs/"):
            opened.add(r)


sys.addaudithook(audit)
# keep everything in-process so the tracer sees it
concurrent.futures.ProcessPoolExecutor = concurrent.futures.ThreadPoolExecutor
import concurrent.futures.process as _p  # noqa
_p.ProcessPoolExecutor = concurrent.futures.ThreadPoolExecutor

tmp = Path(tempfile.mkdtemp(prefix="live_"))
steps_done = []


def step(name, fn):
    try:
        fn()
        steps_done.append((name, "ok"))
    except SystemExit:
        steps_done.append((name, "ok(exit)"))
    except Exception as e:  # keep going; record failure
        steps_done.append((name, f"FAIL {type(e).__name__}: {e}"))
        traceback.print_exc()
    print("step", name, steps_done[-1][1], flush=True)


from training.runner import train  # noqa: E402

for plat in ("quadcopter", "fixed_wing", "missile"):
    for mode in ("apn_rl", "pure_rl"):
        step(f"train {plat} {mode}", lambda plat=plat, mode=mode: train(
            f"live_{plat}_{mode}", {"task": "guidance", "algo": "ppo", "seed": 0, "total_steps": 256,
                                     "checkpoint_every": 256, "asymmetric": True, "actor_obs_dim": 85,
                                     "algo_overrides": {"n_steps": 128, "batch_size": 64},
                                     "guidance": {"platform": plat, "mode": mode, "target_set": "evasive",
                                                  "noise_scale": 1.0}}, run_root=tmp))
for algo in ("sac", "td3"):
    step(f"train quad {algo}", lambda algo=algo: train(
        f"live_q_{algo}", {"task": "guidance", "algo": algo, "seed": 0, "total_steps": 256,
                           "checkpoint_every": 256, "algo_overrides": {"learning_starts": 64},
                           "guidance": {"platform": "quadcopter", "mode": "apn_rl"}}, run_root=tmp))
step("train fc", lambda: train("live_fc", {"task": "fc", "algo": "ppo", "seed": 0, "total_steps": 256,
                                           "checkpoint_every": 256,
                                           "algo_overrides": {"n_steps": 128, "batch_size": 64}}, run_root=tmp))


def script(mod, argv):
    import importlib
    m = importlib.import_module(mod)
    return m.main(argv)


step("evaluate_guidance", lambda: script("evaluate_guidance", ["--episodes", "1", "--workers", "2"]))
step("evaluate_closed_loop", lambda: script("evaluate_closed_loop", ["--episodes", "1", "--workers", "2"]))
step("evaluate_fc", lambda: script("evaluate_fc", []))
step("sweep_nav", lambda: script("sweep_nav", ["--episodes", "1", "--workers", "2"]))
step("sweep_robustness", lambda: script("sweep_robustness", ["--episodes", "1", "--workers", "2"]))

mon.set_events(TOOL, 0)
OUT.write_text(json.dumps({"lines": sorted(lines), "starts": sorted(starts), "opened": sorted(opened),
                           "steps": steps_done}), encoding="utf-8")
print("wrote", OUT, len(lines), "lines", len(starts), "functions", sorted(opened))
