# BUGS — known non-blocking issues

Unknown date — all findings below are non-blocking: they do not prevent
correct training runs, affect only resume/accounting edge cases, or are
documented design compromises. They are tracked as a follow-up list, not
fixed mid-task (per task instructions).

**Last reviewed 2026-09-30 (smoke run on CUDA host completed):** Phase 9
in-container smoke ran successfully. Two runtime bugs found and fixed (items
11–12). Items 1–10 unchanged from prior review.

---

## 1. Mid-rollback resume retrains the retry stage from transfer weights
**Area:** `src/training/orchestrator.py`
**Impact:** Low. If a run is interrupted while a rollback retry is in flight,
resuming rebuilds the retry stage from either a fresh model (stage 1) or the
previous stage's final weights instead of the retried stage's own final
checkpoint. The scheduler's episode records are intentionally NOT restored
on resume (advisory only), so the retry restarts its success window anyway.
**Acceptable:** Yes — documented design change (rollback semantics §6.2 of
the plan preserved; only the resume path differs).

## 2. `steps_added` undercounts rolled-back stages in resume accounting
**Area:** `src/training/orchestrator.py` (`_run_config` / `_write_state`)
**Impact:** Low. `run_state.json` `total_steps_all_stages` and `run_summary`
step counts count each retry attempt separately but never add on the full
budget wasted by the rolled-back attempt (re-entry resets
`model.num_timesteps = 0`, so only the retry's steps are counted). Numbers
are informational only — the budget caps remain correct.

## 3. `run_state.json` `episode_buffer` serializes numpy scalars as strings
**Area:** `CheckpointManager._atomic_write_json` (`default=str`) +
`_write_state` episode_buffer
**Impact:** None today — the scheduler's records are stored for audit only
and deliberately not restored on resume. A future "resume exact episode
window" feature must round-trip numpy types properly (e.g. custom encoder).

## 4. `requirements.txt` relies on the base image's torch satisfying `>=2.7.0`
**Area:** `requirements.txt`
**Impact was HIGH (now FIXED):** pip resolves `torch>=2.7.0` to the latest
release (2.14.0 as of 2026-09-30), downloading a 554 MB CPU-only wheel and
silently overwriting the CUDA build from the base image. `torch` has been
removed from `requirements.txt` — the base image
`pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime` provides torch 2.7.0+cu128;
pip must not touch it.

## 5. `scripts/smoke_test.py` needs `hover_env.py` at the repo root
**Area:** scripts/smoke_test.py
**Impact:** Low. The parity test imports `from hover_env import
AltitudeHoldEnv`, so the suite runs on the host (repo root on `sys.path`),
not inside the container (the Dockerfile copies `src/`, `scripts/`,
`configs/` only). The in-container verification path is `train.py --smoke`,
which does not need `hover_env.py`.

## 6. Off-policy TB histograms may be sparser than on-policy — FIXED
**Area:** `src/training/callbacks.py`
**Impact:** Resolved. The reward-component/metric callbacks no longer
introspect `model.log_interval` (SB3 does not reliably propagate it for
off-policy algos); the orchestrator now passes `log_interval` explicitly (10
for PPO, else `max(10000, checkpoint_interval_steps)`). SAC/TD3 still never
call `_on_rollout_end`, so their TensorBoard step means are batched
differently from PPO's per-rollout dumps — cosmetic only, nothing is lost.

## 7. Target integration uses first-order Euler, drone physics is RK4
**Area:** `src/envs/target_generator.py` vs `src/physics/pipeline.py`
**Impact:** None (design compromise). The drone's equations of motion are
integrated with a classical 4th-order Runge–Kutta (`rk4()`, port of
`hover_env._ph_rk4`), while the target kinematics advance with plain
first-order Euler (`vel += acc·dt; pos += vel·dt; acc += jerk·dt`) and no
sub-stepping. `predicted_pos()` (§6.2 lookahead) uses the exact closed form,
so the two disagree slightly over a 0.3–1.0 s manoeuvre segment — negligible
at 100 Hz with ≤ 30 m/s targets. Keeping the target on Euler is adequate
for its purpose (a moving goal, not a physical plant simulation).

## 8. `Monitor` overwrites its CSV on resume (`override_existing=True`)
**Area:** `src/envs/base_env.py` (`make_env_factory`) / orchestrator
**Impact:** Low. On resume, `Monitor(filename=...)` truncates the previous
`monitor.csv` for the same seed. Checkpoints and run state are unaffected;
only the raw per-episode CSV history of the interrupted run is lost.

## 9. `J_inv` recomputed per acceleration component per step
**Area:** `src/physics/pipeline.py` (dynamics)
**Impact:** Perf nit only. The Jacobian inverse is recomputed for each
derivative stage (≈3 small 3×3 solves per step). Fine at 100 Hz with 8
parallel envs; cache per-step if profiling ever demands it.

## 10. Container runs as root with a writable configs mount
**Area:** `Dockerfile` / `docker-compose.yml`
**Impact:** Low for a single-user training host. There is no non-root
`USER` directive and `./configs:/data/configs` mounts read-write, so an
accident inside the container could churn the shipped configs. Harden with
`USER` + `:ro` before any multi-user deployment.

## 11. `_atomic_json` TypeError: `os.open` returns int, not tuple — FIXED
**Area:** `src/training/orchestrator.py` (`_atomic_json`)
**Impact:** Would crash every JSON write at runtime. `os.open()` returns a
single file descriptor (`int`), but the original code unpacked it as
`fd, tmp = os.open(...)`. In the smoke run this was masked by a host-side
file-mount patch; now fixed in-tree. The `tmp` path is constructed first,
then `fd = os.open(str(tmp), ..., 0o644)`, and `os.replace` reuses `tmp`.

## 12. `opencv-python` crashes `SubprocVecEnv` workers (missing `libxcb`) — FIXED
**Area:** `Dockerfile` / `requirements.txt` (transitive via `sb3[extra]`)
**Impact:** High (performance). `stable-baselines3[extra]` installs the full
`opencv-python`, which needs X11/GUI libraries (`libxcb.so.1`, etc.) absent
from the headless PyTorch container. Each `SubprocVecEnv` worker crashes on
`import cv2`, the orchestrator silently falls back to `DummyVecEnv`
(sequential, single-process), and the 8-env parallelism is lost. Fixed by
adding a Dockerfile step that replaces `opencv-python` with
`opencv-python-headless` after `pip install`.

## 13. ESC polynomial is monotonically decreasing (R-1) — KNOWN LIMITATION
**Area:** `src/physics/constants.py` (L121–L122), `src/physics/pipeline.py` (L99–L110)
**Impact:** Sim-to-real only — does not affect training inside the simulator.
The fitted ESC steady-state polynomial `Ω_ss(c) = 2461.18 − 170.06√c − 1818.9c`
is monotonically **decreasing** over the entire command range:
`Ω_ss(0.02) ≈ 2400.75 rad/s`, `Ω_ss(0.2325) ≈ 1956 rad/s` (hover),
`Ω_ss(1.0) ≈ 472 rad/s`. The derivative at hover command is ≈ −1995 rad/s
per unit command. This means increasing throttle *reduces* motor speed and
therefore thrust — the opposite of a real motor. A policy can still learn to
hover and intercept inside the simulator (it finds the equilibrium at cmd ≈ 0.23),
but the learned command mapping will not transfer to a real Crazyflie.
**Decision (2026-10-03):** Do not correct `PH_BAT` or the polynomial coefficients
at this time. Any change would alter `PH_C_HOVER = 0.23253743635354834` and
break the bit-exact Stage-1 parity with `hover_env.py` (a hard requirement).
**Recommended fix for future work:** re-identify the ESC polynomial on the real
drone; replace coefficients; recompute `PH_C_HOVER`; reset Stage-1 parity baseline.
**Acceptable for current scope:** Yes — training goal is algorithmic validation,
not immediate sim-to-real transfer.