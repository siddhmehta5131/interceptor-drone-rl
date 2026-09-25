# BUGS — known non-blocking issues

Unknown date — all findings below are non-blocking: they do not prevent
correct training runs, affect only resume/accounting edge cases, or are
documented design compromises. They are tracked as a follow-up list, not
fixed mid-task (per task instructions).

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
**Area:** `requirements.txt` / `Dockerfile`
**Impact:** Low. `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime` ships
torch 2.7.0+cuda12.8 which satisfies `torch>=2.7.0`, so pip skips the
reinstall (no CPU-wheel overwrite of the CUDA build). If the image tag ever
changes to a build that still satisfies the constraint but is CPU-only this
silently reverts to CPU training. Mitigation if it ever bites: pin the exact
index URL (`pip install torch --index-url .../whl/cu128`).

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