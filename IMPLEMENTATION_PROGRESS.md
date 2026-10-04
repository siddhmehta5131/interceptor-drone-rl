# Implementation Progress — `implementation_plan_d2_d58.md`

Append-only log. Each entry records one completed task, what changed, and how it
was verified. Plan phases: **A** config split · **B** public API · **C** engine ·
**D** environment · **E** observations/policy · **F** predictors · **G** rewards ·
**H** smoke/fixes/docs · **I** repo sync, Docker, push.

Overall completion: **100 %** (Phases A–I complete).

---

## 2026-10-03 — Phase A: split configs into `model.yaml` + `config.yaml` ✅

**Why:** D-2/D-7 — model architecture and training settings live in separate files;
stage list moves from YAML into the `train_model(stages=...)` argument.

**Changed**
- NEW `configs/model.yaml` (`ppo_baseline`), `configs/config.yaml`,
  `configs/smoke_model.yaml` (`ppo_baseline` + `ppo_5layer_deep`),
  `configs/smoke_config.yaml`.
- DELETED `configs/default_run.yaml`.
- NEW `src/utils/model_loader.py`: `ModelDef` (algo, net_arch, activation,
  `obs_history {frames, skip}`, hyperparameters, `privileged_critic`),
  `load_models`, `models_signature`, `model_def_from_dict`, `PRIVILEGED_FIELDS`
  = `{time_remaining:1, facing_error:1, target_true_pos:3, target_true_vel:3}`.
  Plan-schema canonicalisation (`algorithm`, `policy_kwargs.net_arch` list → `{pi,vf}`,
  `vf`/`qf` head key per algo, `activation_fn` string) and D-34 identifier
  validation (valid Python identifier, no keyword, no dunder).
- REWRITTEN `src/utils/config_loader.py`: `ObservationConfig`, `TrainConfig`,
  `parse_range`, `ON_CAPPED_CHOICES = ("continue","stop")`, stage-override merging
  (`reward`/`spawn`/`intercept`/`obs` subblocks + `_STAGE_SPAWN_ALIASES` for
  `path_end_distance`/`path_speed_cap`/`path_shape`), combined `config_hash` over
  `training_signature_dict()` (D-20), `TrainConfig` passed through `train_model`.
- REWRITTEN `src/utils/__init__.py` exports; `RunConfig`/`load_run_config` removed.
- `Dockerfile`: `CMD ["--help"]`, added `COPY run.py ./run.py`.

**Verified:** `load_models("configs/smoke_model.yaml")` → both model ids;
`ppo_5layer_deep` history `{frames:5, skip:3}`; `ppo_baseline` privileged dim 8;
`load_config("configs/smoke_config.yaml")` + `bind_models` → 64-char combined hash,
`seed 42`, `n_parallel_envs 2`, `on_capped continue`, `future_source pred`;
`parse_range` scalar/range; per-stage `episode_length_s` 10/15/20/20/25/30/30/30 s.

---

## 2026-10-03 — Phase D: environment (target altitude, polynomial path, episode length, facing) ✅

**Why:** D-1 config-driven altitude, D-2 fixed polynomial target path for stages
5–7, D-13/D-14 per-episode duration, D-4 facing-error signal.

**Changed**
- `src/envs/stage_config.py`: `episode_length_s` (float or `{min,max}`) replaces the
  stored step count, with a derived `max_episode_steps` property; removed lookahead
  fields/`LOOKAHEAD_HORIZON_S` (E-1); added `RewardConfig.k_facing` + `FACING_CUTOFF_M`;
  added `SpawnConfig.path_end_distance/path_speed_cap/path_shape`; stage durations
  10/15/20/20/25/30/30/30 s.
- `src/envs/target_generator.py` rewritten: `PolynomialTargetPath` (monotone random
  shape coefficients + endpoint scaling so the speed cap always holds; `position_at`,
  `velocity_at`, `acceleration_at`, `jerk_at`, `peak_speed`), `polynomial` property,
  `elapsed` property, `reset(drone_pos, episode_length_s=...)`, closed-form `step`
  with no re-sampling/containment for stages 5–7; evasive-only `_resample_maneuver`.
- `src/envs/base_env.py`: `target_alt` kwarg → `_coerce_alt_range`, drawn per reset
  **after** the six parity draws; `_draw_episode_length()` draws only when `hi > lo`;
  `_facing_error()`; `facing_error=`/`desired_yaw` surfaced in `info`;
  `make_env_factory` threads the new args and wraps `Monitor(override_existing=False)`
  (BUG 8).

**Verified:** all 8 stages run with the new signature; episode lengths match the
table; stages 5/6/7 report `polynomial=True`, stage 8 `False`; **Stage-1 parity vs
`hover_env` intact** (`np.array_equal` observations, `==` rewards, identical
termination across seeds 12345/777/0).

---

## 2026-10-03 — Phase F: `src/prediction` package ✅

**Why:** D-5/D-6 — pluggable future-position predictors.

**Changed**
- NEW `src/prediction/base.py` (`Predictor.predict(...) -> (pred_pos, pred_vel)`
  with row 0 = "now"), `const_vel.py` (`ConstVelPredictor`),
  `linear_ridge.py` (`LinearRidgePredictor`, ridge `1e-3`, window 16, degree 2,
  unshrunk intercept, row 0 overwritten with the measurement),
  `__init__.py` with `PREDICTOR_REGISTRY`/`get_predictor`.

**Verified:** imported and instantiated for both predictors; shape contract
`(n+1, 3)` for position and velocity.

---

## 2026-10-03 — Phase E: observations, future samples, privileged critic fields ✅

**Why:** D-15/D-16/D-54 — history + future stacking and an asymmetric
actor/critic observation split.

**Changed**
- `src/envs/obs_builder.py` rewritten: `FUTURE_SAMPLE_DIM=7`,
  `PRIVILEGED_FIELDS`, `privileged_dim`, `observation_dim`
  (`19*(m+1) + 7*n + priv` for target stages, `14` for stage 1),
  `make_future_block(n,7)`, `make_privileged_block(fields, ...)`,
  `ObsBuilder(..., future_samples, future_skip, privileged_fields)` with
  `obs_dim`/`privileged_dim`/`actor_obs_dim`; exact stage-1 noise draw order kept.
- `src/envs/base_env.py`: `future_source` kwarg (`true` → exact simulator path,
  `pred` → predictor, `true` degrades to `pred` for targets without a closed-form
  path), `_future_samples_world()`, future/privileged blocks expressed in the drone's
  current body frame (D-54); privileged target fields = ideal − current.
- NEW `src/training/asymmetric_policy.py`: `_SplitExtractor`,
  `AsymmetricActorCriticPolicy` (two `MlpExtractor`s, `extract_features` slices the
  actor width), `_get_constructor_parameters()` re-declares `actor_obs_dim` so the
  split survives `save()`/`load()`, `policy_kwargs_for(...)` helper.

**Verified:** stage 2–8 obs dims match `observation_dim` and
`observation_space.shape` (105 for `ppo_baseline` at m=3/n=3, 135 for
`ppo_5layer_deep` at m=5/n=3); `future_source true` vs `pred` identical for stages
2/3/4/8, identical for stage 5 (linear path), different for stages 6/7;
**Stage-1 parity re-verified** after the changes.

---

## 2026-10-03 — Phase G: facing reward term ✅

**Why:** D-4 — reward for keeping the nose on the target.

**Changed**
- `src/envs/reward.py`: `k_facing` term `-k_facing * |facing_error|` inside the
  `distance <= FACING_CUTOFF_M (0.5 m)` gate, `facing_error` kwarg, `metrics["facing_error"]`,
  `TERMINAL_INFO_KEYS += "r_facing"`; explicit left-fold summation untouched (parity).
- `src/training/callbacks.py`: `REWARD_COMPONENT_KEYS += "r_facing"`,
  `METRIC_KEYS += "facing_error"`.

**Verified:** stage table shows `k_facing = 0.15` for stages 2–8, `0.0` for stage 1.

---

## 2026-10-03 — Phase B: `TrainResult` API, results types, weight transfer ✅

**Why:** D-26/D-28/D-29/D-32/D-33 — a `train_model()` entry point, persisted result
objects, and cross-stage weight transfer.

**Changed**
- NEW `src/results.py`: status constants (`completed/capped/stuck/skipped/crashed/denied`),
  `StageOutcome`, `ModelResult` (`save`/`load`/`load_model`, training-meta sidecar),
  `TrainResult` (attribute + item access by model id, `save()` → `run_summary.json`
  plus per-model artifacts), `check_continuation_eligibility` (denied for capped
  sources, skipped when the last completed stage does not precede the request),
  `save_result_json`/`load_result_json`.
- NEW `src/api.py`: `train_model(source, stages, name, *, config, model_ids, seed,
  vecenv, resume) -> TrainResult`; accepts a path or a prior `TrainResult`, reuses the
  source config, verifies continuation hashes (D-28), `run_dir = data_dir/runs/<name>`.
- NEW `run.py`: executable documentation — two-phase example (stages 1–4 then 5–7)
  plus a CLI with `--source/--continue-from`, `--stages`, `--name`, `--models`.
- NEW `src/training/weight_transfer.py`: `SHARED_STAGE1_PREFIX = 12`,
  `stage1_prefix_for`, `plan_transfer`, `transfer_state_dicts`, `transfer_weights`,
  `transfer_model`.
- `src/training/curriculum.py`: `valid_predecessors(first_stage)`.
- `src/training/checkpoint_manager.py`: `training_meta_path`, `save_training_meta`,
  `load_training_meta`.

**Verified:** `results.py` round-trips; eligibility returns the exact plan messages;
`obs_dim_for` → 14 / 105 / 135; weight transfer keeps the first 12 observation columns
bit-identical and random-inits the rest at scale `1e-3`; `py_compile` of all new files.

---

## 2026-10-03 — Phase C: orchestrator rewrite (internal engine) ✅

**Why:** B-5 + C-1..C-9 — stages/name/seed as arguments, `TrainResult` return,
per-model isolation, `on_capped`, mid-stage resume, replay buffers.

**Changed**
- `src/training/orchestrator.py` fully rewritten: `Orchestrator(cfg, *, name, run_dir,
  stages, source_result, result, vecenv, resume)`; `RetryLedger`
  (`<data_dir>/curriculum_state/retries.json`, keyed `model_id:stage_N`, D-45);
  per-model `try/except` → `status="crashed"` and logging with `exc_info` (D-31);
  `on_capped == "stop"` halts that model only (D-24/D-55); skip/denied verdicts recorded;
  model selection = mid-stage resume → rollback re-entry → cross-stage transfer → fresh;
  `_training_meta.json` written next to each final checkpoint (D-28); SAC/TD3 replay
  buffers persisted (D-47); `StageOutcome` per stage (D-32); tensorboard/logs under the
  run dir (D-22); `vecenv`/`resume` plumbed.
- `scripts/train.py` fully rewritten as a thin CLI over `train_model()` while staying
  the container ENTRYPOINT: `--source/--continue-from`, `--stages`, `--models`,
  `--data-dir`, `--seed`, `--vecenv`, `--no-resume`, `--save`, and the smoke switches
  (`--smoke`, `--smoke-stages`, `--smoke-steps`, `--smoke-models`) which shrink budgets
  through `TrainConfig.stage_overrides` instead of a temp YAML.
- `scripts/evaluate.py` rewritten: `--run` accepts a run folder, `run_summary.json`,
  `<model>.result.json` or `<model>.zip`; `--model` wins; model definition comes from
  `configs/model.yaml`; the environment is built with that model's history/privileged
  layout and the config's future/predictor settings; metrics file written to
  `<run_dir>/results/eval_<model_id>_stage_<N>.json`.

**Verified:** `py_compile` for all three; `train_model` signature
`(source, stages, name, *, config='configs/config.yaml', model_ids=None, seed=None,
vecenv='auto', resume=True)`; `api._check_eligibility` delegation returns the plan
messages; `evaluate.py` resolves run dirs, `run_summary.json` and `.result.json`
sidecars; `evaluate.py` metrics loop is bounded by `stage.max_episode_steps`.
**Not yet executed:** the SB3-dependent training loop (host has no torch) — validated in
Phase I inside Docker.

---

## 2026-10-03 — Phase H (partial): R-1 ESC investigation ✅

**Why:** H-7 — "motor speed falls as thrust command rises"; non-blocking for training,
critical for sim-to-real.

**Finding:** confirmed. `Ω_ss(c) = 2461.18 − 170.06√c − 1818.9c` is monotonically
**decreasing** over the whole command range (2400.75 rad/s at `c = 0.02`,
472.22 rad/s at `c = 1.0`); the derivative at the hover command is `−1995.23`.

**Decision:** **do not change `PH_BAT`.** Any correction alters
`PH_C_HOVER = 0.23253743635354834` and would break the bit-exact Stage-1 parity with
`hover_env.py`, which is a hard requirement. Documented as a known limitation; a
diagnostic will be added instead.

**Still open in Phase H:** rewrite `scripts/smoke_test.py`, add the R-1 entry to
`BUGS.md`, refresh `README.md` (mass 0.04085 kg, new API/commands).

---

## 2026-10-04 — Phase H: `scripts/smoke_test.py` rewritten, 12/12 green ✅

**Why:** D-58 — the suite had to cover the new observation layout, polynomial
targets, predictors, results/continuation and the two-file config split.

**Changed**
- REWRITTEN `scripts/smoke_test.py`: 12 dependency-light groups (no torch/SB3),
  each wrapped by a `test` decorator, with a summary line and exit code:
  1. `parity_stage1_vs_hover` — 111 steps bit-identical obs/reward/term/info
  2. `all_stage_envs_run` — 8 stages, dims (105 target / 14 hover), `episode_stats`
  3. `obs_history_and_future` — history stride, future rows, privileged order, hidden-zero
  4. `target_generator_kinematics` — stages 5–7 closed-form paths (fixed, monotone,
     speed-capped, analytic velocity == finite differences, no drone steering),
     stages 2/3/4/8 containment, stage-8 evasion
  5. `prediction_suite` — `const_vel` exact, `linear_ridge` curvature recovery,
     registry fallback, history contract
  6. `reward_terms_sanity` — kill/time bonus, stage-1 dense weights, `r_facing` gating
  7. `curriculum_advance_cap_rollback` — advance/cap/rollback/disable/roundtrip/predecessors
  8. `results_and_eligibility` — dict+attr access, round trip, denied/skipped/eligible
  9. `checkpoint_manager_roundtrip` — layout, run state, training-meta sidecar, discovery
  10. `weight_transfer_suite` — 12-column prefix copy, small-random tail, plan shape
  11. `config_and_model_loader_validation` — both loaders, hashing, stage overrides,
      spawn aliases, plan-schema canonicalisation
  12. `future_source_true_vs_pred` — stage 5 `3.6e-15`, stage 6 `1.3e-3`,
      stage 7 `4.0e-3`, stages 3/8 exactly `0.0`
- Source fixes the suite exposed (all non-blocking, recorded in `BUGS.md`):
  - `smoke_test.py` was missing its `if __name__ == "__main__": sys.exit(_run_all())`
    guard, so it exited 0 silently.
  - `config_loader.parse_range` now also accepts the `[lo, hi]` YAML sequence form
    (previously only a number or `{min:, max:}`).
  - New `_assign_stage_field` normalises spawn ranges to `(lo, hi)` tuples for both the
    flat stage aliases and nested `spawn:` overrides.
  - `model_loader.load_models` unwraps a single-key `models:`/`configs:` mapping so
    pre-split files keep loading; a missing `algo` now says so explicitly.

**Verified:** `C:\Python314\python.exe scripts/smoke_test.py` → `12 passed, 0 failed`
(exit 0) from `interceptor-training/`.

---

## 2026-10-04 — Phase H: R-1 ESC diagnostic script ✅

**Why:** H-7 — the R-1 decision (keep `PH_BAT`, preserve Stage-1 parity) promised a
diagnostic so the flaw is quantified instead of silently inherited.

**Changed**
- NEW `interceptor-training/scripts/esc_diagnostic.py` — read-only, NumPy-only.
  Prints the `Omega_ss(c)`/thrust sweep at `U_bat = 4.2 V`, the slope at the hover
  command and at full command, the hover-thrust residual and the monotonicity verdict;
  `--csv PATH` exports the 201-point sweep, `--rows N` sets the table resolution.
  It never mutates the constants.

**Verified:** `Omega_ss(0.02) = 2400.752`, `Omega_ss(PH_C_HOVER) = 1956.211`,
`Omega_ss(1.0) = 472.220` rad/s, `dOmega_ss/dc = -1995.230` at hover, hover thrust
error `+5.6e-17 N`, monotonically decreasing over `[cut, 1] = True`.

---

## 2026-10-04 — Phase H: `BUGS.md` refreshed ✅

**Why:** D-58 — the bug register must reflect the D-2 → D-58 work.

**Changed** (`BUGS.md`, repository root)
- Review header updated (2026-10-04) with the smoke-suite outcome.
- Item 8 (`Monitor` truncating its CSV on resume) marked **FIXED** — Phase D now
  passes `override_existing=False`.
- Items 14–19 added: missing `__main__` guard in the smoke suite, `parse_range`
  rejecting `[lo, hi]`, stage-override spawn aliases skipping range normalisation,
  `load_models` rejecting the pre-split `models:`/`configs:` wrapper, the stray
  `interceptor-training/src.zip`, and `future_source` being the string
  `"true"`/`"pred"` rather than a bool.

**Verified:** reviewed against the fixes applied in this phase; item 13 (R-1) now
points at the new diagnostic.

---

## 2026-10-04 — Phase H: `README.md` refreshed ✅

**Why:** H-8 — the README still described the pre-D-2 layout (single
`default_run.yaml`, 13-item bug list) and quoted physical constants that no
longer match the code.

**Changed**
- Physical-parameters table re-derived from `src/physics/constants.py`: inertia
  `diag([2.3951e-5, 2.3951e-5, 3.2347e-5])`, rotor inertia `2.0e-9`,
  `PH_C_L = 2.618e-8`, `PH_C_D = 5.45e-11`, `PH_K_MOT = 0.02 s`,
  `PH_OMEGA_MAX = 2800 rad/s`, rate-PID gains and `PH_OMEGA_HOVER`
  (the old table claimed `1.4e-5 / 2.17e-5` inertia, `5.0e-8` lift and
  `1.25e-9` drag — all wrong).
- R-1 note rewritten with the three measured `Omega_ss` endpoints, the reason
  `PH_BAT` is deliberately untouched (parity) and the
  `python scripts/esc_diagnostic.py` command.
- `vecenv="auto"` documented in the `train_model()` signature.
- Warning added next to `observation:` that stage 8 requires
  `future_source: pred` (no closed-form path exists for the evasive target) and
  that `true` leaks the answer on stages 3–7.
- Docker quick start corrected to comma-separated `--stages 1,2,3,4`, added the
  `--continue-from` example and the `train.py` exit-code contract;
  `scripts/esc_diagnostic.py` added to the layout tree; bug-list count updated to
  19 items (8, 11–12, 14–18 fixed); noted that the smoke suite needs no torch.
- Fixed a stale code comment in `src/physics/constants.py`: the `PH_C_HOVER`
  comment read `# ~0.37` while the value is `0.23253743635354834`.

**Verified:** re-read end to end against the CLI flags in `scripts/train.py`,
the `ModelResult` surface in `src/results.py` and the constants module.

---

## 2026-10-04 - Phase I: clean sync to `github_repos/rl-drone-flight-simulator` and push

**Why:** the authoritative deliverable is the public repository
(`https://github.com/siddhmehta5131/interceptor-drone-rl.git`, branch `main`), not the
scratch workspace, which still holds the legacy Swift/PyBullet simulators.

**Changed**
- Copied the 8 files that had drifted since the previous sync: `BUGS.md`,
  `README.md`, `interceptor-training/scripts/smoke_test.py`,
  `interceptor-training/src/physics/constants.py`,
  `interceptor-training/src/utils/config_loader.py`,
  `interceptor-training/src/utils/model_loader.py` and the two configs that the target
  tree had modified locally (now reverted to the workspace versions).
- Added 6 previously untracked files: `IMPLEMENTATION_PROGRESS.md`, `VERSIONS.md`,
  `complete_changelog.md`, `implementation_plan.md`,
  `implementation_plan_d2_d58.md` and
  `interceptor-training/scripts/esc_diagnostic.py`.
- Deliberately NOT synced (legacy / superseded, already excluded in earlier syncs):
  `swift_*.py`, `compare_*.py`, `demo.py` / `default_demo.py` /
  `updated demo.py`, `validate*.py`, `self_test.py`, `identify_params.py`,
  notebooks, images, `Docker_Setup_and_Running_Guide.md` (superseded by the repo's
  `DOCKER_GUIDE.md`), `run_summary.json`, `references.txt`, root
  `requirements.txt`, root `src/` and the versioned `context_v*.md` copies.
- Deleted the stray `interceptor-training/src.zip` (151 KB duplicate of `src/`).

**Notes**
- Tag `v6.0` already exists as an annotated tag on `2a04093`
  ("feat: implement D-2 to D-58 all design decisions") and was left in place; this
  sync is a follow-up commit, so the tag was not moved.
- The target tree had uncommitted local edits to `configs/config.yaml`
  (`device: cuda`, `on_capped: stop`, `future_samples: 0`) and
  `configs/model.yaml` (a `td3_baseline` block replacing `ppo_baseline`). These
  were scratch experiments, not part of the workspace state, so they were reverted;
  copies are kept outside the repo at
  `%TEMP%/rl_target_scratch_config/`.

**Verified:** `python scripts/smoke_test.py` re-run **from the synced repository**
(not the workspace) -> `12 passed, 0 failed` (exit 0).

---

## 2026-10-04 - Phase I: Docker build, container validation and fresh-clone check

**Why:** the SB3 paths (policy construction, `learn`, `PPO.load`, checkpoint
sidecars, TensorBoard events) cannot be exercised on the Windows host, which has
no torch. Phase I closes that gap and proves a fresh clone of the published
repository is trainable.

**Changed**
- No source changes; one new defect was found and logged as `BUGS.md` item 20
  (`--config` / `--models` default to `/data/configs/...` and fail outside
  the compose mount; workaround is to pass the repo-relative paths).
- Docker Desktop was not running on this host, so it was started before the build.

**Verified**
- `docker build -t interceptor-drone:v6.0 .` from the repository's
  `interceptor-training/` -> exit 0 (base image
  `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime`, daemon 29.8.1, linux, 4 CPUs).
- Container smoke train, `--smoke --smoke-models ppo_baseline --smoke-stages 1,2
  --smoke-steps 512`, artefacts written to a host-mounted `/data`:
  exit 0, `974 steps in 12.8 s`, stage 2 seeded from the stage-1 checkpoint with
  `copied 25, padded 4` (14 -> 105 observation columns, 12-column shared
  prefix), both stages `capped` as expected for a 512-step budget.
  Full artefact tree present: per-stage `*.zip` + `*.training_meta.json`,
  `results/ppo_baseline/ppo_baseline.{zip,result.json}`,
  `run_state.json`, `retries.json`, monitor CSVs and TensorBoard event files.
- `scripts/evaluate.py --run <run> --stage 2 --episodes 5` -> exit 0; the saved
  asymmetric policy reloaded through `PPO.load`, rolled out deterministically
  and wrote `eval_ppo_baseline_stage_2.json`. (Metrics are meaningless for a
  512-step model; the point of the run is the load/predict path.)
- Full suite inside the image (torch 2.7.0 + SB3 present): `12 passed, 0 failed`.
- Fresh `git clone` of `https://github.com/siddhmehta5131/interceptor-drone-rl.git`
  into a new directory -> HEAD `254f621`, 62 files. Suite passes `12/12` on the
  host **and** in the container, and a container smoke train from the clone exits 0.

**Pushed:** commit `254f621` on `main`; tag `v6.0` (annotated) remains on
`2a04093` and was not moved.
