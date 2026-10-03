# CURRENT_WORK — RL-based Autonomous Interceptor Drone Training Pipeline

Implementation of `implementation_plan.md` v1.0. All deliverables live in
`interceptor-training/`. Status: **all phases complete — code complete, host-side
smoke suite green (8/8), critique-review remediation merged, in-container CUDA
smoke passed (Phase 9), repo published on GitHub as
`siddhmehta5131/interceptor-drone-rl`, beginner Docker setup guide shipped.
Ready for real multi-stage training.**

---

## Deliverable layout (`interceptor-training/`)

```
interceptor-training/
├─ Dockerfile                 # pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
├─ docker-compose.yml         # nvidia runtime, GPU reservation, ./data:/data
├─ requirements.txt           # torch/SB3/gymnasium/numpy/scipy/tensorboard/pyyaml
├─ configs/default_run.yaml   # global + 4 algo configs (ppo, sac, td3, ppo_deep)
├─ src/
│  ├─ physics/   constants, quaternion, aero, pipeline   (port of hover_env._ph_*)
│  ├─ envs/      stage_config, obs_builder, reward, target_generator, base_env
│  ├─ training/  curriculum, checkpoint_manager, callbacks, orchestrator
│  └─ utils/     config_loader, logger
├─ scripts/train.py           # entrypoint (+ --smoke for in-container smoke)
├─ scripts/evaluate.py        # model evaluation against a stage
├─ scripts/smoke_test.py      # host-side no-SB3 verification suite (8 tests)
└─ data/                      # host-mounted volume skeleton (checkpoints/,
                              # tb_logs/, curriculum_state/, results/)
```

## Build phases — status

| Phase | Deliverable | Status |
|-------|-------------|--------|
| 1 | Physics extraction (constants, quaternion, aero, pipeline) | DONE — bit-identical to `hover_env._ph_*` (parity script, max diff 0.0) |
| 2 | Core env framework (`stage_config`, `obs_builder`, `reward`, `base_env`) | DONE |
| 3 | Target generator (static/waypoint/order-1/2/3/evasive) | DONE |
| 4 | Stage-1 parity vs `AltitudeHoldEnv` | DONE — 5000-step loop `np.array_equal` obs, `==` reward, identical term/trunc/info, PASS |
| 5 | Stages 2–8 env verification | DONE — smoke suite covers all stages |
| 6 | Curriculum scheduler + checkpoint manager + resume logic | DONE — advances/caps/rollback tested |
| 7 | Orchestrator + callbacks + TensorBoard | DONE — static review + 5 fixes (see BUGS.md); runtime test pending container |
| 8 | Docker infra + configs | DONE |
| 9 | Integration smoke (1 config, stages 1–3, 1000 steps each) | DONE — `run_summary.json` 2026-09-30; two runtime bugs found & fixed (see below) |
| 10 | CURRENT_WORK.md / BUGS.md | DONE |
| 11 | Critique-review remediation | DONE — all findings addressed; smoke suite re-run green |

## Verified behaviour (host, Windows: Python 3.14 / numpy / scipy / gymnasium only)

- `python scripts/smoke_test.py` → **8/8 PASS, exit 0** (post-remediation):
  parity_stage1_vs_hover, all_stage_envs_run, obs_history_stacking,
  target_generator_kinematics, reward_terms_sanity,
  curriculum_advance_cap_rollback, checkpoint_manager_roundtrip,
  config_loader_validation. All `src/` + `scripts/` files pass `py_compile`.
- Stage-1 parity is exact: same seed + same action stream → identical obs
  arrays, rewards, and infos over a full 111-step episode.
- Key implementation notes:
  - Obs arithmetic stays float64 until a single `np.array(..., float32)`
    cast at the end (matches hover bit-for-bit).
  - Reward terms are summed with an explicit left fold, NOT `builtins.sum()`
    (Python 3.12+ compensated summation differs by 1 ULP).
  - `PH_C_HOVER = 0.23253743635354834`, `PH_OMEGA_HOVER = 1956.211093185006`.

## Critique-review remediation (phase 11)

A code review of the phase-1–10 build found deviations from
`implementation_plan.md`; all were verified and fixed:

1. **Reward formulas → plan §5.5.** Kill bonus is now a single term
   `max(0, k_kill_bonus·max(0,1−d/kill_radius)·(1−t/T))` (miss-distance gate
   stages 3+, time gate stages 4+). The old 0.2×time bonus and near-miss
   terms were removed. Progress normalize (stages 6+) uses
   `max(distance,1)`. Stage-1 parity unchanged.
2. **Obs history stacking → plan §5.3.** Current-first
   `[t, t-s, t-2s, …]` with trailing zero-pads; hidden-target slots stay
   exactly zero even under sensor noise (§5.4). Smoke suite asserts both.
3. **Config loader accepts plan schema.** `algorithm`/`policy_kwargs
   {net_arch, activation_fn}`/`obs_history {m,s}` and SB3-named
   hyperparameters canonicalize to the internal schema; a shared `net_arch`
   list expands to `{pi, vf}`; lowercase activation aliases normalise.
4. **Default config path fixed** to `/data/configs/default_run.yaml`
   (Dockerfile CMD, `train.py`, `evaluate.py`) — the old `/data/configs/run.yaml`
   did not exist, which would have crashed the container at startup.
5. **Resume state schema unified** to the plan name set (`current_stage`,
   `current_stage_steps`, `total_steps_all_stages`, `last_checkpoint_path`);
   reader/writer agree, a periodic-checkpoint branch is consulted before the
   transfer path, and off-policy replay buffers are persisted/restored.
6. **Evasive target flees the drone** (`pos − drone_pos`); containment
   places targets exactly on the boundary with radial velocity reflection;
   `predicted_pos()` includes the jerk term. Smoke suite asserts flee
   direction and containment.
7. **`make_env_factory` takes the effective `StageConfig`** (with YAML
   overrides merged) instead of re-looking-up the built-in table — override
   plumbing now reaches the env end-to-end. Callers in orchestrator and
   evaluate updated.
8. **Evaluation fixed**: model precedence is the stage's own final →
   `latest_periodic` → config-wide final; runs the effective stage config;
   reports kill rate / success rate / mean final distance / time-to-intercept
   (steps + seconds) / miss distance and persists to
   `data/results/eval_<config>_stage_<N>.json`.
9. **Logger hierarchy**: all module loggers are `interceptor.*` children;
   the rotating file handler attaches once to the `interceptor` root and
   therefore captures every child logger.
10. **Callbacks** take `log_interval` explicitly (no SB3 introspection).

## How to run

Host smoke (no GPU/torch needed):

```
cd interceptor-training
python scripts/smoke_test.py
```

Container training (CUDA host):

```
docker compose up --build            # trains configs/default_run.yaml
# or override the config file:
docker compose run --rm interceptor-train --config /data/configs/default_run.yaml
```

Evaluation (in container):

```
docker compose run --rm --entrypoint python interceptor-train scripts/evaluate.py \
  --config /data/configs/default_run.yaml --config-name ppo_baseline --stage 3 --episodes 50
```

## End-of-day status — 2026-09-25

**Repo published on GitHub.** The merged repository (simulator + reference env +
Dockerized pipeline) was assembled into a clean, committed clone and pushed to
`https://github.com/siddhmehta5131/interceptor-drone-rl` (branch `main`) as three
commits:

- `5debf50` feat: add Dockerized RL interceptor training pipeline (8-stage SB3 curriculum)
- `3075a74` docs: rewrite README for merged simulator + RL training repo
- `433d5b9` docs: add Docker setup and running guide (Ubuntu + NVIDIA + CUDA training)

The published repo contains only the essential files (no scratch); the local
workspace keeps the unsynced scratch copies. Verified before pushing: smoke suite
run from the published clone's `interceptor-training/` → **8/8 PASS, exit 0**.
Repo name was renamed on GitHub from `rl-drone-flight-simulator` → `interceptor-drone-rl`.

**Docker setup guide shipped.** `Docker_Setup_and_Running_Guide.pdf` (+ `.md`
source) created for a first-time user: Ubuntu + NVIDIA driver + Docker +
NVIDIA Container Toolkit install, clone, in-container smoke, background training,
TensorBoard, evaluation, CPU-only fallback, quick-reference and troubleshooting
tables. Saved in the workspace **and** committed to the repo (`433d5b9`).

**Remaining:** Phase 9 — run the in-container integration smoke on the Ubuntu GPU
box exactly as the guide describes, then start the real training
(`docker compose up -d --build`).

## Integration smoke (Phase 9) — COMPLETE

Run on CUDA host 2026-09-30 (`run_id: 20260930-130451`). `ppo_baseline`
stages 1→2→3, ~1000 steps each, all `capped` with `success_rate: 0.0`
(expected for a smoke budget). Final model saved to
`/data/results/ppo_baseline_final.zip`. Wall time: 2.7 s.

Two runtime bugs discovered and fixed:

- **Bug #11:** `_atomic_json` used `fd, tmp = os.open(...)` — `os.open`
  returns a single `int`, not a tuple. Fixed: path constructed first, then
  `fd = os.open(str(tmp), ..., 0o644)`.
- **Bug #12:** `opencv-python` (from `sb3[extra]`) needs `libxcb.so.1` and
  other GUI libs absent from the container. Each `SubprocVecEnv` worker
  crashed on `import cv2`, silently falling back to `DummyVecEnv`
  (sequential, single-process). Fixed: Dockerfile now replaces
  `opencv-python` with `opencv-python-headless`.

Additional observation: SB3 warns that PPO with `MlpPolicy` may run faster
on CPU. Once `SubprocVecEnv` is restored, benchmark `device: cpu` vs
`device: auto` (cuda) for the PPO configs.

## Config coverage (`configs/default_run.yaml`)

- Global: seed 42, device auto, data_dir /data, 8 parallel envs, checkpoint
  interval 50k steps, tensorboard on, sequential mode, rollback
  {max_attempts 2, retry_budget_scale 0.5, threshold_scale 0.5,
  min_steps_scale 0.25}.
- Stages: all 8 stage definitions built into `src/envs/stage_config.py` with
  the plan's tuned reward weights, spawn ranges, thresholds, windows, rates
  and step caps (no YAML overrides needed).
- Configs: `ppo_baseline`, `sac_baseline`, `td3_baseline`,
  `ppo_5layer_deep` — all with obs history (m3/s2; deep PPO m5/s3), stages
  1–7.
- Optional DR/wind/ground-effect stage flags default **off** (Stage 1 stays
  bit-identical to the reference `hover_env.py`).

## Session log — Phase 0 discovery (2026-10-02)

Read-only discovery for the repo-clean/sync/push/Docker engagement
(role.txt inputs: NEW = `context_v5_0.md`, OLD = `context_v0.0.md`;
the TODO to-do list was **not** supplied — Phase 1 planning is blocked
on it).

- Mode: command execution available (Windows PowerShell 5.1 build 26100,
  Windows NT 10.0.26200).
- Tools: git 2.53.0.windows.2; Docker 29.8.1 + Compose v5.5.1 — **daemon
  not running** (npipe missing); Python 3.14.6 (`C:\Python314`);
  gh 2.98.0 authenticated as `siddhmehta5131` (scopes: gist, read:org,
  repo). Free space on C: 38 GB.
- Target repo: `C:\Users\ADMIN\Desktop\projects\github_repos\rl-drone-flight-simulator`,
  remote `https://github.com/siddhmehta5131/interceptor-drone-rl.git`,
  PUBLIC, default branch `main`, 8 commits, working tree clean,
  local `main` == `origin/main`, last push 2026-09-30T13:30:13Z,
  45 tracked files / 673,043 B, no `.gitattributes`, no submodules,
  `git lfs ls-files` returned no entries.
- Workspace vs repo: 96 workspace files (excl. `.git/`, `__pycache__/`);
  51 exist only in the workspace; 0 exist only in the repo; 10 differ
  byte-wise: `.gitignore` (528 vs 807 B), `BUGS.md` (6384 vs 5374),
  `context.md` (88641 vs 4672), `CURRENT_WORK.md` (10073 vs 9596),
  `interceptor-training/configs/default_run.yaml` (1967 vs 1932),
  `interceptor-training/Dockerfile` (1359 vs 1427),
  `interceptor-training/requirements.txt` (332 vs 314),
  `interceptor-training/src/training/orchestrator.py` (23213 vs 23224),
  `README.md` (6571 vs 11280), `requirements.txt` (47 vs 441).
- Workspace git: unborn `HEAD` (no commits), 53 staged (`A`) entries.
- This log entry is the only workspace file touched in Phase 0; the
  original staged content is recoverable with
  `git show :CURRENT_WORK.md > CURRENT_WORK.md`.

## Session log — Phase 1 plan delivered (2026-10-02)

- The TODO file was never supplied; on the user's "continue" the work
  list was **derived from NEW itself** (§13.9 stale-dependents table,
  Decision Log D-1…D-58, Session Handoff FIX/TO-DO/PROPOSED lists),
  yielding work items T-01…T-23. Substitution disclosed as Assumption
  A-1 / Open question O-0 in the plan — confirmation required at GATE.
- Phase 1 deliverable written: `plan_2026-10-02.md` (sections 0–11 per
  role.txt: backups, work breakdown + T→S traceability, 96-file
  inventory, target repo tree, Git/Docker strategy, 11 verification
  gates, risk register, rollback, 15-line DoD, 17 open questions,
  6 assumptions).
- Scope stance recorded: decided-but-unimplemented NEW decisions are in
  scope; PROPOSED/OPEN values are questions (never guessed); NEW's own
  spec-body FIX pass is out of scope by default (O-6).
- Workspace files touched in Phase 1: `CURRENT_WORK.md` (this log) and
  new `plan_2026-10-02.md`. Nothing else; no backups taken yet (S-00 is
  the first Phase 2 step); no deletions; no git/push/docker actions.
- Status: **GATE — waiting for literal "APPROVED"** before Phase 2.