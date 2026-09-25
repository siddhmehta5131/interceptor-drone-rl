# CURRENT_WORK — RL-based Autonomous Interceptor Drone Training Pipeline

Implementation of `implementation_plan.md` v1.0. All deliverables live in
`interceptor-training/`. Status: **build complete — code complete, host-side
smoke suite green, critique-review remediation merged, container integration
run still to be executed on a CUDA/Docker host.**

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
| 9 | Integration smoke (1 config, stages 1–3, 1000 steps each) | **PENDING** — needs CUDA/Docker host (see below) |
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

## Integration smoke (Phase 9) — blocked locally

This machine has no Docker and no NVIDIA GPU, so `train.py` cannot run locally
(Python 3.14 has no torch/SB3 wheels; training requires the Linux+CUDA
container). The `--smoke` path exists and is implemented:

```
docker compose run --rm interceptor-train --smoke \
  --smoke-config ppo_baseline --smoke-stages 1,2,3 --smoke-steps 1000
```

Expected outcome per stage: 1000 steps → budget exhausted → CAPPED → stage
final saved → advance to next stage → config completed → `run_summary.json` +
`results/ppo_baseline_final.zip` written under `data/`.

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