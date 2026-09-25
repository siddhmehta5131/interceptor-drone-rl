# CONTEXT — RL-Based Autonomous Interceptor Drone

**Last updated:** 2026-09-25 (end of day)
**GitHub repo:** https://github.com/siddhmehta5131/interceptor-drone-rl (branch `main`)
**Repo owner/author:** Siddh Mehta (siddhmehta5131)

---

## 1. What this project is

A single repo that bundles three things:

1. **Flight simulator** (`src/simulation.py`, `src/visualiser.py`) — original 9-stage
   Crazyflie-style drone simulator with coefficient fitting (`docs/coefficient_fitting.md`).
2. **Reference Stage-1 env** (`hover_env.py`) — gymnasium altitude-hold env, the
   bit-exact parity target for the pipeline's Stage 1.
3. **Dockerized multi-stage RL training pipeline** (`interceptor-training/`) — the main
   deliverable: an 8-stage curriculum that trains an interceptor drone policy with
   Stable-Baselines3 (PPO/SAC/TD3) on a GPU via Docker + CUDA.

## 2. Where the code lives (two locations)

| Location | Role |
|---|---|
| `C:\Users\ADMIN\Desktop\projects\RL based autonomous interceptor drone` | **Workspace.** Full working copy incl. scratch files. Its own git repo (`master`, zero commits, no remote) — effectively unsynced. |
| `C:\Users\ADMIN\Desktop\projects\github_repos\rl-drone-flight-simulator` | **Published clone.** Clean, committed copy that tracks `origin` = GitHub. All pushes happen here. |

Git history on GitHub (`main`, 3 commits):

| Commit | Message |
|---|---|
| `5debf50` | feat: add Dockerized RL interceptor training pipeline (8-stage SB3 curriculum) |
| `3075a74` | docs: rewrite README for merged simulator + RL training repo |
| `433d5b9` | docs: add Docker setup and running guide (Ubuntu + NVIDIA + CUDA training) |

## 3. Pipeline facts (implementation_plan.md v1.0)

- **Stages:** 8 curriculum stages in `interceptor-training/src/envs/stage_config.py`;
  default run trains stages 1–7. Stage 1 must stay **bit-identical** to `hover_env.py`.
- **Observation:** Option A — 14-dim (stage 1) growing to 76-dim (history-framed
  stages); fresh network created per stage when obs dims change; weights transferred
  with `set_parameters(exact_match=True)` when dims match.
- **Physics:** drone = classical **RK4** (`src/physics/pipeline.py`, port of
  `hover_env._ph_rk4`); target motion = first-order Euler (known asymmetry, BUGS.md #7).
- **Reward (plan §5.5):** kill bonus `k_kill_bonus·max(0,1−d/kill_radius)·(1−t/T)` with
  the two gates enabled by stage config; progress normalized by `max(distance,1)`.
- **Curriculum:** advance at success-rate threshold; CAPPED on budget exhaustion;
  rollback below 50% threshold, max 2 attempts.
- **Algo presets** (`configs/default_run.yaml`): `ppo_baseline` (default, stages 1–7),
  `sac_baseline`, `td3_baseline`, `ppo_5layer_deep`. Global: seed 42, 8 parallel envs,
  checkpoint every 50k steps, TensorBoard on.

## 4. Hard invariants (must never regress)

- Stage-1 reward uses an **explicit left-fold** sum, NOT `builtins.sum()` (1-ULP parity).
- Observation arithmetic stays **float64** until the final `float32` cast.
- `PH_C_HOVER = 0.23253743635354834`, `PH_OMEGA_HOVER = 1956.211093185006`.
- Obs history stacks current-first `[t, t-s, …]` with trailing zero-pads; hidden-target
  slots exactly zero even under sensor noise.
- Default config path is `/data/configs/default_run.yaml` (Dockerfile CMD + train.py).

## 5. Environment constraints

- Development host: **Windows, Python 3.14, no Docker, no NVIDIA GPU, no torch/SB3.**
- Therefore the runnable verification is **`python scripts/smoke_test.py` → 8/8 PASS**
  (host-side, no SB3). SB3 runtime paths are statically reviewed + `py_compile` only.
- **Phase 9 (in-container integration smoke) is still PENDING** — it must run on the
  Ubuntu + NVIDIA GPU machine using `docker compose run --rm interceptor-train --smoke ...`.

## 6. How to run (quick reference)

```
# Host smoke (any machine):
cd interceptor-training && python scripts/smoke_test.py

# GPU training on the Ubuntu box (full steps in Docker_Setup_and_Running_Guide.pdf):
cd interceptor-training
docker compose up -d --build
docker compose logs -f
```

Training outputs land in `interceptor-training/data/` (checkpoints/, tb_logs/,
curriculum_state/run_state.json, results/).

## 7. Documentation set

| File | Purpose |
|---|---|
| `implementation_plan.md` | The plan (v1.0) the pipeline implements |
| `CURRENT_WORK.md` | Build status, verified behaviour, remediation summary |
| `BUGS.md` | Known non-blocking issues / design compromises |
| `Docker_Setup_and_Running_Guide.pdf/.md` | Beginner walk-through: Ubuntu + Docker + NVIDIA setup |
| `VERSIONS.md`, `complete_changelog.md` | Older simulator-side version/history notes |