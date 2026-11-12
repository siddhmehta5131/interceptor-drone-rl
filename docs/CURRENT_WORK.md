# CURRENT_WORK — APN + RL Autonomous Interceptor Drone

Implementation of `updation_plan.md` v1.0: a pivot **from** a physics-based
quadrotor simulator (RK4 / PID / motor RPM / Crazyflie) **to** a pure 3-D
path-planning interceptor that combines classical **Augmented Proportional
Navigation (APN)** with a **PPO residual** policy. The environment emits velocity
commands (`vx, vy, vz, ωx, ωy, ωz`), not motor commands.

Status: **all 12 phases complete; suite green.** The previous physics project was
archived to `C:\Users\ADMIN\Desktop\projects\github_repos\rl-drone-flight-simulator`
(commit `493d370`, remote `github.com/siddhmehta5131/interceptor-drone-rl`) and
the old tree was deleted here per plan §5.1. This workspace is a fresh
implementation.

---

## Deliverable layout

```
├── run.py                      ← one-file training runner (edit this)
├── conftest.py                 ← pytest sys.path bootstrap + 'slow' marker
├── requirements.txt            ← runtime + training deps (plan §18.1)
├── configs/
│   ├── stages.yaml             ← 3-stage curriculum (authoritative table)
│   ├── model.yaml              ← PPO architecture + 6 privileged critic fields
│   ├── config.yaml             ← global/obs/interceptor/apn/stages settings
│   └── visualiser.yaml         ← PyVista scene / camera / HUD settings
├── src/
│   ├── api.py                  ← train_model() / evaluate_model()
│   ├── results.py              ← TrainResult / ModelResult / StageOutcome
│   ├── guidance/               ← apn.py, kinematic_model.py, optimal_trajectory.py
│   ├── envs/                   ← stage_config, target_generator, obs_builder,
│   │                             reward, intercept_env
│   ├── prediction/             ← base, const_vel, kalman
│   ├── training/               ← orchestrator, callbacks, curriculum,
│   │                             asymmetric_policy, checkpoint_manager,
│   │                             weight_transfer
│   └── utils/                  ← config_loader, model_loader, logger
├── visualiser/                 ← renderer, animator, scene_builder, cli
├── scripts/                    ← train, evaluate, visualise, smoke_test,
│                                 download_models
├── tests/                      ← pytest suite (unit + integration)
├── models/                     ← interceptor_drone.obj, target_drone.obj
└── data/                       ← runs/, videos/, evaluation/
```

## Build phases — status

| Phase | Deliverable | Status |
|-------|-------------|--------|
| 0 | Backup old project to GitHub | DONE — `493d370`, repo renamed `interceptor-drone-rl` |
| 1 | Clean-slate deletion (plan §5.1) | DONE — old physics tree removed, index cleared |
| 2 | New project structure (plan §6) | DONE |
| 3 | APN engine (`apn.py`, `kinematic_model.py`, `optimal_trajectory.py`) | DONE |
| 4 | Gymnasium env (`intercept_env.py`) | DONE — obs 101, action 3 |
| 5 | Observation builder (16-dim frame, history, future, privileged) | DONE — actor 85 |
| 6 | Reward (`reward.py`) | DONE — 13 tunable terms |
| 7 | Optimal-trajectory oracle | DONE — trapezoidal + cubic-Hermite |
| 8 | PPO training pipeline (orchestrator, policy, callbacks) | DONE |
| 9 | 3-stage curriculum (`configs/stages.yaml`, scheduler) | DONE |
| 10 | PyVista 3D visualiser + MP4 export | DONE |
| 11 | Evaluation (`scripts/evaluate.py`) | DONE |
| 12 | Tests + smoke suite; docs | DONE |

## Verified behaviour

* `python scripts/smoke_test.py` → **11/11 PASS**.
* `python -m pytest tests/ -q -m "not slow"` → **52 passed, 1 deselected**.
* `python -m pytest tests/ -q` (incl. end-to-end) → **53 passed**.
* Observation dimensions: actor **85**, privileged **16**, env **101**;
  action space **(3,)** — asserted for every stage.
* **Pure-APN** (zero RL action) rollouts: stage 1 **12/12** kills, stage 2
  **12/12**, stage 3 **0/12** — APN suffices for predictable targets; the RL
  residual is required for evasive ones.
* Visualiser writes valid MP4s off-screen. Two demo animations rendered
  (APN-only, no trained model): `data/videos/stage1_static.mp4` (68 frames) and
  `data/videos/stage2_polynomial.mp4` (105 frames).
* Real drone meshes are wired up with **per-model transforms**: raw units differ
  (interceptor 12.4×3.4×8.8, target 3.1×2.0×3.9), so each uses its own scale
  (`interceptor_scale: 0.1612`, `target_scale: 0.5114`, ~2 m max dimension) and
  proper cube-rotation code (`interceptor_rotation: I08`, `target_rotation: T21`)
  to render upright and front-facing.

## Key implementation choices (vs. the plan)

1. **APN form.** The plan's literal `N'(V_rel × ω_LOS)` cross-product is not
   dimensionally consistent along the LOS; the equivalent zero-effort-miss form
   `N'·perp(V_rel)/t_go + (N'/2)·perp(a_T)` is implemented.
2. **`t_go` bound.** Added `t_go_max = 20 s` (clip) — without it `t_go` explodes
   when the interceptor opens the range and ZEM (and reward) blow up.
3. **ZEM normalisation.** Reward uses `−k_ZEM·|ZEM|/max(distance,1)` instead of
   the raw metric term.
4. **Windows vec-env.** `DummyVecEnv` (SB3 `SubprocVecEnv` needs a `__main__`
   guard on Windows); `n_parallel_envs = 8`.
5. **Models & orientation.** `models/*.obj` are real user-supplied meshes in
   mixed units, so the visualiser applies **per-model scale and rotation** (see
   "Verified behaviour"). `scripts/download_models.py` can still emit
   placeholders but only overwrites existing files with `--force`.
6. **Shadows opt-in.** Plan §14.3 called `enable_shadows()` unconditionally; on
   this headless GLES host the shadow shaders spam-error and roughly double
   render time, so shadows are now opt-in (`visualiser.yaml` → `lighting.shadows`).
   The visualiser CLI also gained `--resolution WxH` (e.g. `1280x720`) to speed
   up CPU-only renders.

## How to run

```bash
pip install -r requirements.txt

python scripts/smoke_test.py                                   # 11/11 fast checks
python -m pytest tests/ -q -m "not slow"                       # 52 unit tests
python -m pytest tests/ -q                                     # + end-to-end

python scripts/train.py --stages 1 --name smoke --steps 20000  # smoke train
python scripts/train.py --stages 1,2,3 --name apn_rl_v1        # full curriculum

python -m visualiser --model apn --stage all --output-dir data/videos
python -m visualiser --model apn --stage 1 --resolution 1280x720 \
  --output data/videos/stage1_static.mp4
python scripts/evaluate.py --model data/runs/apn_rl_v1/ppo_interceptor/final.zip \
  --stage 3 --episodes 100 --render
```

## Model orientation workflow

`models/*.obj` do not share an axis convention or unit system, so each needs a
rotation code + scale. To re-derive (or set up new assets):

1. Run the orientation-shot generator (normalises each mesh to ~2 m, renders all
   **24 proper cube rotations**, labels each with a code `I00..I23` / `T00..T23`
   and a red=+X / green=+Y / blue=+Z triad) and inspect the contact sheets.
2. Pick the code where the drone is upright and front-facing (+X forward).
3. Set `models.interceptor_rotation` / `target_rotation` (and the matching
   `interceptor_scale` / `target_scale`) in `configs/visualiser.yaml`.

Helpers live in `visualiser/scene_builder.py` (`cube_rotations`,
`rotation_matrix_from`, `apply_mesh_rotation`). Currently configured:
`I08`, `T21`.

## Open items / next work

* Full training budgets are large (2 M / 5 M / 10 M steps). Only a smoke budget
  has been run on this CPU host; the real curriculum should run on a GPU host.
* `models/*.obj` are placeholders; drop in real meshes (see `models/README.md`).
* Non-blocking defects are tracked in `BUGS.md` (VTK shader spam,
  `weight_transfer.py` stale constants).
