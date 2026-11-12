# APN + RL Interceptor Drone — Project Context

This document is the authoritative context for the APN + RL interceptor project:
the goal, the architecture, the observation/reward design, the known issues and
their fixes, and how to run everything. It supersedes the previous physics-based
quadrotor context (`context_v5_0.md`).

## 1. Project Goal

Build a hybrid **APN + RL** 3D interception system:

* A classical **Augmented Proportional Navigation** guidance law produces the
  main acceleration command from the target's relative kinematics.
* A **PPO** policy (Stable-Baselines3) learns a bounded **residual** correction
  that recovers performance the pure APN law loses against evasive targets.
* A 3-stage curriculum (static → polynomial → evasive) trains the policy.
* A **PyVista** visualiser exports full-length MP4 demonstrations.

The system is a **path-planning** problem: the environment consumes a 3-D
residual-acceleration action and emits a **velocity command**
(`vx, vy, vz, ωx, ωy, ωz`). There is no motor/RPM/PID dynamics — that was the
previous (now-archived) physics project.

## 2. Architecture

```
a_cmd = APN_command + RL_residual
```

### 2.1 APN law (`src/guidance/apn.py`)

For `R = p_T − p_M`, `V_rel = v_T − v_M`, `R̂ = R/|R|`,
`V_c = −V_rel · R̂`, `t_go = clip(R / max(V_c, 0.1), t_go_min, t_go_max)`,
`perp(x) = x − (x·R̂)R̂`:

```
a_APN = N' · perp(V_rel) / t_go  +  (N'/2) · perp(a_T)
```

`N'` (`apn_N_prime`) defaults to 4.0 and rises to 4.5 in stage 3.
`compute_zem()` returns the scalar zero-effort-miss and its perpendicular vector.

### 2.2 Kinematic model (`src/guidance/kinematic_model.py`)

`KinematicModel(v_max, a_max, omega_max, dt)` integrates an `InterceptorState`
(position, velocity, orientation wxyz-quaternion, angular_velocity) with
first-order Euler for translation and slerp for attitude; acceleration is
clamped to `a_max` and speed to `v_max`. `_compute_heading_rate()` in the env
turns an acceleration command into a bounded body-rate command
`ω = cross(v̂, a) / max(|v|, 1)`, clipped to `omega_max`.

### 2.3 Optimal-trajectory oracle (`src/guidance/optimal_trajectory.py`)

`compute_optimal_intercept()` returns a minimum-time, `v_max`/`a_max`-constrained
trajectory (trapezoidal profile for static targets, cubic-Hermite for moving
ones) plus `optimal_time`, `optimal_accel`, `optimal_vel` and `optimal_cost`.
The env feeds these to the privileged critic.

### 2.4 Interceptor / APN config

| Field | Value |
|---|---|
| `v_max` | 50 m/s |
| `a_max` | 15 m/s² |
| `omega_max` | 3.0 rad/s |
| `speed_ratio` | 1.5 |
| `dt` | 0.02 s (50 Hz) |
| `N_prime` | 4.0 (stage 3: 4.5) |
| `t_go_min` / `t_go_max` | 0.05 s / 20 s |

## 3. Curriculum, Observations, Reward

### 3.1 Three-stage curriculum (`configs/stages.yaml`)

| Stage | target_type | world_r | episode | budget | threshold | `rl_action_scale` |
|---|---|---|---|---|---|---|
| 1 | `static` | 150 | 3–5 s | 2 M | 0.95 | 1.5 |
| 2 | `order_1` | 200 | 5–10 s | 5 M | 0.90 | 4.5 |
| 3 | `evasive` | 250 | 8–15 s | 10 M | 0.75 | 7.5 |

Stage 2 target speed 5–20 m/s, accel 0–5 m/s²; stage 3 target speed 10–30 m/s,
3 g accel limit, jerk limit 20 m/s³, evasive probability 0.6, kill radius 1.5 m.

### 3.2 Observation layout

Single 16-dim frame:

| index | field | |
|---|---|---|
| 0–2 | `rel_pos` | target − interceptor |
| 3–5 | `rel_vel` | target_vel − interceptor_vel |
| 6–8 | `interceptor_vel` | |
| 9 | `range` | |
| 10 | `closing_vel` | |
| 11–13 | `LOS_rate` | `(rel_pos × rel_vel)/range²` |
| 14 | `t_go_est` | `clip(range/max(closing,0.1), 0, 20)` |
| 15 | `visible` | 1/0 |

* History block = `16 × (1 + m)`, `m = 3`, `p = 2` → 64 dims (newest first,
  padded with the oldest frame when the buffer is short).
* Future block = `7 × n`, `n = 3`, `q = 5` → 21 dims; each sample is
  `future_rel_pos(3) + future_rel_vel(3) + visible(1)`.
* **Actor obs = 64 + 21 = 85.**
* **Privileged critic = 16** (`time_remaining, optimal_accel, optimal_vel,
  target_true_future_pos, target_true_future_vel, target_true_accel`).
* **Env observation space = 101.** The asymmetric policy slices `[:, :85]` for
  the actor and uses all 101 for the critic.

### 3.3 Action space

`spaces.Box(low=-1, high=1, shape=(3,), float32)` — a world-frame residual
acceleration, scaled by the stage's `rl_action_scale` (m/s²). The **logged
output** (not the action) is the 6-D `velocity_output = [vx, vy, vz, ωx, ωy, ωz]`
in `info` and the trajectory log.

### 3.4 Reward (`RewardConfig`)

Normalised progress, LOS-rate, closing-velocity, energy, RL-residual,
smoothness, time, ZEM, plus a kill bonus scaled by miss accuracy and elapsed
time, and an out-of-bounds crash penalty. All weights are per-stage tunable in
`configs/stages.yaml`.

### 3.5 Environment step order (`src/envs/intercept_env.py`)

`Δa_RL = action × scale` → advance target → `compute_apn_command` →
`a_total = clamp(a_APN + Δa_RL)` → heading rate → `KinematicModel.step` →
log `velocity_output` → check kill / OOB / timeout → reward → obs → append
trajectory-log entry.

## 4. Training

`src.api.train_model()` → `Orchestrator` → `InterceptEnv` (+`Monitor`,
`DummyVecEnv` on Windows) → `AsymmetricActorCriticPolicy`. PPO hyperparameters
live in `configs/model.yaml`. `InterceptMetricsCallback` tracks a sliding
kill-rate window and stops a stage when the threshold is reached or the budget
is exhausted; `CurriculumScheduler` advances/caps/rolls back.

## 5. Issues & Known Fixes

### Issue 1: Plan §2.3 APN cross-product is not dimensionally consistent (FIXED)
The literal `a_M = N'(V_rel × ω_LOS) + …` form has a non-zero along-LOS
component and an inconsistent lateral magnitude. The mathematically equivalent
zero-effort-miss form `a_APN = N'·perp(V_rel)/t_go + (N'/2)·perp(a_T)` is
implemented instead. Verified against static/crossing/accelerating targets.

### Issue 2: `t_go` unbounded explosion (FIXED)
When the interceptor opens the range (`V_c ≤ 0`), `t_go = R/max(V_c,0.1)`
becomes `10·R` and ZEM (∝ `V_rel·t_go`) exploded to ~260 k/step, driving the
reward to ≈ −1.75 M. Fixed with `t_go = clip(R/max(V_c,0.1), t_go_min, t_go_max)`
and a default `t_go_max = 20 s`.

### Issue 3: Raw ZEM reward unbounded (FIXED)
Plan §10.3 used `−k_ZEM · ZEM` directly; the ZEM scale is metric, so the term
dominated. Normalised to `−k_ZEM · |ZEM| / max(distance, 1)`.

### Issue 4: VTK GLES shader warnings during off-screen render (RESOLVED)
VTK emitted `vtkShaderProgram` / `vtkOpenGLPolyDataMapper` ERROR spam on this
headless GPU-less host. Root cause: the unconditional `enable_shadows()` from
plan §14.3 — a headless GLES context cannot compile the shadow shaders. Shadows
are now **opt-in** (`visualiser.yaml` → `lighting.shadows: false`), and the spam
is gone (log ≈ 10 lines). Logged in `BUGS.md`.

### Issue 5: `weight_transfer.py` constants tied to the old obs layout (NON-BLOCKING)
`SHARED_STAGE1_PREFIX = 12` / `STAGE1_OBS_DIM = 14` reflect the previous
physics observation. The module is retained per plan but unused by the new
orchestrator. Logged in `BUGS.md`.

### Issue 6: Per-model mesh scale + orientation (RESOLVED)
The real `models/*.obj` files use different raw units and axis conventions, so a
single `scale` is insufficient. Each model now gets its own scale
(`interceptor_scale: 0.1612`, `target_scale: 0.5114`) and a proper cube-rotation
code (`interceptor_rotation: I08`, `target_rotation: T21`). Helpers in
`visualiser/scene_builder.py`; workflow in `CURRENT_WORK.md`. Logged in `BUGS.md`.

## 6. Interpreting config access (gotcha)

`cfg.apn` is an `APNSpec` (`N_prime`, `t_go_min`, `t_go_max`), while a
`StageConfig` exposes the APN values as flat fields `stage_cfg.apn_N_prime`
and `stage_cfg.apn_t_go_min` (there is **no** `stage_cfg.apn`).

## 7. How to run

```bash
pip install -r requirements.txt

python scripts/smoke_test.py                                   # 11/11 quick checks
python -m pytest tests/ -q -m "not slow"                       # 52 unit tests
python -m pytest tests/ -q                                     # + end-to-end

python scripts/train.py --stages 1 --name smoke --steps 20000  # smoke train
python scripts/train.py --stages 1,2,3 --name apn_rl_v1        # full curriculum

python -m visualiser --model apn --stage all --output-dir data/videos
python scripts/evaluate.py --model data/runs/apn_rl_v1/ppo_interceptor/final.zip \
  --stage 3 --episodes 100 --render
```

## 8. Verification status

* `scripts/smoke_test.py` → 11/11 pass.
* `pytest -m "not slow"` → 52 pass, 1 deselected; full suite → 53 pass.
* Pure-APN (zero RL action) rollout: stage 1 kills 12/12, stage 2 kills 12/12,
  stage 3 kills 0/12 — confirming APN suffices for predictable targets and the RL
  residual is required for evasive ones.
