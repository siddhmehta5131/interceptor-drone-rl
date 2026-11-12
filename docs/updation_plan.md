# Implementation Plan: APN + RL Path-Planning Interceptor

> **Project shift**: From physics-based quadrotor control (motor commands, PID, RK4 rigid-body sim) → **pure 3D path planning** using Augmented Proportional Navigation (APN) with RL fine-tuning for evasive target compensation. Output is velocity commands (vx, vy, vz, ωx, ωy, ωz), not motor RPMs.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [What is Augmented Proportional Navigation (APN)](#2-what-is-augmented-proportional-navigation-apn)
3. [Architecture Overview: APN + RL Hybrid](#3-architecture-overview-apn--rl-hybrid)
4. [Phase 0 — Git Backup of Current Codebase](#4-phase-0--git-backup-of-current-codebase)
5. [Phase 1 — Clean Slate: Remove Old Code, Retain Reusable Parts](#5-phase-1--clean-slate-remove-old-code-retain-reusable-parts)
6. [Phase 2 — New Project Structure](#6-phase-2--new-project-structure)
7. [Phase 3 — Guidance Core: APN Engine](#7-phase-3--guidance-core-apn-engine)
8. [Phase 4 — Gymnasium Environment](#8-phase-4--gymnasium-environment)
9. [Phase 5 — Observation & Action Spaces](#9-phase-5--observation--action-spaces)
10. [Phase 6 — Reward Design](#10-phase-6--reward-design)
11. [Phase 7 — Optimal Trajectory Oracle (Critic Privileged Data)](#11-phase-7--optimal-trajectory-oracle-critic-privileged-data)
12. [Phase 8 — RL Training Pipeline](#12-phase-8--rl-training-pipeline)
13. [Phase 9 — 3-Stage Curriculum](#13-phase-9--3-stage-curriculum)
14. [Phase 10 — 3D Visualiser with Drone Model](#14-phase-10--3d-visualiser-with-drone-model)
15. [Phase 11 — Evaluation & Metrics](#15-phase-11--evaluation--metrics)
16. [Phase 12 — Testing & Validation](#16-phase-12--testing--validation)
17. [File-by-File Implementation Specification](#17-file-by-file-implementation-specification)
18. [Dependency List](#18-dependency-list)
19. [Implementation Timeline](#19-implementation-timeline)

---

## 1. Executive Summary

The project pivots from a Crazyflie 2.0 rigid-body physics simulator (RK4, PID rate control, ESC modelling, motor dynamics) to a **pure kinematic path-planning problem**. The interceptor is now an abstract agent (could be a drone, missile, or any vehicle) that must navigate in 3D space to intercept a target.

### Key Changes

| Aspect | Old (Physics-Based) | New (Path-Planning) |
|---|---|---|
| **Physics** | Full rigid-body dynamics, RK4 integration, motor model | Kinematic point-mass with velocity limits |
| **Output** | 4 motor commands `[-1, 1]` → RPM via ESC curve | 6-DOF velocity: `(vx, vy, vz, ωx, ωy, ωz)` |
| **Input (obs)** | rot6d, body velocities, ω, alt_err, target rel | `n, m, p, q` history/future config + relative target state |
| **Guidance** | None (RL learns from scratch) | APN baseline + RL residual correction |
| **Stages** | 8 stages (hover → evasive) | 3 stages (static → dynamic → evasive) |
| **Visualisation** | Matplotlib 3D arrows | PyVista with .OBJ drone model, MP4 video |
| **Critic Info** | time_remaining, facing_error, true_pos, true_vel | Optimal trajectory from analytic APN solution |

### Core Insight: Why APN + RL?

**Augmented Proportional Navigation** is the mathematically optimal intercept guidance law for constant-acceleration targets. It works perfectly against non-maneuvering or predictably-maneuvering targets. However, against **evasive targets** that change their acceleration unpredictably, APN suffers because:
- It assumes constant target acceleration over the remaining engagement
- It cannot anticipate future maneuvers
- It has no memory of past evasion patterns

**RL fills this gap** by learning a **residual correction** on top of APN's command. The RL policy sees the history of target behavior and learns to:
- Pre-compensate for anticipated evasive maneuvers
- Adapt the effective navigation constant N dynamically
- Add a bias acceleration in the direction the target is likely to evade

The hybrid command is:

```
a_cmd = APN_command + RL_residual_correction
```

where `APN_command = N' * Vc * ω_LOS + (N'/2) * a_T_perp` and the RL agent outputs a small correction vector.

---

## 2. What is Augmented Proportional Navigation (APN)

### 2.1 Classical Proportional Navigation (PN)

In a 2D engagement, PN commands lateral acceleration proportional to the Line-of-Sight (LOS) rotation rate:

```
a_M = N * Vc * λ̇
```

where:
- `a_M` = missile (interceptor) commanded lateral acceleration
- `N` = navigation constant (effective, typically 3–5)
- `Vc` = closing velocity (`-Ṙ`, rate of range decrease)
- `λ̇` = LOS angular rate

This drives the LOS rate to zero → collision course if the target flies straight.

### 2.2 Augmented PN (APN) — Adding Target Maneuver Compensation

APN extends PN by adding a term that compensates for measured/estimated target acceleration:

```
a_M = N' * Vc * λ̇  +  (N'/2) * a_T
```

- **First term**: Classical PN — drives LOS rate to zero
- **Second term**: Augmentation — pre-compensates for target maneuver `a_T`

### 2.3 Three-Dimensional Vector Form

In full 3D, the APN guidance law uses vector cross products:

```
a⃗_M = N' * (V⃗_c × ω⃗_LOS) + (N'/2) * a⃗_T_perp
```

where:
- `V⃗_c = V⃗_T - V⃗_M` = relative velocity vector (target minus interceptor)
- `ω⃗_LOS = (R⃗ × V⃗_rel) / |R⃗|²` = LOS angular velocity vector
- `a⃗_T_perp` = component of target acceleration **perpendicular** to the LOS
- `R⃗ = p⃗_T - p⃗_M` = relative position vector (LOS vector)

### 2.4 Zero Effort Miss (ZEM)

APN is derived from minimizing the **Zero Effort Miss** — the predicted miss distance if no further control is applied:

```
ZEM = Vc * t_go² * λ̇ + ½ * a_T * t_go²
```

The optimal command that drives ZEM to zero is:

```
a_M = N' * ZEM / t_go²
```

Substituting the ZEM expression recovers the APN law.

### 2.5 Time-to-Go Estimation

```
t_go = R / Vc
```

where `R = |R⃗|` is the current range and `Vc = -Ṙ` is the closing velocity (positive when closing).

### 2.6 Key Properties

| Property | Value |
|---|---|
| Navigation constant N' | Typically 3–5 (3 = optimal energy, 5 = faster convergence) |
| Optimality | Energy-optimal for constant-acceleration targets |
| Assumption | Target acceleration is constant over remaining engagement |
| Weakness | Cannot handle unpredictable maneuvers or acceleration changes |
| Singularity | `t_go → 0` causes divergence (clamp required) |

---

## 3. Architecture Overview: APN + RL Hybrid

```
┌──────────────────────────────────────────────────────────────────────┐
│                        INTERCEPTOR AGENT                            │
│                                                                      │
│  ┌─────────────────┐     ┌──────────────────────┐                   │
│  │   APN Engine     │     │   RL Policy (PPO)     │                  │
│  │                  │     │                       │                   │
│  │  IN: R⃗, V⃗_rel, │     │  IN: obs vector       │                  │
│  │      ω_LOS,      │     │      (history, future, │                 │
│  │      a_T, t_go   │     │       relative state)  │                 │
│  │                  │     │                       │                   │
│  │  OUT: a⃗_APN     │     │  OUT: Δa⃗_RL          │                  │
│  │  (3D accel cmd)  │     │  (3D residual corr.)  │                  │
│  └────────┬─────────┘     └──────────┬────────────┘                  │
│           │                          │                                │
│           └──────────┬───────────────┘                                │
│                      ▼                                                │
│           ┌──────────────────┐                                        │
│           │  a⃗_cmd = a⃗_APN  │                                       │
│           │        + Δa⃗_RL  │                                        │
│           └────────┬─────────┘                                        │
│                    ▼                                                  │
│           ┌──────────────────┐                                        │
│           │ Velocity Update   │                                       │
│           │ v⃗_new = v⃗ + a⃗*dt│                                       │
│           │ (clamped to v_max)│                                       │
│           │                  │                                        │
│           │ OUT: (vx,vy,vz,  │                                       │
│           │       ωx,ωy,ωz)  │                                       │
│           └──────────────────┘                                        │
└──────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
              ┌────────────────────┐
              │  Kinematic Sim     │
              │  p⃗ += v⃗ * dt      │
              │  θ⃗ += ω⃗ * dt      │
              │  (point-mass, no   │
              │   rigid body)      │
              └────────────────────┘
```

### How the RL Residual Works

1. **Stage 1 (Static Target)**: APN alone is sufficient. The RL agent learns that `Δa⃗_RL ≈ 0` is optimal. This stage trains the RL agent to *not interfere* with a working guidance law.

2. **Stage 2 (Dynamic Target)**: APN handles the predictable acceleration component. RL begins to learn small corrections for trajectory curvature and closing geometry. The RL correction is still small.

3. **Stage 3 (Evasive Target)**: The target actively changes its acceleration profile to evade. APN reacts with one-step-lag to the measured acceleration. **RL learns to anticipate** evasive patterns from the observation history and adds predictive corrections. This is where RL provides the most value — closing the gap between APN's "current-state-only" guidance and a truly anticipatory policy.

### RL Role Per Stage

| Stage | APN Contribution | RL Contribution | RL Action Scale |
|---|---|---|---|
| Static | 100% — drives straight to target | ~0% — learns to stay silent | `scale = 0.1 * a_max` |
| Dynamic | ~85% — tracks predictable path | ~15% — smooths trajectory | `scale = 0.3 * a_max` |
| Evasive | ~50% — tracks current accel | ~50% — predicts evasion | `scale = 0.5 * a_max` |

---

## 4. Phase 0 — Git Backup of Current Codebase

### 4.1 Steps

```
Step 0.1:  Create the target directory:
           Desktop/projects/github_repos/rl-drone-flight-simulator/

Step 0.2:  Copy the ENTIRE current working tree (minus .git, __pycache__,
           *.pyc, .env, data/, *.zip model archives) to the target directory.
           Use robocopy with exclusion filters.

Step 0.3:  cd into Desktop/projects/github_repos/rl-drone-flight-simulator/
           Initialize a fresh git repo:
             git init
             git add .
             git commit -m "Archive: physics-based interceptor drone v6.0 (pre-pivot)"

Step 0.4:  Create a new GitHub repo via the gh CLI or manually:
             gh repo create siddhmehta5131/rl-drone-flight-simulator --public --source=. --push
           OR manually:
             git remote add origin https://github.com/siddhmehta5131/rl-drone-flight-simulator.git
             git branch -M main
             git push -u origin main

Step 0.5:  Verify the push succeeded:
             git log --oneline -3
             # Should show the archive commit on GitHub
```

### 4.2 .gitignore for the Archive Repo

```gitignore
__pycache__/
*.pyc
*.pyo
.env
data/
*.zip
*.tar.gz
models and meta data stage 1-8.zip
final_videos/
*.mp4
*.avi
.idea/
.vscode/
```

### 4.3 What Gets Archived

Everything currently in `RL based autonomous interceptor drone/`:
- All Python source files (src/, interceptor-training/, stage-runtime/)
- All config files (YAML, JSON)
- All documentation (*.md, *.docx)
- All notebooks (*.ipynb)
- All probe/test scripts (*_probe.py, validate*.py)
- The visualiser code
- Docker files
- PNG/image assets

### 4.4 What Does NOT Get Archived

- `.git/` (we create a fresh git history)
- `__pycache__/` directories
- `models and meta data stage 1-8.zip` (30 MB binary — too large, add to .gitignore)
- `final_videos/` directory (MP4 outputs)
- Any `.env` files

---

## 5. Phase 1 — Clean Slate: Remove Old Code, Retain Reusable Parts

### 5.1 Files/Directories to DELETE from `RL based autonomous interceptor drone/`

After the backup is safely pushed to GitHub, remove the following from the **working project directory**:

#### Entire Directories to Delete
- `interceptor-training/src/physics/` — rigid-body dynamics, RK4, ESC, PID, aero
- `interceptor-training/src/prediction/` — const_vel, linear_ridge (we rebuild prediction)
- `stage-runtime/` — old visualiser runner (replaced by PyVista visualiser)
- `__pycache__/` (all)
- `final_videos/` — old output videos

#### Root-Level Files to Delete
- All probe scripts: `_axis_probe.py`, `_dir_probe.py`, `_guid_probe.py`, `_hover_probe.py`, `_hover_trace.py`, `_man_probe.py`, `_reset_probe.py`, `_sign_probe.py`, `_yaw_probe.py`
- All comparison scripts: `compare_pybullet.py`, `compare_v3_multirate.py`, `compare_v3_vs_pybullet.py`
- All old physics test scripts: `swift_physics_headless*.py`, `swift_demo_wind_only.py`, `swift_rl_env*.py`
- All old demo scripts: `demo.py`, `default_demo.py`, `updated demo.py`, `maneuver_video.py`
- All validation scripts: `validate*.py`, `self_test.py`
- All old visualisation images: `*.png` (trajectories.png, inputs.png, etc.)
- All notebooks: `*.ipynb` (Coeff solver, SB3_Altitude, phy_sim_val, pybullet_vs_swift, etc.)
- Old implementation plans: `implementation_plan.md`, `implementation_plan_d2_d58.md`, `plan_2026-10-02.md`
- Old context files: `context_v0.0.md`, `context_v5_0.md`, `complete_changelog.md`
- Docker files: `Docker_Setup_and_Running_Guide.md`, `Docker_Setup_and_Running_Guide.pdf`
- Old documents: `BTP plan preliminary.docx`, `Interceptor_Complete_Project_Scope.docx`
- Misc: `esc_ctrl_test.py`, `esc_fix_test.py`, `esc_lin_test.py`, `identify_params.py`, `diag_local.py`
- Results files: `results_trajectory.txt`, `results_wind_rlvec.txt`, `run_summary.json`
- Old interceptor log: `interceptor.log`
- Model archive: `models and meta data stage 1-8.zip`
- `hover_env.py` — Stage-1 parity reference, no longer needed
- `role.txt`, `references.txt`

#### From interceptor-training/ to Delete
- `Dockerfile`, `docker-compose.yml` — no Docker in new setup
- `interceptor-training/scripts/` — old CLI entry points
- `interceptor-training/src/physics/` — **entire directory** (pipeline.py, constants.py, aero.py, quaternion.py)
- `interceptor-training/src/prediction/` — **entire directory**
- `interceptor-training/data/` — old training data

### 5.2 Files/Modules to RETAIN and ADAPT

These files contain patterns and logic that transfer to the new system with moderate changes:

| File | What to Keep | What Changes |
|---|---|---|
| `interceptor-training/src/envs/target_generator.py` | `PolynomialTargetPath` class (stages 5-7 path generation), evasive target logic | Remove drone coupling in `_resample_maneuver`, adapt spawn geometry. This is **highly reusable** — it already generates static, polynomial, and evasive targets. |
| `interceptor-training/src/envs/obs_builder.py` | History stacking pattern (`ObsBuilder` class), future block assembly | Rewrite frame dimensions. Remove rot6d/omega_B body states. Replace with relative kinematics (rel_pos, rel_vel, LOS_rate, closing_vel). |
| `interceptor-training/src/envs/stage_config.py` | `StageConfig`/`RewardConfig`/`SpawnConfig` dataclass pattern | Rewrite for 3 stages, remove physics constants, add APN parameters. |
| `interceptor-training/src/envs/reward.py` | `RewardComputer` class structure | Completely rewrite reward terms for path planning. |
| `interceptor-training/src/training/asymmetric_policy.py` | `AsymmetricActorCriticPolicy`, `_SplitExtractor` | Keep exactly as-is — the asymmetric actor-critic with privileged critic is the same architecture. |
| `interceptor-training/src/training/orchestrator.py` | Training loop pattern, SB3 integration | Simplify — 3 stages instead of 8, remove physics coupling. |
| `interceptor-training/src/training/callbacks.py` | TensorBoard logging, curriculum metrics | Adapt metric names for new reward terms. |
| `interceptor-training/src/training/curriculum.py` | Stage advancement logic | Simplify for 3 stages. |
| `interceptor-training/src/training/checkpoint_manager.py` | Checkpoint save/load | Keep as-is. |
| `interceptor-training/src/training/weight_transfer.py` | Network weight transfer between stages | Keep as-is. |
| `interceptor-training/src/results.py` | `TrainResult`/`ModelResult` types | Minor rename of stage count. |
| `interceptor-training/src/api.py` | `train_model()` public API | Simplify. |
| `interceptor-training/src/utils/` | Config/model loaders, logger | Keep as-is. |
| `interceptor-training/configs/model.yaml` | Model definition YAML pattern | Update `privileged_critic` fields for new oracle data. |
| `interceptor-training/configs/config.yaml` | Training config YAML pattern | Rewrite for new observation layout, remove physics params. |
| `interceptor-training/run.py` | Entry point pattern | Simplify for 3 stages. |
| `interceptor-training/requirements.txt` | Dependency list | Update for pyvista, trimesh. |
| `src/visualiser.py` | **Delete and replace** | Complete rewrite using PyVista. |
| `.gitignore` | Git ignore patterns | Update for new project. |
| `README.md` | Project README | Complete rewrite. |
| `context.md` | Context tracking | Complete rewrite. |
| `BUGS.md`, `CURRENT_WORK.md`, `VERSIONS.md` | Tracking docs | Archive / rewrite. |

---

## 6. Phase 2 — New Project Structure

```
RL based autonomous interceptor drone/
├── .git/
├── .gitignore
├── README.md
├── context.md                          ← updated project context
├── requirements.txt                    ← project-wide dependencies
│
├── configs/
│   ├── model.yaml                      ← RL model architecture + privileged critic fields
│   ├── config.yaml                     ← training settings, obs layout, APN params
│   ├── stages.yaml                     ← 3-stage curriculum definitions
│   └── visualiser.yaml                 ← visualiser settings (model path, camera, resolution)
│
├── models/                             ← downloaded 3D models
│   ├── interceptor_drone.obj           ← quadcopter OBJ from GrabCAD/Free3D
│   ├── target_drone.obj                ← target OBJ (simpler/different colour)
│   └── README.md                       ← model attribution and download links
│
├── src/
│   ├── __init__.py
│   │
│   ├── guidance/                       ← APN engine + optimal trajectory solver
│   │   ├── __init__.py
│   │   ├── apn.py                      ← 3D Augmented Proportional Navigation
│   │   ├── optimal_trajectory.py       ← analytic optimal intercept path (critic oracle)
│   │   └── kinematic_model.py          ← point-mass kinematics (v, ω integration)
│   │
│   ├── envs/                           ← Gymnasium environments
│   │   ├── __init__.py
│   │   ├── intercept_env.py            ← main 3-stage Gym environment
│   │   ├── obs_builder.py              ← observation stacking (history + future)
│   │   ├── reward.py                   ← per-stage reward computation
│   │   ├── stage_config.py             ← 3-stage curriculum table
│   │   └── target_generator.py         ← static / dynamic / evasive target paths
│   │
│   ├── training/                       ← RL training pipeline
│   │   ├── __init__.py
│   │   ├── asymmetric_policy.py        ← actor-critic with privileged critic (RETAINED)
│   │   ├── orchestrator.py             ← training loop, SB3 integration
│   │   ├── callbacks.py                ← TensorBoard logging
│   │   ├── curriculum.py               ← stage advancement logic
│   │   ├── checkpoint_manager.py       ← save/load checkpoints
│   │   └── weight_transfer.py          ← transfer weights between stages
│   │
│   ├── prediction/                     ← target state prediction
│   │   ├── __init__.py
│   │   ├── base.py                     ← predictor interface
│   │   ├── const_vel.py                ← constant-velocity predictor
│   │   └── kalman.py                   ← Kalman filter predictor (new, better for evasive)
│   │
│   ├── utils/                          ← config loading, logging
│   │   ├── __init__.py
│   │   ├── config_loader.py
│   │   ├── model_loader.py
│   │   └── logger.py
│   │
│   ├── api.py                          ← train_model() public function
│   └── results.py                      ← TrainResult / ModelResult types
│
├── visualiser/                         ← 3D visualisation suite
│   ├── __init__.py
│   ├── renderer.py                     ← PyVista-based 3D scene renderer
│   ├── animator.py                     ← frame-by-frame animation + MP4 export
│   ├── scene_builder.py                ← load models, set up lighting, camera
│   └── cli.py                          ← CLI: python -m visualiser --model drone --stage 2
│
├── scripts/
│   ├── train.py                        ← training entry point
│   ├── evaluate.py                     ← evaluation + metrics
│   ├── visualise.py                    ← generate demonstration videos
│   ├── smoke_test.py                   ← quick validation suite
│   └── download_models.py              ← download 3D models from online sources
│
├── data/                               ← training outputs (gitignored)
│   ├── runs/                           ← TensorBoard logs, checkpoints
│   └── videos/                         ← generated MP4 videos
│
└── run.py                              ← one-file user script
```

---

## 7. Phase 3 — Guidance Core: APN Engine

### 7.1 `src/guidance/apn.py` — The APN Guidance Law

This is the mathematical heart of the project. It implements the 3D vector form of Augmented Proportional Navigation.

```python
"""
apn.py — 3D Augmented Proportional Navigation (APN) Guidance Law.

The APN command is:
    a⃗_M = N' * (V⃗_c × ω⃗_LOS) + (N'/2) * a⃗_T_perp

In this codebase:
    - Interceptor = the agent we control
    - Target = the entity we want to intercept
    - All vectors are in the world (inertial) frame
    - LOS = Line of Sight = vector from interceptor to target
"""
```

#### Core Function Signatures

```python
def compute_apn_command(
    p_interceptor: np.ndarray,   # (3,) interceptor position
    v_interceptor: np.ndarray,   # (3,) interceptor velocity
    p_target: np.ndarray,        # (3,) target position
    v_target: np.ndarray,        # (3,) target velocity
    a_target: np.ndarray,        # (3,) target acceleration (estimated or measured)
    N_prime: float = 4.0,        # navigation constant
    t_go_min: float = 0.1,       # minimum t_go clamp (avoid singularity)
) -> Tuple[np.ndarray, dict]:
    """
    Compute the 3D APN acceleration command.
    
    Returns:
        a_cmd: (3,) acceleration command vector in world frame
        info: dict with LOS_rate, closing_vel, t_go, ZEM, range
    """
```

#### Detailed Implementation Logic

```python
# 1. Relative geometry
R_vec = p_target - p_interceptor           # LOS vector (interceptor → target)
R = np.linalg.norm(R_vec)                  # range (scalar)
R_hat = R_vec / max(R, 1e-9)              # LOS unit vector

# 2. Relative velocity
V_rel = v_target - v_interceptor           # relative velocity
V_c = -np.dot(V_rel, R_hat)               # closing velocity (positive when closing)

# 3. Time-to-go estimate
t_go = max(R / max(V_c, 0.1), t_go_min)   # clamp to avoid singularity

# 4. LOS angular velocity vector
#    ω_LOS = (R⃗ × V⃗_rel) / |R⃗|²
omega_LOS = np.cross(R_vec, V_rel) / max(R**2, 1e-9)

# 5. Target acceleration perpendicular to LOS
a_T_along_LOS = np.dot(a_target, R_hat) * R_hat
a_T_perp = a_target - a_T_along_LOS

# 6. APN command
#    a⃗_M = N' * (V⃗_rel × ω⃗_LOS) + (N'/2) * a⃗_T_perp
#    Note: V_rel × ω_LOS gives acceleration perpendicular to LOS
#    Equivalent form: N' * Vc * ω_LOS_magnitude * direction
a_pn = N_prime * np.cross(V_rel, omega_LOS)   # classical PN term
a_aug = (N_prime / 2.0) * a_T_perp             # augmentation term
a_cmd = a_pn + a_aug

# 7. Zero Effort Miss (for diagnostics)
ZEM_vec = V_rel * t_go + 0.5 * a_target * t_go**2
# Perpendicular component of ZEM
ZEM_along = np.dot(ZEM_vec, R_hat) * R_hat
ZEM_perp = ZEM_vec - ZEM_along
ZEM = np.linalg.norm(ZEM_perp)
```

### 7.2 `src/guidance/kinematic_model.py` — Point-Mass Kinematics

```python
"""
kinematic_model.py — Simple kinematic interceptor model.

The interceptor is a point mass with:
- Position p⃗ (3D)
- Velocity v⃗ (3D) — clamped to [0, v_max]
- Orientation θ⃗ (3D Euler angles or quaternion) — for visualisation only
- Angular velocity ω⃗ (3D) — for heading changes during intercept

No aerodynamics, no motor model, no rigid-body dynamics.
The velocity is updated by the acceleration command from APN + RL.
"""

@dataclass
class InterceptorState:
    position: np.ndarray       # (3,) world frame [x, y, z]
    velocity: np.ndarray       # (3,) world frame [vx, vy, vz]
    orientation: np.ndarray    # (4,) quaternion [w, x, y, z] (for vis)
    angular_velocity: np.ndarray  # (3,) [ωx, ωy, ωz]

class KinematicModel:
    def __init__(self, v_max: float, a_max: float, omega_max: float, dt: float):
        self.v_max = v_max      # maximum speed (m/s)
        self.a_max = a_max      # maximum acceleration (m/s²)
        self.omega_max = omega_max  # maximum angular rate (rad/s)
        self.dt = dt
    
    def step(self, state: InterceptorState, a_cmd: np.ndarray, omega_cmd: np.ndarray) -> InterceptorState:
        """Advance one timestep.
        
        1. Clamp acceleration to a_max
        2. Update velocity: v += a * dt, clamp to v_max
        3. Update position: p += v * dt
        4. Clamp angular velocity to omega_max
        5. Update orientation quaternion
        """
```

### 7.3 `src/guidance/optimal_trajectory.py` — Critic Oracle

This module computes the **mathematically optimal intercept trajectory** given complete knowledge of the target's future path. This is used as **privileged critic information** during training.

```python
"""
optimal_trajectory.py — Compute the optimal intercept trajectory.

Given the interceptor's current state and the COMPLETE target path
(all future positions and velocities), compute:
1. The optimal time-to-intercept
2. The optimal velocity profile
3. The optimal acceleration sequence

This is fed to the critic as ground-truth "what the perfect policy would do"
so the value function learns faster.

Method: For a known target path p_T(t), the optimal interceptor path
minimises ∫|a(t)|² dt subject to p_M(t_f) = p_T(t_f).
This has a closed-form solution for polynomial target paths.
"""

def compute_optimal_intercept(
    p_interceptor: np.ndarray,
    v_interceptor: np.ndarray,
    target_path: PolynomialTargetPath,  # or callable(t) -> pos, vel
    v_max: float,
    a_max: float,
    dt: float,
) -> dict:
    """
    Returns:
        optimal_time: float — minimum time to intercept
        optimal_positions: (T, 3) — position trajectory
        optimal_velocities: (T, 3) — velocity trajectory
        optimal_accelerations: (T, 3) — acceleration trajectory
        optimal_cost: float — total ∫|a|² dt
    """
```

For a static target, this reduces to a straight-line constant-acceleration / constant-deceleration profile (bang-coast-bang or direct approach). For polynomial targets, we solve a boundary-value problem using the target's known path.

---

## 8. Phase 4 — Gymnasium Environment

### 8.1 `src/envs/intercept_env.py` — Main Environment

```python
"""
intercept_env.py — Gymnasium environment for 3D interception.

Three stages:
  Stage 1 (Static):   Fixed target position, no movement
  Stage 2 (Dynamic):  Target follows polynomial path (order 1-3)
  Stage 3 (Evasive):  Target actively evades with reactive maneuvers

The agent controls the interceptor's acceleration (which gets summed
with APN's baseline command). The environment outputs 6-DOF velocity
commands as the "action taken" (for logging/vis).

Action space: Box([-1, 1], shape=(3,))
  → scaled to [-a_max_rl, +a_max_rl] as residual acceleration correction
  → added to APN command
  → total acceleration integrated to velocity
  → velocity clamped to v_max

Observation: structured flat vector with history + future stacking
"""
```

#### Environment Step Logic

```python
def step(self, action: np.ndarray):
    # 1. Decode RL action → residual acceleration correction
    delta_a_rl = action * self.rl_action_scale  # scale from [-1,1] to m/s²
    
    # 2. Advance target
    self.target.step(self.dt, self.interceptor.position)
    
    # 3. Compute APN baseline command
    a_apn, apn_info = compute_apn_command(
        p_interceptor=self.interceptor.position,
        v_interceptor=self.interceptor.velocity,
        p_target=self.target.pos,
        v_target=self.target.vel,
        a_target=self.target.acc,
        N_prime=self.N_prime,
    )
    
    # 4. Combine: total command = APN + RL residual
    a_total = a_apn + delta_a_rl
    
    # 5. Clamp total acceleration
    a_norm = np.linalg.norm(a_total)
    if a_norm > self.a_max:
        a_total = a_total * (self.a_max / a_norm)
    
    # 6. Compute angular velocity to align heading with velocity direction
    omega_cmd = self._compute_heading_rate(a_total)
    
    # 7. Integrate kinematics
    self.interceptor = self.kinematic_model.step(
        self.interceptor, a_total, omega_cmd
    )
    
    # 8. Record the OUTPUT velocity commands (vx, vy, vz, ωx, ωy, ωz)
    velocity_output = np.concatenate([
        self.interceptor.velocity,
        self.interceptor.angular_velocity
    ])
    
    # 9. Check termination
    distance = np.linalg.norm(self.target.pos - self.interceptor.position)
    killed = distance < self.kill_radius
    out_of_bounds = self._check_bounds()
    timeout = self.step_count >= self.max_steps
    
    # 10. Compute reward
    reward, terminated, components, metrics = self.reward_computer.compute(...)
    
    # 11. Build observation
    obs = self._build_observation()
    
    # 12. Store trajectory for visualisation
    self._trajectory_log.append({
        'interceptor_pos': self.interceptor.position.copy(),
        'interceptor_vel': self.interceptor.velocity.copy(),
        'interceptor_quat': self.interceptor.orientation.copy(),
        'target_pos': self.target.pos.copy(),
        'target_vel': self.target.vel.copy(),
        'a_apn': a_apn.copy(),
        'a_rl': delta_a_rl.copy(),
        'a_total': a_total.copy(),
        'velocity_output': velocity_output.copy(),
        'distance': distance,
        'LOS_rate': apn_info['LOS_rate'],
        'closing_vel': apn_info['closing_vel'],
        't_go': apn_info['t_go'],
    })
    
    truncated = timeout and not terminated
    return obs, reward, terminated, truncated, info
```

---

## 9. Phase 5 — Observation & Action Spaces

### 9.1 Observation Vector Layout

The observation is domain-agnostic (works for any interceptor, not just drones). It consists of **relative kinematics** between interceptor and target.

#### Single Frame (current timestep) — 16 dimensions

| Index | Name | Dim | Description |
|---|---|---|---|
| 0-2 | `rel_pos` | 3 | Relative position (target - interceptor) in world frame |
| 3-5 | `rel_vel` | 3 | Relative velocity (target - interceptor) |
| 6-8 | `interceptor_vel` | 3 | Interceptor velocity in world frame |
| 9 | `range` | 1 | Euclidean distance to target |
| 10 | `closing_vel` | 1 | Rate of range decrease (positive = closing) |
| 11-13 | `LOS_rate` | 3 | Line-of-sight angular velocity vector |
| 14 | `t_go_est` | 1 | Estimated time-to-go (R / Vc) |
| 15 | `visible` | 1 | Target visibility flag (always 1 in this setup) |

**Total single frame: 16 dimensions**

#### History Stacking

With `m` history frames and `p` skip:
- History block = `16 * (1 + m)` dimensions (current + m past frames)

#### Future Block (per sample) — 7 dimensions

| Index | Name | Dim | Description |
|---|---|---|---|
| 0-2 | `future_rel_pos` | 3 | Predicted relative position at future timestep |
| 3-5 | `future_rel_vel` | 3 | Predicted relative velocity at future timestep |
| 6 | `visible` | 1 | Visibility flag |

With `n` future samples and `q` skip:
- Future block = `7 * n` dimensions

#### Privileged Critic Fields

| Name | Dim | Description |
|---|---|---|
| `time_remaining` | 1 | Normalised time left in episode |
| `optimal_accel` | 3 | Acceleration from optimal trajectory oracle |
| `optimal_vel` | 3 | Velocity from optimal trajectory oracle |
| `target_true_future_pos` | 3 | Ground-truth target position 1s ahead |
| `target_true_future_vel` | 3 | Ground-truth target velocity 1s ahead |
| `target_true_accel` | 3 | Ground-truth target acceleration (exact) |

**Privileged block: 16 dimensions**

#### Total Observation Dimension

With default `m=3, p=2, n=3, q=5`:

```
Actor obs = 16 * (1 + 3) + 7 * 3 = 64 + 21 = 85 dimensions
Critic obs = 85 + 16 = 101 dimensions
```

### 9.2 Action Space

```python
action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
```

The 3-dimensional action represents the **RL residual acceleration correction** in world frame (x, y, z). It is scaled by a per-stage `rl_action_scale` before being added to the APN command.

### 9.3 Output (Not an Action Space — Logged for Downstream)

The environment **logs** the resulting 6-DOF velocity command after integration:

```python
velocity_output = [vx, vy, vz, ωx, ωy, ωz]  # 6 dimensions
```

This is what would be sent to a real vehicle's velocity controller. It is stored in `info['velocity_output']` and in the trajectory log for visualisation.

### 9.4 Configuration in `config.yaml`

```yaml
observation:
  frame_dim: 16           # per-frame observation width
  history_frames: 3       # m — number of past frames
  history_skip: 2         # p — spacing between past frames
  future_samples: 3       # n — number of future predictions
  future_skip: 5          # q — spacing between future samples (in env steps)
  future_source: pred     # 'true' (oracle) or 'pred' (predictor)
  predictor: const_vel    # const_vel | kalman

interceptor:
  v_max: 50.0             # m/s — max interceptor speed
  a_max: 15.0             # m/s² — max total acceleration
  omega_max: 3.0          # rad/s — max angular rate
  speed_ratio: 1.5        # interceptor_speed / target_speed (>1 always)

apn:
  N_prime: 4.0            # navigation constant
  t_go_min: 0.05          # minimum time-to-go clamp (s)
```

---

## 10. Phase 6 — Reward Design

### 10.1 Overview

The reward is designed to:
1. Guide the interceptor toward the target (progress)
2. Encourage efficient trajectories (energy minimisation)
3. Penalise deviation from APN baseline when unnecessary (regularisation)
4. Reward fast interceptions (time bonus)

### 10.2 Reward Components

```python
@dataclass
class RewardConfig:
    # --- Dense per-step terms ---
    k_progress: float = 1.0          # reward for closing distance
    k_progress_normalize: bool = True # divide by max(distance, 1.0)
    k_LOS_rate: float = 0.1          # penalise high LOS rate (want λ̇ → 0)
    k_energy: float = 0.02           # penalise |a_total|² (energy cost)
    k_rl_residual: float = 0.05      # penalise |Δa_RL|² (prefer APN alone)
    k_smooth: float = 0.03           # penalise |a(t) - a(t-1)|² (jerk)
    k_time_penalty: float = 0.001    # flat per-step time cost
    k_closing_vel: float = 0.1       # reward for positive closing velocity
    
    # --- Terminal bonuses ---
    k_kill_bonus: float = 100.0      # intercept reward
    k_miss_distance_scale: bool = True   # scale kill bonus by miss accuracy
    k_time_bonus_scale: bool = True      # scale kill bonus by speed of kill
    k_crash_penalty: float = -50.0       # out of bounds penalty
    k_ZEM: float = 0.05              # reward for low Zero Effort Miss
```

### 10.3 Reward Function

```python
def compute(self, *, distance, prev_distance, closing_vel, LOS_rate_norm,
            a_total_norm, a_rl_norm, action, prev_action, ZEM,
            step_count, max_steps, killed, out_of_bounds):
    
    comp = {}
    
    # Progress: reward for decreasing distance
    progress = prev_distance - distance
    if self.cfg.k_progress_normalize:
        progress = progress / max(distance, 1.0)
        progress = np.clip(progress, -1.0, 1.0)
    comp['r_progress'] = self.cfg.k_progress * progress
    
    # LOS rate: penalise high rotation (want collision course)
    comp['r_LOS_rate'] = -self.cfg.k_LOS_rate * LOS_rate_norm
    
    # Closing velocity: reward for approaching
    comp['r_closing_vel'] = self.cfg.k_closing_vel * max(0, closing_vel) / self.v_max
    
    # Energy: penalise large accelerations
    comp['r_energy'] = -self.cfg.k_energy * a_total_norm**2
    
    # RL residual: penalise deviating from APN (regularisation)
    comp['r_rl_residual'] = -self.cfg.k_rl_residual * a_rl_norm**2
    
    # Smoothness: penalise jerk (action changes)
    comp['r_smooth'] = -self.cfg.k_smooth * np.sum((action - prev_action)**2)
    
    # Time penalty: flat cost per step
    comp['r_time'] = -self.cfg.k_time_penalty
    
    # ZEM: reward for low predicted miss
    comp['r_ZEM'] = -self.cfg.k_ZEM * ZEM
    
    # --- Terminal ---
    comp['r_kill'] = 0.0
    terminated = False
    
    if killed:
        bonus = self.cfg.k_kill_bonus
        if self.cfg.k_miss_distance_scale:
            bonus *= max(0, 1 - distance / self.kill_radius)
        if self.cfg.k_time_bonus_scale:
            bonus *= (1 - step_count / max_steps)
        comp['r_kill'] = max(0, bonus)
        terminated = True
    
    if out_of_bounds:
        comp['r_oob'] = self.cfg.k_crash_penalty
        terminated = True
    
    reward = sum(comp.values())
    return reward, terminated, comp
```

---

## 11. Phase 7 — Optimal Trajectory Oracle (Critic Privileged Data)

### 11.1 Purpose

The **critic** receives privileged information that the actor never sees. This is the same asymmetric actor-critic architecture from the original project, but with new privileged fields.

### 11.2 Privileged Fields

```python
PRIVILEGED_FIELDS = {
    "time_remaining": 1,        # normalised time left
    "optimal_accel": 3,         # optimal acceleration from oracle
    "optimal_vel": 3,           # optimal velocity from oracle
    "target_true_future_pos": 3,# ground-truth target position 1s ahead
    "target_true_future_vel": 3,# ground-truth target velocity 1s ahead
    "target_true_accel": 3,     # exact target acceleration
}
```

### 11.3 Optimal Trajectory Computation

For each step, the oracle computes what the **perfect** interceptor would do given full knowledge of the target's future path:

**Static target**: 
- Optimal path = straight line with trapezoidal velocity profile
- `optimal_accel` = constant acceleration toward target, then deceleration

**Dynamic target (polynomial path)**:
- Given `p_T(t)` for all future t, solve for the interceptor path that minimises `∫|a(t)|²dt`
- This is a linear boundary-value problem with closed-form solution
- Use the `PolynomialTargetPath` class (retained from old code) which provides `position_at(t)`, `velocity_at(t)`, `acceleration_at(t)`

**Evasive target**:
- Oracle gets the *actual* future target states (cheating data)
- Computes the intercept using these true future states
- The RL agent can never access this; only the critic sees it

### 11.4 How the Oracle Helps Training

The critic uses the oracle to provide a much better value estimate:
- Without oracle: the critic must learn the value function purely from rewards
- With oracle: the critic knows "the optimal policy would achieve reward X from here" → faster convergence
- The actor still only sees its own observations → no information leakage at inference time

---

## 12. Phase 8 — RL Training Pipeline

### 12.1 Algorithm: PPO with Asymmetric Actor-Critic

We retain the same PPO setup from the original project, with the `AsymmetricActorCriticPolicy`:

```yaml
# configs/model.yaml
ppo_interceptor:
  algo: PPO
  policy: MlpPolicy  # overridden to AsymmetricActorCriticPolicy when privileged_critic is set
  net_arch: {pi: [256, 256, 128], vf: [256, 256, 128]}
  activation: ReLU
  obs_history: {frames: 3, skip: 2}
  hyperparameters:
    n_steps: 4096
    batch_size: 512
    gamma: 0.995
    learning_rate: 3.0e-4
    gae_lambda: 0.95
    clip_range: 0.2
    ent_coef: 0.005
    n_epochs: 10
    max_grad_norm: 0.5
    vf_coef: 0.5
  privileged_critic:
    - time_remaining
    - optimal_accel
    - optimal_vel
    - target_true_future_pos
    - target_true_future_vel
    - target_true_accel
```

### 12.2 Training Loop

```python
from src.api import train_model

# Train through all 3 stages
result = train_model(
    source="configs/model.yaml",
    stages=[1, 2, 3],
    name="apn_rl_v1",
    config="configs/config.yaml",
)

# Save the final model
result.ppo_interceptor.save("data/models/apn_rl_v1_final")
```

### 12.3 Key Training Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `n_steps` | 4096 | Sufficient trajectory length for PPO |
| `batch_size` | 512 | Standard for continuous control |
| `gamma` | 0.995 | Long-horizon discount (intercepts take many steps) |
| `learning_rate` | 3e-4 | Standard PPO LR |
| `ent_coef` | 0.005 | Low entropy for precise control tasks |
| `clip_range` | 0.2 | Standard PPO clip |
| `n_parallel_envs` | 8 | Parallel environments for variance reduction |
| `total_timesteps` per stage | 2M / 5M / 10M | Increasing complexity needs more data |

---

## 13. Phase 9 — 3-Stage Curriculum

### 13.1 Stage 1: Static Target

| Parameter | Value |
|---|---|
| Target type | `static` — fixed position, zero velocity |
| Episode length | 3.0 – 5.0 s |
| Spawn distance | 20 – 60 m |
| Kill radius | 1.0 m |
| RL action scale | `0.1 * a_max` (RL can barely affect APN) |
| APN N' | 4.0 |
| Success metric | Kill rate > 95% |
| Training budget | 2M steps |

**Goal**: Learn that APN alone is sufficient for static targets. The RL agent should learn to output near-zero corrections. This baseline ensures the agent doesn't fight a working guidance law.

### 13.2 Stage 2: Dynamic Target (Polynomial Path)

| Parameter | Value |
|---|---|
| Target type | `order_1`, `order_2`, `order_3` (randomised) |
| Episode length | 5.0 – 10.0 s |
| Spawn distance | 30 – 80 m |
| Kill radius | 1.0 m |
| Target speed range | 5 – 20 m/s |
| Target accel range | 0 – 5 m/s² |
| RL action scale | `0.3 * a_max` |
| APN N' | 4.0 |
| Success metric | Kill rate > 90% |
| Training budget | 5M steps |

**Goal**: Learn small corrections for curved target paths. APN handles the bulk of the guidance; RL smooths the trajectory and compensates for higher-order path curvature.

### 13.3 Stage 3: Evasive Target

| Parameter | Value |
|---|---|
| Target type | `evasive` — reactive jerk with flee bias |
| Episode length | 8.0 – 15.0 s |
| Spawn distance | 40 – 100 m |
| Kill radius | 1.5 m (slightly larger for harder task) |
| Target speed range | 10 – 30 m/s |
| Evasive probability | 0.4 – 0.8 |
| Target jerk limit | 20 m/s³ |
| Target accel g-limit | 3g |
| RL action scale | `0.5 * a_max` (RL has significant authority) |
| APN N' | 4.0 – 5.0 (can be adapted by RL) |
| Success metric | Kill rate > 75% |
| Training budget | 10M steps |

**Goal**: This is where RL earns its keep. The evasive target changes acceleration unpredictably, and APN can only react with one-step lag. The RL agent, using its history stack (3 past frames), learns to anticipate evasion patterns and pre-compensate. The agent effectively learns a "predictive augmentation" on top of APN's reactive augmentation.

### 13.4 Stage Transition Logic

```python
# Curriculum advancement (simplified from original)
def should_advance(stage_metrics, stage_config):
    """Advance when success rate exceeds threshold over last N episodes."""
    recent_kills = stage_metrics.recent_kill_rate(window=100)
    return recent_kills >= stage_config.success_threshold
```

---

## 14. Phase 10 — 3D Visualiser with Drone Model

### 14.1 Approach

- Use **PyVista** (VTK wrapper) for high-quality 3D rendering
- Load **.OBJ** drone models for both interceptor and target
- Render frame-by-frame animation and export as MP4
- Support both off-screen (headless) and interactive modes

### 14.2 Drone Model Sourcing

Download free quadcopter 3D models from:
1. **Free3D** (https://free3d.com/3d-models/quadcopter) — free OBJ/STL models
2. **GrabCAD** (https://grabcad.com/library?query=quadcopter) — engineering CAD models
3. **Sketchfab** (https://sketchfab.com/tags/drone) — some free OBJ downloads

Save to `models/interceptor_drone.obj` and `models/target_drone.obj`.

A `scripts/download_models.py` script will automate the download (or prompt the user to manually download and place the files).

### 14.3 `visualiser/renderer.py` — Scene Setup

```python
"""
renderer.py — PyVista-based 3D scene renderer.

Loads drone OBJ models, sets up lighting, ground plane,
and provides methods to update positions/orientations per frame.
"""

class InterceptRenderer:
    def __init__(self, config: dict):
        self.plotter = pv.Plotter(
            off_screen=config.get('offscreen', True),
            window_size=config.get('resolution', [1920, 1080]),
        )
        
        # Sky gradient background
        self.plotter.set_background('lightskyblue', top='midnightblue')
        
        # Ground plane
        ground = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1),
                          i_size=200, j_size=200, i_resolution=20, j_resolution=20)
        self.plotter.add_mesh(ground, color='darkseagreen', opacity=0.5)
        
        # Load drone models
        self.interceptor_mesh = pv.read(config['interceptor_model_path'])
        self.target_mesh = pv.read(config['target_model_path'])
        
        # Scale models to reasonable size
        self.interceptor_mesh = self.interceptor_mesh.scale(
            config.get('interceptor_scale', 0.01))
        self.target_mesh = self.target_mesh.scale(
            config.get('target_scale', 0.01))
        
        # Add to scene
        self.interceptor_actor = self.plotter.add_mesh(
            self.interceptor_mesh.copy(), color='steelblue',
            smooth_shading=True, specular=0.7)
        self.target_actor = self.plotter.add_mesh(
            self.target_mesh.copy(), color='crimson',
            smooth_shading=True, specular=0.7)
        
        # Trajectory trails
        self.interceptor_trail = []
        self.target_trail = []
        
        # Lighting
        self.plotter.add_light(pv.Light(
            position=(50, 50, 100), focal_point=(0, 0, 0),
            color='white', intensity=0.8))
        self.plotter.enable_shadows()
        
        # Coordinate axes
        self.plotter.add_axes()
    
    def update_frame(self, interceptor_pos, interceptor_quat,
                     target_pos, target_quat):
        """Update mesh positions and orientations for one frame."""
        # Transform interceptor mesh
        # Transform target mesh
        # Update trail lines
        # Update camera (chase-cam following interceptor)
```

### 14.4 `visualiser/animator.py` — Animation & Video Export

```python
"""
animator.py — Frame-by-frame animation renderer and MP4 exporter.

Usage:
    animator = InterceptAnimator(
        model_name="quadcopter_v2",
        stage=2,
        trajectory_data=trajectory_log,  # from env._trajectory_log
    )
    animator.render("output/stage2_intercept.mp4", fps=30, quality=10)
"""

class InterceptAnimator:
    def __init__(self, model_name: str, stage: int,
                 trajectory_data: list, config_path: str = "configs/visualiser.yaml"):
        self.model_name = model_name
        self.stage = stage
        self.trajectory = trajectory_data
        self.config = load_config(config_path)
        self.renderer = InterceptRenderer(self.config)
    
    def render(self, output_path: str, fps: int = 30, quality: int = 10):
        """Render the full trajectory to MP4."""
        self.renderer.plotter.open_movie(output_path, framerate=fps, quality=quality)
        
        # Add stage info text overlay
        self.renderer.plotter.add_text(
            f"Model: {self.model_name}  |  Stage {self.stage}  |  "
            f"APN + RL Interceptor",
            position='upper_left', font_size=14, color='white'
        )
        
        for frame_idx, frame_data in enumerate(self.trajectory):
            self.renderer.update_frame(
                interceptor_pos=frame_data['interceptor_pos'],
                interceptor_quat=frame_data['interceptor_quat'],
                target_pos=frame_data['target_pos'],
                target_quat=np.array([1, 0, 0, 0]),  # target heading from vel
            )
            
            # Update HUD
            self._update_hud(frame_idx, frame_data)
            
            # Write frame
            self.renderer.plotter.write_frame()
        
        self.renderer.plotter.close()
    
    def _update_hud(self, frame_idx, data):
        """Update heads-up display with telemetry."""
        # Distance, closing velocity, LOS rate, time, 
        # APN vs RL contribution, velocity output
```

### 14.5 `visualiser/cli.py` — Command-Line Interface

```python
"""
CLI for generating visualisation videos.

Usage:
    python -m visualiser --model quadcopter --stage 2 --output video.mp4
    python -m visualiser --model quadcopter --stage all --output-dir videos/
    python -m visualiser --model custom_model.obj --stage 3 --fps 60
"""
```

### 14.6 Camera Modes

The visualiser supports multiple camera perspectives:
1. **Chase cam** — follows behind the interceptor, looking forward
2. **Top-down** — bird's eye view showing both trajectories
3. **Fixed** — stationary camera observing the full engagement
4. **LOS cam** — camera on the interceptor looking at the target
5. **Free orbit** — slowly orbiting the engagement center

### 14.7 Visual Elements

- **Interceptor drone model** — colored blue/steel, with rotor disc animation
- **Target drone model** — colored red/crimson
- **Interceptor trajectory trail** — blue dashed line with fade
- **Target trajectory trail** — red dashed line with fade
- **LOS line** — thin yellow line between interceptor and target
- **APN acceleration vector** — green arrow from interceptor
- **RL correction vector** — orange arrow from interceptor
- **Total acceleration vector** — white arrow from interceptor
- **Ground plane** — semi-transparent grid
- **HUD overlay** — distance, speed, time, stage info
- **Kill zone sphere** — semi-transparent sphere around target showing kill radius

---

## 15. Phase 11 — Evaluation & Metrics

### 15.1 Key Metrics

| Metric | Description | Target |
|---|---|---|
| Kill rate | Fraction of episodes ending in intercept | >95% / >90% / >75% |
| Mean time-to-kill | Average steps to intercept | Minimise |
| Mean miss distance | Average closest approach distance | < kill_radius |
| Mean energy (∫|a|²dt) | Total acceleration effort | Minimise |
| Mean RL contribution | Average |Δa_RL| / |a_total| | Low for stage 1, higher for stage 3 |
| ZEM at terminal | Zero Effort Miss at intercept time | < 0.1 m |
| Trajectory efficiency | Path length / straight-line distance | Close to 1.0 |

### 15.2 Evaluation Script

```bash
python scripts/evaluate.py \
    --model data/runs/apn_rl_v1/ppo_interceptor/final.zip \
    --stage 3 \
    --episodes 100 \
    --render \
    --output-dir data/evaluation/
```

---

## 16. Phase 12 — Testing & Validation

### 16.1 Smoke Tests (No GPU Required)

1. **APN unit test** — verify guidance command against analytic solution for known geometry
2. **Kinematic model test** — verify position/velocity integration
3. **Observation dimension test** — verify obs vector sizes for all stage configs
4. **Reward computation test** — verify reward terms for known inputs
5. **Target generator test** — verify static/polynomial/evasive target paths
6. **Stage config test** — verify 3-stage curriculum table
7. **Optimal trajectory test** — verify oracle solution for straight-line intercept
8. **History stacking test** — verify observation builder with mock frames

### 16.2 Integration Tests

1. **APN-only interception** — run environment with zero RL action; verify APN alone intercepts static targets
2. **RL training smoke** — run 1000 steps of PPO training; verify loss decreases
3. **Visualiser test** — render 10-frame test animation; verify MP4 output
4. **End-to-end** — train 10K steps, evaluate, visualise

---

## 17. File-by-File Implementation Specification

### 17.1 Core Implementation Order

The files should be implemented in this exact order to maintain testability at each step:

```
 1. configs/stages.yaml            — define 3-stage curriculum
 2. configs/config.yaml            — training and observation config
 3. configs/model.yaml             — PPO model definition
 4. configs/visualiser.yaml        — visualiser settings
 5. src/guidance/apn.py            — APN guidance law (CORE)
 6. src/guidance/kinematic_model.py — point-mass kinematics
 7. src/guidance/optimal_trajectory.py — critic oracle
 8. src/envs/stage_config.py       — stage definitions + dataclasses
 9. src/envs/target_generator.py   — adapt from old code
10. src/envs/obs_builder.py        — rewrite frame layout
11. src/envs/reward.py             — new reward terms
12. src/envs/intercept_env.py      — main Gymnasium environment
13. src/prediction/base.py         — predictor interface
14. src/prediction/const_vel.py    — constant velocity predictor
15. src/prediction/kalman.py       — Kalman filter predictor
16. src/training/asymmetric_policy.py — RETAIN AS-IS
17. src/training/orchestrator.py   — simplify for 3 stages
18. src/training/callbacks.py      — update metric names
19. src/training/curriculum.py     — simplify
20. src/training/checkpoint_manager.py — RETAIN AS-IS
21. src/training/weight_transfer.py — RETAIN AS-IS
22. src/utils/*                    — RETAIN / minor updates
23. src/api.py                     — simplify
24. src/results.py                 — minor updates
25. visualiser/renderer.py         — PyVista scene setup
26. visualiser/animator.py         — frame-by-frame MP4 export
27. visualiser/scene_builder.py    — model loading, lighting
28. visualiser/cli.py              — command-line interface
29. scripts/download_models.py     — model download helper
30. scripts/train.py               — training entry point
31. scripts/evaluate.py            — evaluation + metrics
32. scripts/visualise.py           — generate demo videos
33. scripts/smoke_test.py          — validation suite
34. run.py                         — one-file user script
35. requirements.txt               — updated dependencies
36. README.md                      — project documentation
37. context.md                     — project context (updated)
```

### 17.2 Detailed File Specifications

#### File 5: `src/guidance/apn.py` (≈150 lines)

```
PURPOSE: 3D Augmented Proportional Navigation guidance law
INPUTS:  Interceptor state (pos, vel), Target state (pos, vel, accel), N'
OUTPUTS: 3D acceleration command, engagement diagnostics

FUNCTIONS:
  compute_apn_command(p_int, v_int, p_tgt, v_tgt, a_tgt, N', t_go_min)
    → (a_cmd: ndarray(3), info: dict)
  
  compute_los_rate(R_vec, V_rel)
    → omega_LOS: ndarray(3)
  
  compute_closing_velocity(V_rel, R_hat)
    → V_c: float
  
  compute_time_to_go(R, V_c, t_go_min)
    → t_go: float
  
  compute_zem(V_rel, a_target, t_go, R_hat)
    → ZEM: float, ZEM_vec: ndarray(3)

DEPENDENCIES: numpy only
TESTS: 8 unit tests (head-on, crossing, maneuvering, edge cases)
```

#### File 6: `src/guidance/kinematic_model.py` (≈100 lines)

```
PURPOSE: Point-mass interceptor kinematics (no rigid body)
INPUTS:  State, acceleration command, angular velocity command
OUTPUTS: Updated state after dt integration

CLASSES:
  InterceptorState(position, velocity, orientation, angular_velocity)
  KinematicModel(v_max, a_max, omega_max, dt)
    .step(state, a_cmd, omega_cmd) → InterceptorState
    .reset(position, velocity) → InterceptorState

DEPENDENCIES: numpy, scipy.spatial.transform (for quaternion updates)
```

#### File 7: `src/guidance/optimal_trajectory.py` (≈200 lines)

```
PURPOSE: Compute optimal intercept trajectory (critic oracle)
INPUTS:  Interceptor state, complete target path, constraints
OUTPUTS: Optimal acceleration/velocity sequence, cost

FUNCTIONS:
  compute_optimal_intercept(p_int, v_int, target_path, v_max, a_max, dt)
    → dict with optimal_time, optimal_accel, optimal_vel, cost
  
  _optimal_static(p_int, v_int, p_target, v_max, a_max, dt)
    → straight-line trapezoidal profile
  
  _optimal_polynomial(p_int, v_int, poly_path, v_max, a_max, dt)
    → energy-optimal intercept of polynomial path
  
  _optimal_known_future(p_int, v_int, future_states, v_max, a_max, dt)
    → greedy optimal given known future target states

DEPENDENCIES: numpy, scipy.optimize (for time-optimal solution)
```

#### File 12: `src/envs/intercept_env.py` (≈400 lines)

```
PURPOSE: Main Gymnasium environment for 3-stage interception
INPUTS:  Action (3D RL residual acceleration correction)
OUTPUTS: Observation, reward, terminated, truncated, info

CLASS: InterceptEnv(gymnasium.Env)
  __init__(stage_config, obs_config, apn_config, interceptor_config)
  reset(seed, options) → obs, info
  step(action) → obs, reward, terminated, truncated, info
  _build_observation() → ndarray
  _compute_heading_rate(a_cmd) → omega_cmd
  _check_bounds() → bool
  get_trajectory_log() → list[dict]

DEPENDENCIES: gymnasium, numpy, src.guidance, src.envs.*
```

---

## 18. Dependency List

### 18.1 `requirements.txt`

```
# Core
numpy>=1.24
scipy>=1.11
gymnasium>=0.29

# RL Training
stable-baselines3>=2.1
torch>=2.0

# Visualisation
pyvista>=0.43
trimesh>=4.0
imageio>=2.31
imageio-ffmpeg>=0.4

# Configuration
pyyaml>=6.0

# Logging
tensorboard>=2.14

# Optional: target prediction
filterpy>=1.4           # Kalman filter library

# Dev / Testing
pytest>=7.0
```

### 18.2 Installation

```bash
pip install -r requirements.txt
```

---

## 19. Implementation Timeline

| Phase | Description | Estimated Effort | Dependencies |
|---|---|---|---|
| **Phase 0** | Git backup + GitHub push | 30 min | None |
| **Phase 1** | Clean slate (delete old, identify reusable) | 1 hour | Phase 0 |
| **Phase 2** | Create new directory structure | 30 min | Phase 1 |
| **Phase 3** | APN engine + kinematic model | 3 hours | Phase 2 |
| **Phase 4** | Gymnasium environment | 4 hours | Phase 3 |
| **Phase 5** | Observation builder (adapt from old) | 2 hours | Phase 4 |
| **Phase 6** | Reward function | 1.5 hours | Phase 4 |
| **Phase 7** | Optimal trajectory oracle | 3 hours | Phase 3 |
| **Phase 8** | RL training pipeline (adapt from old) | 3 hours | Phase 4-7 |
| **Phase 9** | 3-stage curriculum configs | 1 hour | Phase 8 |
| **Phase 10** | 3D visualiser (PyVista) | 5 hours | Phase 4 |
| **Phase 11** | Evaluation scripts | 2 hours | Phase 8, 10 |
| **Phase 12** | Smoke tests + validation | 3 hours | All |
| | **Total** | **~30 hours** | |

---

> [!IMPORTANT]
> The implementation order is strictly: **Phase 0 → 1 → 2 → 3 → 4 → 5/6/7 (parallel) → 8 → 9 → 10 → 11 → 12**. Each phase is testable independently before moving to the next.

> [!TIP]
> The `target_generator.py` from the old codebase is the most reusable module — it already implements static, polynomial (order 1-3), and evasive target types with the exact kinematics we need. The `asymmetric_policy.py` is retained completely unchanged.

> [!NOTE]
> The RL agent's action space is intentionally small (3D residual acceleration) because APN handles the baseline guidance. This makes training much faster than learning guidance from scratch. The agent only needs to learn when and how much to deviate from APN's recommendation.

---

*Plan version: 1.0 — Generated 2026-10-16*
