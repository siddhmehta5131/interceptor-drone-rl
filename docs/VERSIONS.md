# Version Registry

**Project:** APN + RL Autonomous Interceptor Drone
**Last updated:** 2026-10-16

---

## v1.0.0 — APN + RL path-planning interceptor (2026-10-16)

Full pivot from the previous physics-based quadrotor simulator (RK4 / PID /
motor RPM / Crazyflie, archived at `github.com/siddhmehta5131/interceptor-drone-rl`)
to a **pure 3-D path-planning** interceptor: classical **Augmented Proportional
Navigation** plus a **PPO residual** policy over a 3-stage curriculum.

### Core modules

| File | Role |
|---|---|
| `src/guidance/apn.py` | APN guidance law (ZEM-equivalent form) |
| `src/guidance/kinematic_model.py` | Velocity/attitude integrator (InterceptorState) |
| `src/guidance/optimal_trajectory.py` | Min-time oracle for the privileged critic |
| `src/envs/intercept_env.py` | Gymnasium env (obs 101, action 3) |
| `src/envs/obs_builder.py` | 16-dim frames + history/future + privileged block |
| `src/envs/reward.py` | 13-term reward (progress, LOS, energy, ZEM, kill, …) |
| `src/envs/target_generator.py` | Static / polynomial / evasive target paths |
| `src/envs/stage_config.py` | Stage + reward + spawn dataclasses |
| `src/prediction/` | `const_vel`, `kalman` target predictors |
| `src/training/orchestrator.py` | PPO training + evaluation driver |
| `src/training/asymmetric_policy.py` | Asymmetric actor-critic (actor 85 / critic 101) |
| `src/api.py` | `train_model()` / `evaluate_model()` public API |
| `visualiser/` | PyVista off-screen renderer + MP4 animator |
| `scripts/` | train, evaluate, visualise, smoke_test, download_models |

### Configuration

| File | Role |
|---|---|
| `configs/stages.yaml` | 3-stage curriculum table (authoritative) |
| `configs/model.yaml` | PPO architecture + privileged critic fields |
| `configs/config.yaml` | global / observation / interceptor / APN settings |
| `configs/visualiser.yaml` | PyVista scene, camera, HUD settings |

### Notes

* **Observation** — actor 85, privileged critic 16, env 101.
* **Action** — `Box([-1, 1], (3,))` residual acceleration, scaled per stage
  (1.5 / 4.5 / 7.5 m/s²).
* **Default `dt`** — 0.02 s (50 Hz).
* **Deviation from plan §2.3** — the literal APN cross-product is replaced by the
  equivalent zero-effort-miss form; see `BUGS.md` item 1.
