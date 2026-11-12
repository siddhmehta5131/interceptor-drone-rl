# APN + RL Autonomous Interceptor Drone

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![StableBaselines3](https://img.shields.io/badge/SB3-2.x-green)
![PyVista](https://img.shields.io/badge/PyVista-0.43%2B-purple)

A **hybrid guidance** system that intercepts a manoeuvring target in 3D. A
classical **Augmented Proportional Navigation (APN)** law supplies the bulk of
the acceleration command, and a **PPO** policy (Stable-Baselines3) learns a
small *residual* correction on top of it:

```
a_cmd = APN_command(target_state, interceptor_state) + RL_residual(observation)
```

The environment outputs **velocity commands** (`vx, vy, vz, ωx, ωy, ωz`) — it is
a pure 3D path-planning problem, not a motor-level quadrotor simulation. A
3-stage curriculum progresses from a static target, through predictable
polynomial paths, to a fully evasive target. A PyVista visualiser renders
full-length MP4 demonstrations.

---

## 🏗️ Architecture

```
run.py / scripts/train.py
  │
  ├── configs/model.yaml      ← PPO architecture, net_arch, privileged critic fields
  ├── configs/config.yaml     ← training settings, observation layout, interceptor + APN
  ├── configs/stages.yaml     ← the 3-stage curriculum table (authoritative)
  │
  └── src.api.train_model(source, stages, name, ...) → TrainResult
         │
         └── Orchestrator: build InterceptEnv → AsymmetricActorCriticPolicy → run stages
```

Inside one environment step:

```
action (3D, [-1,1])  ──× rl_action_scale──►  Δa_RL
target.advance(dt)   ──►  compute_apn_command ──►  a_APN
a_total = clamp(a_APN + Δa_RL, a_max)
ω = cross(velocity_hat, a_total)          # heading-rate command
interceptor = KinematicModel.step(a_total, ω)
```

### APN law

For relative position `R = p_T − p_M`, relative velocity `V_rel = v_T − v_M`,
LOS rate `ω_LOS = (R × V_rel)/|R|²`, time-to-go `t_go = R / max(V_c, 0.1)`:

```
a_APN = N' · perp(V_rel) / t_go  +  (N'/2) · perp(a_T)
perp(x) = x − (x · R̂) R̂
```

with navigation constant `N' ≈ 4.0`. (The plan's literal cross-product form is
not dimensionally consistent along the LOS; the equivalent zero-effort-miss
form above is implemented — see `docs/BUGS.md`.)

### Three-stage curriculum

| Stage | Target | `rl_action_scale` | Step budget | Advance at |
|---|---|---|---|---|
| 1 | Static | `1.5` | 2 M | kill rate ≥ 95 % |
| 2 | Polynomial path (order 1) | `4.5` | 5 M | kill rate ≥ 90 % |
| 3 | Evasive (reactive) | `7.5` | 10 M | kill rate ≥ 75 % |

### Observation layout

* **Actor** = 85-dim: `16 × (1 + history_frames) + 7 × future_samples`
  * single 16-dim frame = `rel_pos(3), rel_vel(3), interceptor_vel(3), range(1),
    closing_vel(1), LOS_rate(3), t_go_est(1), visible(1)`
  * history: 4 frames spaced 2 steps apart
  * future: 3 predicted sample pairs (rel_pos, rel_vel, visible) at horizons of
    5 steps
* **Privileged critic** = 16-dim (oracle-only, never shown to the actor):
  `time_remaining(1), optimal_accel(3), optimal_vel(3),
  target_true_future_pos(3), target_true_future_vel(3), target_true_accel(3)`
* **Env observation space** = actor ⊕ privileged = **101-dim**.

The `AsymmetricActorCriticPolicy` slices the first 85 features for the actor and
feeds the full 101 to the critic.

---

## 🚀 Quick start

### Install

```bash
pip install -r requirements.txt
```

### Train (smoke budget)

```bash
python scripts/train.py --stages 1 --name smoke --steps 20000
```

### Train the full curriculum

```bash
python scripts/train.py --stages 1,2,3 --name apn_rl_v1
# final model → data/runs/apn_rl_v1/ppo_interceptor/final.zip
```

### Evaluate

```bash
python scripts/evaluate.py \
  --model data/runs/apn_rl_v1/ppo_interceptor/final.zip \
  --stage 3 --episodes 100 --render --output-dir data/evaluation
```

### Visualise (APN-only, no trained model needed)

```bash
python -m visualiser --model apn --stage 2 --output data/videos/stage2_apn.mp4
# or every stage:
python -m visualiser --model apn --stage all --output-dir data/videos
```

### Test

```bash
python -m pytest tests/ -q -m "not slow"   # 52 fast tests
python -m pytest tests/ -q                 # + end-to-end integration
python scripts/smoke_test.py               # dependency-light 11-check harness
```

---

## 📁 Repository layout

```
├── run.py                      ← one-file training runner (edit this)
├── conftest.py                 ← pytest sys.path bootstrap + 'slow' marker
├── configs/
│   ├── stages.yaml             ← 3-stage curriculum (authoritative table)
│   ├── model.yaml              ← PPO architecture + privileged critic fields
│   ├── config.yaml             ← training settings, obs layout, interceptor, APN
│   └── visualiser.yaml         ← PyVista scene / camera / HUD settings
├── src/
│   ├── api.py                  ← train_model() / evaluate_model()
│   ├── results.py              ← TrainResult / ModelResult / StageOutcome
│   ├── guidance/               ← apn.py, kinematic_model.py, optimal_trajectory.py
│   ├── envs/                   ← stage_config, target_generator, obs_builder,
│   │                             reward, intercept_env
│   ├── prediction/             ← const_vel + kalman target predictors
│   ├── training/               ← orchestrator, callbacks, curriculum, policy,
│   │                             checkpoint_manager, weight_transfer
│   └── utils/                  ← config_loader, model_loader, logger
├── visualiser/                 ← renderer, animator, scene_builder, cli
├── scripts/                    ← train, evaluate, visualise, smoke_test,
│                                 download_models
├── tests/                      ← pytest suite (unit + integration)
├── models/                     ← interceptor_drone.obj, target_drone.obj
└── data/                       ← runs/, videos/, evaluation/
```

---

## ⚙️ Configuration

### `configs/model.yaml`

```yaml
ppo_interceptor:
  algo: PPO
  policy: MlpPolicy            # auto-upgraded to AsymmetricActorCriticPolicy
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

### `configs/config.yaml`

```yaml
global:
  seed: 42
  device: cpu            # or cuda
  data_dir: data/runs
  n_parallel_envs: 8
  checkpoint_interval_steps: 100000

on_capped: stop          # 'stop' or 'continue' when a stage hits its step budget

observation:
  frame_dim: 16
  history_frames: 3
  history_skip: 2
  future_samples: 3
  future_skip: 5
  future_source: pred    # 'pred' (predictor) or 'true' (ground truth)
  predictor: const_vel   # 'const_vel' or 'kalman'

interceptor: {v_max: 50.0, a_max: 15.0, omega_max: 3.0, speed_ratio: 1.5, dt: 0.02}
apn:         {N_prime: 4.0, t_go_min: 0.05, t_go_max: 20.0}
```

---

## 🔧 `train_model()` API

```python
from src.api import train_model

result = train_model(
    source="configs/model.yaml",     # model YAML (or a previous TrainResult)
    stages=[1, 2, 3],                # curriculum stages to run
    name="apn_rl_v1",                # folder under data/runs/
    config="configs/config.yaml",
    device=None,                     # 'cpu' | 'cuda' | None (auto)
    max_steps_per_stage=None,        # cap the per-stage budget (smoke tests)
    verbose=1,
)

model = result.ppo_interceptor.model          # SB3 PPO object
model.save("data/models/apn_rl_v1_final.zip")
print(result.summary())
```

`result["ppo_interceptor"]` also works (results are keyed by model id); a missing
id raises `AttributeError`.

---

## 🔬 Guidance & reward

**Reward terms** (`RewardConfig`, all stage-tunable): normalized progress,
LOS-rate, closing-velocity, energy (`−|a_total|²`), RL-residual regularisation,
action smoothness, time penalty, ZEM penalty, and a kill bonus scaled by miss
accuracy and time. Out-of-bounds terminates with the crash penalty.

**Predictors**: `const_vel` and `kalman` (FilterPy, with an analytic
constant-acceleration fallback) forecast the target for the actor's future
block; the privileged critic uses the *true* future.

**Oracle**: `compute_optimal_intercept()` builds a velocity/acceleration-constrained
minimum-time trajectory (trapezoidal / cubic-Hermite profiles) and supplies the
privileged `optimal_*` fields.

---

## 🎥 Visualiser

PyVista off-screen renderer with drone meshes, dashed trails, a yellow
line-of-sight line, green (APN) / orange (RL) / white (total) command arrows, a
kill-zone sphere and a live HUD. Camera modes: `chase`, `top-down`, `fixed`,
`los`, `orbit`. Model meshes live in `models/`; `scripts/download_models.py`
fetches real `.obj` files or writes placeholder quadcopters (see
`models/README.md` for attribution).

---

## 📄 License

No license — all rights reserved.
