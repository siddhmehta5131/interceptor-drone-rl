# RL-Based Autonomous Interceptor Drone

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![NumPy](https://img.shields.io/badge/NumPy-Simulation-013243?logo=numpy)
![Gymnasium](https://img.shields.io/badge/Gymnasium-RL%20Env-3776AB?logo=openai)
![Stable-Baselines3](https://img.shields.io/badge/SB3-Dockerized%20Training-2ea44f)
![Docker](https://img.shields.io/badge/Docker-CUDA-2496ed?logo=docker)
![License](https://img.shields.io/badge/License-MIT-yellow)

A complete **autonomous interceptor-drone** project: a real-time physics-based
quadrotor flight simulator and a **Dockerized multi-stage reinforcement-learning
training pipeline** that learns to intercept a moving target.

The simulator implements a full 9-stage rigid-body dynamics pipeline calibrated
to the Crazyflie 2.0/2.1 nano-quadrotor, with aerodynamic coefficients fitted
from real PX4 flight-log data and live gamepad control. The training pipeline
sits on top of it: an 8-stage curriculum (hover → waypoint → evasive intercept)
trained with Stable-Baselines3 (PPO/SAC/TD3) inside a CUDA Docker container.

---

## 🧠 Architecture

### Flight simulator — 9-stage physics pipeline

```
Gamepad input (thrust, roll, pitch, yaw)
    │
    ▼
Stage A ─── Rate PID Controller ─────────→ torque command u
Stage B ─── Mixer ────────────────────────→ per-motor commands
Stage 2 ─── ESC / Battery Model ──────────→ steady-state motor speeds
Stage 3 ─── Motor Dynamics (1st-order) ───→ actual motor speeds + derivatives
Stage 4 ─── Propeller Force/Torque ───────→ per-prop thrust + drag
Stage 5 ─── Force Aggregation ────────────→ total body wrench
Stage 6 ─── Gyroscopic Torques ───────────→ reaction + inertial coupling
Stage 7 ─── Aerodynamic Drag Model ───────→ polynomial drag (fitted from PX4)
Stage 8 ─── Newton-Euler Dynamics ────────→ state derivatives
Stage 9 ─── RK4 Integration ──────────────→ new position + attitude
    │
    ▼
Live 3D matplotlib visualisation (position trail + body-frame arrows)
```

### RL training — curriculum (see `implementation_plan.md`)

```
stage_1  hover (bit-identical to hover_env.AltitudeHoldEnv)
stage_2  waypoint tracking (static target)
stage_3  intercept, static target        ── kill bonus scaled by miss distance
stage_4  + time-to-kill bonus            ── + adaptive time penalty
stage_5  moving target (order-1)         ── look-ahead prediction on
stage_6  + noisy/missing target obs      ── + normalized progress reward
stage_7  evasive target (order-2/3)      ── + wind / domain randomisation (opt.)
stage_8  full evasive intercept          ── optional ground effect (opt.)

stage success  → advance     (windowed success rate ≥ threshold)
stage failure  → rollback    (≤ 2 retries) then cap
```

Stage 1 must be **bit-identical** to the reference `hover_env.py` (same seed +
action stream → identical obs arrays, rewards, terminations and infos). This is
enforced by `interceptor-training/scripts/smoke_test.py`.

---

## ✨ Features

- 🎮 **Live gamepad control** — fly the drone with any USB controller (tested on Amkette Evo Gamepad Pro 4)
- 📐 **Paper-derived physics** — all equations follow standard rigid-body + propeller aerodynamics theory
- 📊 **Real-world calibration** — mass, inertia, arm length from Crazyflie 2.0 published data
- 🌊 **Fitted aero model** — drag coefficients from least-squares regression on 15 PX4 flight logs
- 🔋 **ESC polynomial** — battery/motor model fitted to real ESC telemetry
- 📈 **Dual live visualisation** — 3D trajectory + 4-channel input bar chart
- 🤖 **8-stage RL curriculum** — hover → waypoint → moving/evasive intercept, with advance/rollback scheduling
- 🐳 **Dockerized training** — CUDA container, PPO/SAC/TD3 via Stable-Baselines3, TensorBoard logging
- 🔄 **Resumable runs** — stage checkpoints, run-state JSON, off-policy replay buffers persisted
- 🧪 **Host-side verification** — 8-test smoke suite (no torch/SB3 needed) that also enforces stage-1 parity

---

## 🗂️ Project Structure

```
interceptor-drone-rl/
├── src/                     # Flight simulator (gamepad + 9-stage physics + visualiser)
│   ├── simulation.py
│   └── visualiser.py
├── interceptor-training/    # Dockerized RL training pipeline
│   ├── Dockerfile           #   pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
│   ├── docker-compose.yml   #   GPU reservation, ./data:/data volume
│   ├── configs/             #   default_run.yaml — global + 4 algorithm configs
│   ├── src/                 #   physics/, envs/ (8 stages), training/, utils/
│   ├── scripts/             #   train.py (entrypoint), evaluate.py, smoke_test.py
│   └── data/                #   host-mounted volume (checkpoints, tb_logs, ...)
├── docs/
│   └── coefficient_fitting.md   # how the aero + ESC coefficients were fitted
├── hover_env.py             # reference environment (stage-1 parity target)
├── implementation_plan.md   # authoritative spec for the RL training pipeline
├── CURRENT_WORK.md          # build status / verification log
├── BUGS.md                  # known non-blocking issues
├── requirements.txt
├── .gitignore
└── README.md
```

### Module Breakdown

| File | Purpose |
|------|---------|
| `src/visualiser.py` | 3 standalone functions: `plot_trajectory()`, `plot_input_bars()`, `read_gamepad()` — fully decoupled from physics |
| `src/simulation.py` | 10 stage functions (PID → Mixer → ESC → Motors → Props → Forces → Gyro → Aero → Dynamics → RK4 Integration) + calibration constants + live loop |
| `interceptor-training/src/physics/` | Constants, quaternion, aero, dynamics pipeline — ported from `hover_env._ph_*` |
| `interceptor-training/src/envs/` | `stage_config` (8-stage table), `obs_builder` (history stacking), `reward` (plan §5.5 terms), `target_generator`, `base_env` |
| `interceptor-training/src/training/` | `curriculum` scheduler, `checkpoint_manager`, `callbacks` (TB/logging), `orchestrator` (stage loop) |
| `interceptor-training/scripts/train.py` | Container entrypoint; also `--smoke` for an in-container integration run |

---

## ⚙️ Prerequisites

- Python 3.10+ (simulator & smoke suite)
- A USB gamepad / joystick (any standard dual-stick layout) — simulator only
- Display capable of running matplotlib interactively — simulator only
- Docker + NVIDIA GPU driver + nvidia-container-toolkit — training only

---

## 🔨 Setup

```bash
# 1. Clone the repo
git clone https://github.com/siddhmehta5131/interceptor-drone-rl
cd interceptor-drone-rl

# 2. Create virtual environment (simulator + smoke suite)
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install host dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the flight simulator

```bash
cd src
python simulation.py
```

Two windows open: **3D Trajectory** (position trail + body-frame arrows) and
**Input Bar Chart** (live thrust/roll/pitch/yaw). Push the left stick up for
thrust; right stick controls roll/pitch.

### Verify the training pipeline on the host (no GPU/torch needed)

```bash
cd interceptor-training
python scripts/smoke_test.py          # 8/8 tests: parity, envs, reward, scheduler, ...
```

### Train in the container (CUDA host)

```bash
cd interceptor-training
docker compose up --build                                     # default run config
docker compose run --rm interceptor-train --smoke \           # quick in-container smoke
  --smoke-config ppo_baseline --smoke-stages 1,2,3 --smoke-steps 1000
```

Training output lands in `interceptor-training/data/` (mounted at `/data`):
`checkpoints/` (per-stage final + periodic), `tb_logs/` (TensorBoard),
`curriculum_state/` (resume state), `results/` (final model + evaluation JSON).

### Evaluate a trained model

```bash
docker compose run --rm --entrypoint python interceptor-train scripts/evaluate.py \
  --config /data/configs/default_run.yaml --config-name ppo_baseline --stage 3 --episodes 50
```

---

## 🔧 Physical Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Mass | 27 g | Crazyflie 2.0 (Forster thesis) |
| Inertia Ixx/Iyy | 1.4 × 10⁻⁵ kg·m² | Published |
| Inertia Izz | 2.17 × 10⁻⁵ kg·m² | Published |
| Arm length | 39.7 mm | Published |
| Lift coefficient c_l | 5.0 × 10⁻⁸ N·(rad/s)⁻² | Tuned for hover at 50% stick |
| Drag coefficient c_d | 1.25 × 10⁻⁹ N·m·(rad/s)⁻² | Forster ratio |
| Motor time constant | 20 ms | Best estimate |
| PID gains (Kp) | [0.15, 0.15, 0.20] | Median across 15 PX4 logs |
| Aero drag coefficients | 6 polynomials | Fitted via least-squares |
| ESC polynomial | 5 coefficients | Fitted to ESC telemetry |

---

## 🤖 RL Training in Brief

- **Stages 1–8** live in `interceptor-training/src/envs/stage_config.py` with the
  plan's tuned reward weights (kill bonus, time/miss-distance scaling, progress,
  alignment, smoothness/angular-velocity penalties).
- **Algorithm configs** (`configs/default_run.yaml`): `ppo_baseline`,
  `sac_baseline`, `td3_baseline`, `ppo_5layer_deep` (deeper net + longer obs
  history: 5 frames @ 3-step skip vs the default m=3/s=2).
- **Curriculum** advances a stage once its windowed success rate clears the
  threshold (60–85% depending on stage); failures roll back at 50% of the
  threshold, max 2 retries per stage.
- Full specification: [`implementation_plan.md`](implementation_plan.md); build
  status and verified behaviour: [`CURRENT_WORK.md`](CURRENT_WORK.md); known
  non-blocking issues: [`BUGS.md`](BUGS.md).

---

## 🐛 Bugs Fixed (flight simulator, from development versions)

1. **Division by zero in `mixer()`** — when all motors command the same value (`cmd_max_val == c_cmd`), the scaling denominator is zero. Added `abs(denom) > 1e-9` guard.
2. **Double-step motor integration** — `rigid_body_dynamics` was receiving `Omega_new` (already integrated in Stage 3) and then Euler-integrating it again in Stage 9. Fixed to pass old `Omega`.
3. **Reverse-spinning motors** — `Omega` could go negative during fast transients. Clamped with `np.maximum(Omega, 0.0)`.
4. **ESC command inversion** — the fitted ESC polynomial maps cmd=0 → max speed (firmware convention). Added `cmd_esc = 1 - cmd` to align with the simulation's convention.
5. **Throttle-cut ESC bypass** — the ESC polynomial has a non-zero intercept at cmd=0, causing phantom lift. When throttle is cut, the entire ESC model is now bypassed.
6. **Duplicate `n_axes` call** — `joy.get_numaxes()` was called twice in `read_gamepad()`. Removed duplicate.
7. **Global aero coefficients** — `aerodynamic_force_torque()` accepted an `aero_coeffs` parameter but used module globals. Rewritten to accept all 6 coefficient arrays explicitly.

---

## 📄 License

MIT © 2025 **Siddh Mehta**