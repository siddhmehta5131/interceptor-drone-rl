# RL-Based Autonomous Interceptor Drone

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0-EE4C2C?logo=pytorch)
![StableBaselines3](https://img.shields.io/badge/SB3-2.x-green)
![Docker](https://img.shields.io/badge/Docker-GPU-2496ED?logo=docker)

A **reinforcement-learning training pipeline** for an autonomous interceptor drone. A Crazyflie 2.0-frame rigid-body physics simulator (NanoBench-calibrated mass and inertia, 9-stage RK4 pipeline at 100 Hz) is combined with an 8-stage curriculum that teaches a quadrotor to hover, track, and intercept a manoeuvring target.

---

## 🏗️ Architecture

```
run.py                     ← one-file user script
  │
  ├── configs/model.yaml   ← define algorithms, network shape, critic fields
  ├── configs/config.yaml  ← training settings, observation layout, target params
  │
  └── train_model(source, stages, name, ...)  →  TrainResult
         │
         ├── per model_id: build env → build policy → run curriculum
         └── returns TrainResult[model_id] with the final trained model
```

### Eight-stage curriculum

| Stage | Type | Target | Obs dim (ppo_baseline) |
|---|---|---|---|
| 1 | Hover | None | 14 |
| 2 | Directional flight | Linear path | 105 |
| 3 | Intercept (static approach) | Linear path | 105 |
| 4 | Intercept (timed) | Linear path | 105 |
| 5 | Polynomial order-1 intercept | Polynomial | 105 |
| 6 | Polynomial order-2 intercept | Polynomial | 105 |
| 7 | Polynomial order-3 intercept | Polynomial | 105 |
| 8 | Evasive (smoke-only) | Evasive | 105 |

---

## 🚀 Quick start

### Option A — Docker (recommended, GPU training)

```bash
# 1. Clone
git clone https://github.com/siddhmehta5131/interceptor-drone-rl.git
cd interceptor-drone-rl/interceptor-training

# 2. Build image
docker build -t interceptor-drone-rl:v6.0 .

# 3. Run smoke test (no GPU required)
docker run --rm interceptor-drone-rl:v6.0 --smoke

# 4. Train (requires NVIDIA GPU + nvidia-docker)
docker run --gpus all --rm \
  -v $(pwd)/data:/data \
  interceptor-drone-rl:v6.0 \
  --source configs/model.yaml --stages 1,2,3,4 --name experiment_1

# 5. Continue from that run on later stages
docker run --gpus all --rm \
  -v $(pwd)/data:/data \
  interceptor-drone-rl:v6.0 \
  --continue-from /data/runs/experiment_1 --stages 5,6,7 --name experiment_2
```

`scripts/train.py` exits `0` when every requested model finished (`completed` or
`capped`) and `1` otherwise, so it can gate CI directly.

### Option B — run.py (host Python, development)

```bash
cd interceptor-training
pip install -r requirements.txt
python run.py
```

Edit `run.py` to configure your experiment:

```python
from src.api import train_model

# Train through stages 1–4
model1 = train_model(
    source="configs/model.yaml",
    stages=[1, 2, 3, 4],
    name="experiment_1",
    config="configs/config.yaml",
)

# Continue on stages 5–7 from the result
model2 = train_model(
    source=model1,
    stages=[5, 6, 7],
    name="experiment_2",
)

# Save the final model
model2.ppo_baseline.save("my_models/intercept_v1")
```

---

## 📁 Repository layout

```
interceptor-training/
├── run.py                      ← one-file training runner (edit this)
├── configs/
│   ├── model.yaml              ← model definitions (algo, arch, critic fields)
│   ├── config.yaml             ← training settings (rollback, obs layout, ...)
│   ├── smoke_model.yaml        ← smoke test model definitions
│   └── smoke_config.yaml       ← smoke test config
├── scripts/
│   ├── train.py                ← Docker ENTRYPOINT / CLI
│   ├── evaluate.py             ← evaluation script
│   ├── esc_diagnostic.py       ← read-only ESC/thrust diagnostic (BUGS.md 13)
│   └── smoke_test.py           ← 12-group no-SB3 verification suite
└── src/
    ├── api.py                  ← train_model() public function
    ├── results.py              ← TrainResult / ModelResult types
    ├── envs/                   ← Gymnasium environments (8-stage curriculum)
    ├── physics/                ← rigid-body dynamics pipeline (RK4)
    ├── prediction/             ← const_vel + linear_ridge target predictors
    ├── training/               ← orchestrator, callbacks, curriculum, checkpointing
    └── utils/                  ← config/model loaders, logger
hover_env.py                    ← Stage-1 parity reference (AltitudeHoldEnv)
```

---

## ⚙️ Configuration

### `configs/model.yaml` — model definitions

```yaml
ppo_baseline:
  algo: PPO
  policy: MlpPolicy
  net_arch: {pi: [256, 256, 128], vf: [256, 256, 128]}
  activation: ReLU
  obs_history: {frames: 3, skip: 2}
  hyperparameters:
    n_steps: 4096
    batch_size: 512
    gamma: 0.995
    learning_rate: 3.0e-4
  privileged_critic:            # critic-only inputs (actor never sees these)
    - time_remaining
    - facing_error
    - target_true_pos
    - target_true_vel
```

### `configs/config.yaml` — training settings

```yaml
global:
  seed: 42
  device: cpu            # or cuda
  data_dir: /data
  n_parallel_envs: 8

on_capped: continue      # 'continue' or 'stop' when a stage hits its step budget

observation:
  history_frames: 3      # m past frames
  history_skip: 2        # p spacing
  future_samples: 3      # n future samples
  future_skip: 5         # q spacing
  future_source: true    # 'true' (ground truth) or 'pred' (predictor)
  predictor: const_vel   # 'const_vel' or 'linear_ridge'

target_alt: 5.0          # Stage-1 hover altitude in metres (or {min: 3, max: 7})
```

> ⚠️ `future_source` must be `pred` for **stage 8** — the evasive target has no
> closed-form path, so ground-truth futures do not exist and the predictor
> (`const_vel` or `linear_ridge`) is used instead. For stages 3–7 both spellings
> are available; `true` leaks the answer and makes the task unrealistically
> easy, `pred` is what a deployed drone would actually have. Stage 1 has no
> target block at all.

---

## 🔧 `train_model()` API

```python
from src.api import train_model

result = train_model(
    source,               # path to model.yaml OR a previous TrainResult
    stages,               # list[int] — e.g. [1, 2, 3, 4]
    name,                 # str — names the log folder under data/runs/
    *,
    config="configs/config.yaml",
    model_ids=None,       # optional list — train only these model ids
    seed=None,            # override global seed for this call
    vecenv="auto",        # 'auto' | 'subproc' | 'dummy'
    resume=True,          # auto-resume from last checkpoint on restart
)

# Access results by model id
result.ppo_baseline.final_model    # SB3 model object
result["ppo_baseline"].stages      # {1: StageOutcome, 2: StageOutcome, ...}
result.ppo_baseline.status         # 'completed' | 'capped' | 'stuck' | ...
result.ppo_baseline.save("path/")  # save to disk
```

---

## 🔬 Physical parameters

| Parameter | Value | Source |
|---|---|---|
| Mass | **40.85 g** (0.04085 kg) | `constants.py:PH_M` (NanoBench flying mass) |
| Inertia Ixx/Iyy | 2.3951 × 10⁻⁵ kg·m² | `constants.py:PH_J` (Forster 2015) |
| Inertia Izz | 3.2347 × 10⁻⁵ kg·m² | `constants.py:PH_J` (Forster 2015) |
| Rotor + propeller inertia | 2.0 × 10⁻⁹ kg·m² | `constants.py:PH_J_MP` |
| Arm length | 39.7 mm | `constants.py:PH_ARM` |
| Physics step `PH_DT` | 0.01 s (100 Hz) | `constants.py` |
| Hover command `PH_C_HOVER` | 0.23253743635354834 | Root of the ESC polynomial |
| Hover speed `PH_OMEGA_HOVER` | 1956.211093185006 rad/s | √(m·g / 4·c_L) |
| Lift coefficient `PH_C_L` | 2.618 × 10⁻⁸ N·(rad/s)⁻² | NanoBench hover sysid |
| Drag coefficient `PH_C_D` | 5.45 × 10⁻¹¹ N·m·(rad/s)⁻² | c_L / 480 |
| Motor lag `PH_K_MOT` | 20 ms | `constants.py` |
| Max motor speed `PH_OMEGA_MAX` | 2800 rad/s | `constants.py` |
| Rate PID gains | Kp `[0.15, 0.15, 0.20]`, Ki `[0.2, 0.2, 0.1]`, Kd `[0.003, 0.003, 0]` | `constants.py` |

> ⚠️ **Known limitation (BUGS.md item 13):** The fitted ESC polynomial is monotonically
> **decreasing** — `Ω_ss(0.02) = 2400.75`, `Ω_ss(0.2325) = 1956.21`,
> `Ω_ss(1.0) = 472.22` rad/s — so increasing throttle *reduces* motor speed. The
> simulator is valid for RL training (the policy finds the equilibrium at
> `cmd ≈ 0.2325`) but will not transfer to a real drone without re-identification.
> The coefficients are deliberately left untouched because `PH_C_HOVER` depends on
> them and the Stage-1 parity contract with `hover_env.py` is bit-exact.
> Quantify it any time with `python scripts/esc_diagnostic.py`.

---

## 🧪 Smoke test

Runs 12 tests with no GPU/torch required (numpy + scipy + gymnasium + pyyaml only):

```bash
cd interceptor-training
pip install numpy scipy gymnasium pyyaml
python scripts/smoke_test.py
# Expected: 12 passed, 0 failed
```

Tests cover: Stage-1 physics parity · all 8 env obs dims · history+future stacking ·
polynomial target paths · predictors · reward terms · curriculum advance/cap/rollback ·
result objects + eligibility · checkpoint layout · weight transfer · config/model
loading + hashing · future_source true vs pred equivalence.

The suite runs on plain numpy/scipy/gymnasium/pyyaml — no torch, no SB3, no GPU — so it
is also the CI gate for a fresh clone.

---

## 🐛 Known bugs

See [`BUGS.md`](BUGS.md) — 19 items documented; items 8, 11–12 and 14–18 fixed
in-tree, the remainder non-blocking or known limitations.

---

## 📄 License

No license — all rights reserved.
