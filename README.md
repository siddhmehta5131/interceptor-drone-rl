# RL-Based Autonomous Interceptor Drone

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0-EE4C2C?logo=pytorch)
![StableBaselines3](https://img.shields.io/badge/SB3-2.x-green)
![Docker](https://img.shields.io/badge/Docker-GPU-2496ED?logo=docker)

A **reinforcement-learning training pipeline** for an autonomous interceptor drone. A Crazyflie 2.0-calibrated rigid-body physics simulator is combined with an 8-stage curriculum that teaches a quadrotor to hover, track, and intercept a manoeuvring target.

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
  --source configs/model.yaml --stages 1 2 3 4 --name experiment_1
```

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
│   └── smoke_test.py           ← 12-test no-SB3 verification suite
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
| Mass | **40.85 g** (0.04085 kg) | `constants.py:PH_M` |
| Inertia Ixx/Iyy | 1.4 × 10⁻⁵ kg·m² | Crazyflie 2.0 published |
| Inertia Izz | 2.17 × 10⁻⁵ kg·m² | Crazyflie 2.0 published |
| Arm length | 39.7 mm | Crazyflie 2.0 published |
| Physics step `PH_DT` | 0.01 s | `constants.py` |
| Hover command `PH_C_HOVER` | 0.23253743635354834 | Computed from ESC polynomial |
| Lift coefficient | 5.0 × 10⁻⁸ N·(rad/s)⁻² | Tuned |
| Drag coefficient | 1.25 × 10⁻⁹ N·m·(rad/s)⁻² | Forster ratio |
| Motor time constant | 20 ms | Best estimate |

> ⚠️ **Known limitation (Bug #13):** The fitted ESC polynomial is monotonically
> decreasing — increasing throttle reduces motor speed. The simulator is valid for
> RL training but policies will not transfer to a real drone without re-identification.
> See `BUGS.md` for details.

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

---

## 🐛 Known bugs

See [`BUGS.md`](BUGS.md) — 13 items documented; items 11–12 fixed, remainder non-blocking.

---

## 📄 License

No license — all rights reserved.
