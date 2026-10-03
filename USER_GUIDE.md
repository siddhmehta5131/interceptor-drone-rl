# Complete User Guide — Three Files to Run Training

You only ever touch three files to run the system:

| File | What it does |
|---|---|
| `configs/model.yaml` | Define your model(s): algorithm, network shape, critic fields |
| `configs/config.yaml` | Training settings: rollback, observations, target altitude, stage tweaks |
| `run.py` | Call `train_model()` — one Python file you edit and run |

---

## FILE 1: `configs/model.yaml`

Every top-level key is a **model_id**. You can have as many as you want. The model_id must be a valid Python identifier (`ppo_baseline` ✅, `2nd_run` ❌).

```yaml
ppo_baseline:          # ← your model_id; used as result.ppo_baseline in run.py
  algo: PPO            # which RL algorithm
  policy: MlpPolicy    # always MlpPolicy (only supported option)
  net_arch: ...        # network shape
  activation: ReLU     # activation function
  obs_history: ...     # how many past frames the actor sees
  hyperparameters: ... # SB3 training knobs
  privileged_critic:   # extra inputs only the critic sees (optional)
    - ...
```

---

### `algo`
Which RL algorithm to use.

| Value | Notes |
|---|---|
| `PPO` | On-policy. Recommended for curriculum training. Faster on CPU. |
| `SAC` | Off-policy. Saves/loads replay buffer. Requires more RAM. |
| `TD3` | Off-policy. Like SAC, deterministic actor. Saves/loads replay buffer. |

---

### `policy`
```yaml
policy: MlpPolicy   # only valid value
```

---

### `net_arch`
The layer sizes for the actor (`pi`) and critic (`vf`/`qf`) networks.

```yaml
# Explicit (recommended):
net_arch: {pi: [256, 256, 128], vf: [256, 256, 128]}

# Shorthand — same for actor and critic:
net_arch: [256, 256, 128]

# For SAC/TD3, use 'qf' instead of 'vf':
net_arch: {pi: [256, 256], qf: [256, 256]}
```

Numbers = hidden units per layer. More layers / larger layers = slower but more expressive.

---

### `activation`
Activation function between layers.

| Value | Notes |
|---|---|
| `ReLU` | Default, fastest, works well |
| `Tanh` | Bounded, sometimes more stable |
| `ELU` | Smoother than ReLU, avoids dead neurons |

---

### `obs_history`
How many past frames the actor receives. More frames = agent sees recent motion history.

```yaml
obs_history:
  frames: 3    # m — how many past frames (beyond the current one)
  skip: 2      # p — gap between frames in env steps (1 env step = 0.01 s)
```

**What the actor sees:**  
`[current_frame, frame at t-2, frame at t-4, frame at t-6]` (4 total, 3 historical)

| Setting | Total frames | Time span |
|---|---|---|
| `frames: 3, skip: 2` | 4 | 0.06 s of history |
| `frames: 5, skip: 3` | 6 | 0.15 s of history |
| `frames: 0, skip: 1` | 1 | no history |

---

### `hyperparameters`
Passed directly to StableBaselines3. These are the main ones:

| Key | Default | What it does |
|---|---|---|
| `n_steps` | `4096` | Steps collected before each gradient update (PPO only). Larger = more stable but slower. |
| `batch_size` | `512` | Minibatch size for gradient updates. |
| `gamma` | `0.995` | Discount factor. Near 1 = cares about distant future. |
| `learning_rate` | `3.0e-4` | Step size for the Adam optimizer. Reduce if training is unstable. |
| `gae_lambda` | `0.95` | GAE advantage estimation (PPO). Higher = lower variance. |
| `clip_range` | `0.2` | PPO clipping. Controls how large a policy update can be. |
| `ent_coef` | `0.005` | Entropy bonus — encourages exploration. Increase if policy collapses. |
| `n_epochs` | `10` | How many passes over the collected data per update (PPO). |
| `max_grad_norm` | `0.5` | Gradient clipping. Prevents exploding gradients. |
| `vf_coef` | `0.5` | Weight of value function loss vs policy loss (PPO). |
| `action_noise_sigma` | — | TD3 only: standard deviation of exploration noise. |

---

### `privileged_critic`
Extra inputs the **critic** receives that the **actor never sees**. This lets the critic evaluate state quality using information that won't be available on a real drone.

```yaml
privileged_critic:
  - time_remaining      # (1 number) fraction of episode time left, in [0, 1]
  - facing_error        # (1 number) yaw error to target, in radians
  - target_true_pos     # (3 numbers) ideal target position minus observed, body frame
  - target_true_vel     # (3 numbers) ideal target velocity minus observed, body frame
```

List any subset, or omit the key entirely for a standard symmetric policy.

---

### Multiple models in one file

```yaml
ppo_baseline:
  algo: PPO
  net_arch: {pi: [256, 256, 128], vf: [256, 256, 128]}
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

ppo_deep:
  algo: PPO
  net_arch: {pi: [256, 256, 256, 128], vf: [256, 256, 256, 128]}
  obs_history: {frames: 5, skip: 3}
  hyperparameters:
    n_steps: 4096
    batch_size: 512
    gamma: 0.995
    learning_rate: 1.0e-4   # lower LR for deeper net
    gae_lambda: 0.95
    clip_range: 0.15
    ent_coef: 0.005
    n_epochs: 10
    max_grad_norm: 0.5
    vf_coef: 0.5

sac_model:
  algo: SAC
  net_arch: {pi: [256, 256], qf: [256, 256]}
  obs_history: {frames: 3, skip: 2}
  hyperparameters:
    batch_size: 512
    gamma: 0.99
    learning_rate: 3.0e-4
    buffer_size: 1000000
    learning_starts: 10000
    ent_coef: auto
```

When you call `train_model(source="configs/model.yaml", ...)`, **all models** in the file are trained. Pass `model_ids=["ppo_baseline"]` to train only one.

---

---

## FILE 2: `configs/config.yaml`

Training settings — everything that is **not** the model architecture.

---

### `global` section

```yaml
global:
  seed: 42                     # random seed for reproducibility
  device: cpu                  # 'cpu', 'cuda', or 'auto'
  data_dir: /data              # where checkpoints, logs, results are written
  n_parallel_envs: 8           # number of vectorised environments
  checkpoint_interval_steps: 50000   # save a checkpoint every N steps
  tensorboard: true            # write TensorBoard logs
  execution_mode: sequential   # only 'sequential' is supported
```

| Key | Options | Notes |
|---|---|---|
| `seed` | any integer | Set to the same value for reproducible runs |
| `device` | `cpu`, `cuda`, `auto` | PPO on CPU is often faster than GPU due to small batches |
| `data_dir` | any path | Inside Docker this is `/data` (mounted volume) |
| `n_parallel_envs` | 1–32 | More envs = faster data collection; needs more RAM |
| `checkpoint_interval_steps` | 10000–100000 | Smaller = more checkpoints, more disk use |
| `tensorboard` | `true`/`false` | Set false to save disk space |

---

### `rollback` section

When a model's success rate falls too low mid-training, the curriculum rolls it back to the previous stage.

```yaml
rollback:
  max_attempts: 2         # max times to roll back per stage before giving up
  retry_budget_scale: 0.5 # re-entered stage runs on this fraction of the full budget
  threshold_scale: 0.5    # rollback triggers when success_rate < required_rate * this
  min_steps_scale: 0.25   # rollback only checked after this fraction of max_steps
```

**Example:** Stage 3 requires 70% success rate, budget 500k steps.
- Rollback triggers if success rate < 35% (70% × 0.5)
- Only checked after 125k steps (500k × 0.25)
- On rollback → go back to stage 2, retry stage 3 with 250k steps (500k × 0.5)
- If it fails again → mark as stuck, stop that model

---

### `on_capped`

What happens when a stage hits its step budget without reaching the required success rate.

```yaml
on_capped: continue   # or: stop
```

| Value | Behaviour |
|---|---|
| `continue` | Train the next stage anyway (default). Model may learn sub-optimally but keeps going. |
| `stop` | Stop training that model. Status = `capped`. Other models in the same call keep training. |

---

### `observation` section

How the actor observation is built.

```yaml
observation:
  history_frames: 3     # m: how many past frames
  history_skip: 2       # p: steps between past frames
  future_samples: 3     # n: how many future target position samples
  future_skip: 5        # q: steps between future samples
  future_source: true   # 'true' or 'pred'
  predictor: const_vel  # 'const_vel' or 'linear_ridge'
```

**Actor observation size** (stages 2+):
```
19 × (1 + m) + 7 × n + privileged_dim
= 19 × 4 + 7 × 3 + 8  = 105   (ppo_baseline defaults)
```

| Key | What it controls |
|---|---|
| `history_frames` / `history_skip` | How far back the actor looks. Same as `obs_history` in model.yaml — these must be consistent (model.yaml wins). |
| `future_samples` | How many future target positions the actor sees. 0 = no lookahead. |
| `future_skip` | Gap in env steps between future samples. 5 steps = 0.05 s between samples. |
| `future_source: true` | Future positions come from the simulator ground truth (for training). |
| `future_source: pred` | Future positions come from the predictor (for deployment simulation). |
| `predictor: const_vel` | Assume target keeps moving at its last observed velocity. |
| `predictor: linear_ridge` | Fit a linear model (ridge regression) to measurement history. More accurate on curved paths. |

---

### `target_alt`

The altitude the drone hovers at in Stage 1.

```yaml
target_alt: 5.0             # fixed at 5 metres
# or:
target_alt: {min: 3.0, max: 8.0}   # random between 3 and 8 at each episode reset
```

> ⚠️ Using a fixed value `5.0` preserves exact parity with the reference `hover_env.py`. Changing it to a range randomises Stage 1 difficulty.

---

### `stages` section

Override the built-in stage parameters for specific stages.

```yaml
stages: {}   # empty = use all defaults

# Override examples:
stages:
  stage_1:
    episode_length_s: 10.0          # fixed 10-second episodes
  stage_2:
    episode_length_s: {min: 10.0, max: 20.0}   # random length per episode
    reward:
      k_facing: 0.20                # stronger facing reward than default 0.15
  stage_6:
    spawn:
      path_end_distance: {min: 5.0, max: 20.0}   # how far the target travels
      path_speed_cap: {min: 2.0, max: 6.0}        # max target speed (m/s)
      path_shape: {min: 0.0, max: 1.0}            # polynomial curve shape [0=linear, 1=max curve]
```

**Available reward overrides per stage:**

| Key | Default | What it does |
|---|---|---|
| `k_alive` | 0.10 (S1), 0.05 (S2), 0.0 (S3+) | Reward just for surviving each step |
| `k_alt` | 0.15 (S1), 0.0 (S2+) | Penalty for altitude error (Stage 1 only) |
| `k_tilt` | 0.40 (S1), 0.20 (S2), 0.0 (S3+) | Penalty for tilting |
| `k_angvel` | 0.02 (all) | Penalty for high angular velocity |
| `k_smooth` | 0.05 (all) | Penalty for jerky control changes |
| `k_velocity_alignment` | 0.30 (S2+) | Reward for flying toward the target |
| `k_progress_delta` | 0.15 (S2+) | Reward for getting closer each step |
| `k_time_penalty` | 0.02 (S4+) | Flat penalty per step — encourages speed |
| `k_facing` | 0.15 (S2+) | Penalty for not facing the target |
| `k_kill_bonus` | 500.0 (S3+) | Bonus on successful intercept |

---

---

## FILE 3: `run.py`

The file you actually run. Edit it or call it from the command line.

---

### The `train_model()` function

```python
from src.api import train_model

result = train_model(
    source,               # REQUIRED — see below
    stages,               # REQUIRED — list of stage numbers, e.g. [1, 2, 3, 4]
    name,                 # REQUIRED — name for this run's folder
    *,
    config="configs/config.yaml",    # path to your config.yaml
    model_ids=None,       # list of model ids to train, or None = all
    seed=None,            # override global.seed for this call only
    resume=True,          # auto-resume from last checkpoint if run was interrupted
)
```

**`source`** — three forms:

| Form | When to use |
|---|---|
| `"configs/model.yaml"` | Fresh run from scratch |
| A previous `TrainResult` object | Continue training from where a prior call left off |
| Path to a saved `run_summary.json` | Resume from a saved result on disk |

---

### The `TrainResult` object

`train_model()` returns a `TrainResult`. Access results by model id:

```python
result = train_model(source="configs/model.yaml", stages=[1,2,3,4], name="run1")

# Access by attribute:
r = result.ppo_baseline

# Access by key:
r = result["ppo_baseline"]

# List all model ids trained:
print(result.models)        # {'ppo_baseline': ModelResult, ...}

# Overall info:
print(result.name)          # 'run1'
print(result.stages)        # [1, 2, 3, 4]
print(result.config_hash)   # 64-char SHA-256
print(result.seconds)       # wall time in seconds
print(result.run_dir)       # path to artifacts folder
print(result.completed)     # True if all models finished without error
```

---

### The `ModelResult` object

```python
r = result.ppo_baseline

r.status          # 'completed' | 'capped' | 'stuck' | 'skipped' | 'crashed' | 'denied'
r.reason          # plain-English explanation if status != 'completed'
r.config_hash     # 64-char hash of the config that produced this model
r.stage_outcomes  # list of StageOutcome objects
r.final_model     # the SB3 model object itself (None if not completed)
```

**`StageOutcome`:**
```python
o = r.stage_outcomes[0]
o.stage           # 1
o.result          # 'advance' | 'capped' | 'stuck'
o.steps           # steps taken in this stage
o.success_rate    # final success rate when stage ended
```

**Save / load:**
```python
# Save model to disk:
path = r.save("my_models/")
# Creates: my_models/ppo_baseline.zip  +  my_models/ppo_baseline.result.json

# Load back:
from src.results import ModelResult
r2 = ModelResult.load("my_models/ppo_baseline.result.json")
model = r2.load_model()   # returns the SB3 model object
```

---

### Usage patterns

#### Pattern 1 — Simple fresh run

```python
from src.api import train_model
result = train_model("configs/model.yaml", [1, 2, 3, 4], "my_run")
```

#### Pattern 2 — Two-phase (recommended)

```python
from src.api import train_model

# Phase 1: easy stages
model1 = train_model("configs/model.yaml", [1, 2, 3, 4], "phase1")

# Phase 2: continue from phase 1
model2 = train_model(model1, [5, 6, 7], "phase2")

# Save
model2.ppo_baseline.save("final_model/")
```

#### Pattern 3 — Train only specific models

```python
result = train_model(
    "configs/model.yaml", [1, 2, 3], "ppo_only",
    model_ids=["ppo_baseline"]
)
```

#### Pattern 4 — Override seed

```python
result = train_model(
    "configs/model.yaml", [1, 2], "seed_test",
    seed=99
)
```

#### Pattern 5 — Command line

```bash
# Fresh run, stages 1-4
python run.py --stages 1,2,3,4 --name my_run

# Continue from a saved result
python run.py --continue-from data/runs/my_run/results/run_summary.json \
              --stages 5,6,7 --name my_run_continued

# Train only one model from the file
python run.py --stages 1,2 --name test --models ppo_baseline

# Override seed
python run.py --stages 1,2 --name seed99 --seed 99

# Save final models to a folder
python run.py --stages 1,2,3,4 --name full --save saved_models/
```

---

### `format_summary()` — print a readable summary

Already called automatically in `run.py`. Can be called manually:

```python
from run import format_summary
print(format_summary(result))
```

Output example:
```
run 'my_run'  stages=[1, 2, 3, 4]  hash=a3f9d2bc1e7f  142.3s
  artifacts: /data/runs/my_run
  ppo_baseline [completed] 1:advance(0.95/50000 steps), 2:advance(0.87/120000 steps), ...
  ppo_deep [capped] 1:advance(0.91/50000 steps), 2:capped(0.61/200000 steps)
```

---

---

## Quick reference: status values

| Status | Meaning |
|---|---|
| `completed` | All requested stages finished, success rate reached |
| `capped` | A stage hit its step budget; `on_capped: stop` was set, or continuation was denied |
| `stuck` | Rollback attempts exhausted — model cannot learn this stage |
| `skipped` | Source model's last stage doesn't connect to the requested stages |
| `crashed` | Python exception during training; other models kept running |
| `denied` | Source model was previously `capped` — continuation blocked |

---

## Range syntax

Any field marked as a number can also be a min/max range — drawn randomly at each episode reset:

```yaml
target_alt: 5.0                    # fixed
target_alt: {min: 3.0, max: 8.0}   # random each episode

episode_length_s: 20.0
episode_length_s: {min: 15.0, max: 30.0}

path_end_distance: {min: 5.0, max: 20.0}
```
