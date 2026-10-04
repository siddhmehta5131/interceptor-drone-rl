# Implementation Plan: D-2 to D-58

**Date:** 2026-10-03  
**Scope:** All 57 design decisions from `context_v5_0.md` Decision Log  
**Target:** `interceptor-training/` codebase  
**Final stage:** Clean sync to `github_repos/rl-drone-flight-simulator`, push, Docker build

---

## Architecture Overview: Before → After

### BEFORE (current)
```
train.py --config default_run.yaml
         ↓
    load_run_config(yaml)  →  RunConfig
         ↓
    Orchestrator(cfg).run()  →  summary dict
```
Single YAML has `global`, `stages`, `configs` sections. One entry point. One monolith.

### AFTER (target)
```
run.py                          ← user's script (one-liner)
  │
  ├── model.yaml                ← model definitions (algo, net_arch, critic fields...)
  ├── config.yaml               ← training settings (rollback, on_capped, n/m/p/q, future_source...)
  │
  └── train_model(source, stages, name, ...)  →  TrainResult
         │
         ├── loads model.yaml + config.yaml
         ├── per model_id: build env, build model, run curriculum
         └── returns TrainResult[model_id] with final models
```
Three artifacts. `train_model()` is the public API. Models are first-class objects.

---

## Phase A — Config System Split

> **Decisions:** D-9, D-20, D-44, D-48, D-49, D-2

### A-1. Create `interceptor-training/configs/model.yaml`

This replaces the `configs:` section of `default_run.yaml`.

```yaml
# model.yaml — model definitions (user-edited)
# Each top-level key is a model_id (must be a valid Python identifier).

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
    gae_lambda: 0.95
    clip_range: 0.2
    ent_coef: 0.005
    n_epochs: 10
    max_grad_norm: 0.5
    vf_coef: 0.5
  privileged_critic:         # D-11: list of critic-only inputs
    - time_remaining          # D-15: only critic sees time
    - facing_error            # D-57: horizontal facing error to target
    - target_true_pos         # D-12: ideal reference (error = ideal - current)
    - target_true_vel
```

**Files touched:** NEW `interceptor-training/configs/model.yaml`  
**Decision trace:** D-9 (three-artifact design), D-11 (critic inputs in model file), D-44 (name "model" kept)

### A-2. Create `interceptor-training/configs/config.yaml`

This replaces the `global:` + `stages:` sections of `default_run.yaml`.

```yaml
# config.yaml — training settings (user-edited)

global:
  seed: 42
  device: cpu
  data_dir: /data
  n_parallel_envs: 8
  checkpoint_interval_steps: 50000
  tensorboard: true
  execution_mode: sequential

rollback:
  max_attempts: 2
  retry_budget_scale: 0.5
  threshold_scale: 0.5
  min_steps_scale: 0.25

on_capped: continue           # D-25, D-30: 'continue' or 'stop'

# Observation history and future (D-18, D-38)
observation:
  history_frames: 3            # m past frames
  history_skip: 2              # p spacing
  future_samples: 3            # n future samples
  future_skip: 5               # q spacing (in env steps)
  future_source: true          # D-16: 'true' (ground truth) or 'pred'
  predictor: const_vel         # D-19: 'const_vel' or 'linear_ridge'

# D-3, D-50: Stage 1 hover altitude
target_alt: 5.0                # fixed value or {min: 3.0, max: 7.0}

# Per-stage overrides (merged onto built-in STAGES table)
stages: {}
```

**Files touched:** NEW `interceptor-training/configs/config.yaml`  
**Decision trace:** D-9, D-20, D-25, D-30, D-38, D-3, D-50, D-48 (range syntax `{min:, max:}`)

### A-3. Create `interceptor-training/configs/smoke_model.yaml` and `smoke_config.yaml`

Per D-7, D-8, D-49: smoke test gets its own files.

`smoke_model.yaml`:
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
    gae_lambda: 0.95
    clip_range: 0.2
    ent_coef: 0.005
    n_epochs: 10
    max_grad_norm: 0.5
    vf_coef: 0.5

ppo_5layer_deep:
  algo: PPO
  policy: MlpPolicy
  net_arch: {pi: [256, 256, 256, 256, 128], vf: [256, 256, 256, 256, 128]}
  activation: ReLU
  obs_history: {frames: 5, skip: 3}
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
```

`smoke_config.yaml`: same as `config.yaml` but with `future_source: pred` (D-42 for Stage 8).

**Files touched:** NEW `interceptor-training/configs/smoke_model.yaml`, `smoke_config.yaml`  
**Decision trace:** D-7, D-8, D-42, D-49

### A-4. Refactor `config_loader.py` — split into `model_loader.py` + `config_loader.py`

**Current:** `config_loader.py` loads one YAML, returns `RunConfig`.  
**After:** Two loaders:

1. `model_loader.py` — `load_models(path) → Dict[str, ModelDef]`
   - Validates each model_id is a valid Python identifier (D-34)
   - Validates algo, policy, net_arch, activation, obs_history, hyperparameters
   - Validates privileged_critic list (D-11)
   - Returns a dict of `ModelDef` dataclasses

2. `config_loader.py` — `load_config(path) → TrainConfig`
   - Validates global, rollback, on_capped (D-30)
   - Validates observation section (n, m, p, q) (D-18, D-38)
   - Validates future_source, predictor (D-16, D-19)
   - Validates target_alt and range syntax {min:, max:} (D-3, D-48)
   - Validates per-stage overrides
   - Computes `config_hash` spanning BOTH files (D-20: nothing lost)

**New dataclasses:**
```python
@dataclass
class ModelDef:
    model_id: str
    algo: str                  # PPO | SAC | TD3
    policy: str                # MlpPolicy
    net_arch: Dict[str, List[int]]
    activation: str
    obs_history: Dict[str, int]  # {frames, skip}
    hyperparameters: Dict[str, Any]
    privileged_critic: List[str]  # D-11

@dataclass
class TrainConfig:
    global_cfg: Dict[str, Any]
    rollback: Dict[str, Any]
    on_capped: str             # D-30: 'continue' | 'stop'
    observation: ObservationConfig  # D-18, D-38
    target_alt: Any            # float or {min, max} — D-3, D-48
    stage_overrides: Dict[str, Dict]
    config_hash: str
    source_model_path: str
    source_config_path: str
```

**Files touched:** MODIFY `interceptor-training/src/utils/config_loader.py`, NEW `interceptor-training/src/utils/model_loader.py`  
**Decision trace:** D-9, D-20, D-34, D-48

### A-5. Retire `default_run.yaml`

- Delete `interceptor-training/configs/default_run.yaml`
- Update `Dockerfile` CMD to `CMD ["--help"]` (no default config path)
- Update `docker-compose.yml` if needed
- The Docker ENTRYPOINT still calls `train.py` but `train.py` is refactored (Phase B)

**Files touched:** DELETE `default_run.yaml`, MODIFY `Dockerfile`, MODIFY `docker-compose.yml`  
**Decision trace:** D-20

### A-6. Utility: range parser (D-48)

Add to `config_loader.py`:
```python
def parse_range(value, name: str) -> tuple:
    """Parse a fixed value or {min:, max:} range."""
    if isinstance(value, (int, float)):
        return (value, value)  # fixed
    if isinstance(value, dict) and 'min' in value and 'max' in value:
        lo, hi = float(value['min']), float(value['max'])
        assert lo <= hi, f"{name}: min ({lo}) > max ({hi})"
        return (lo, hi)
    raise ConfigError(f"{name}: expected number or {{min:, max:}}, got {value}")
```

**Files touched:** MODIFY `interceptor-training/src/utils/config_loader.py`  
**Decision trace:** D-48

---

## Phase B — `train_model()` Public API

> **Decisions:** D-10, D-21, D-22, D-23, D-26, D-29, D-34, D-36

### B-1. Create `interceptor-training/src/api.py` — the `train_model()` function

```python
def train_model(
    source: Union[str, Path, TrainResult],   # model file path OR previous result
    stages: List[int],                        # D-21: stage list as argument
    name: str,                                # D-22: log folder name
    *,
    config: Union[str, Path] = "configs/config.yaml",  # config file path
    model_ids: Optional[List[str]] = None,    # D-26: optional subset
    seed: Optional[int] = None,               # D-36: optional seed override
) -> TrainResult:
    """Train one or more models through a curriculum of stages.
    
    Returns a TrainResult containing the final model of the last stage
    for each model_id trained.
    """
```

**Behavior:**

1. If `source` is a path string → load model definitions from YAML (A-4)
2. If `source` is a `TrainResult` → extract model definitions + weights from prior run (D-26)
3. If `model_ids` given → train only those; skip others silently
4. For each model_id:
   - Check if source model is capped → deny with plain message (D-56)
   - Check if source model's last stage fits the requested stage list → skip if not (D-27)
   - Load prior weights if continuing (D-28: reuse saved settings, no overrides)
   - Run curriculum for the requested stages
   - If crash → report in result, continue with other models (D-31)
   - If STUCK → stop that model only, continue others (D-55)
5. Save last stage's final model to disk in `<name>/<model_id>/` folder (D-29)
6. Return `TrainResult` keyed by model_id (D-26)

### B-2. Create `TrainResult` class

```python
class TrainResult:
    """Result of a train_model() call, keyed by model_id."""
    
    def __getattr__(self, model_id: str) -> ModelResult:
        """Access as result.ppo_baseline"""
        
    def __getitem__(self, model_id: str) -> ModelResult:
        """Access as result['ppo_baseline']"""

class ModelResult:
    """Result for one model_id."""
    model_id: str
    algo: str
    stages: Dict[int, StageOutcome]     # D-32: each stage's outcome
    final_model: Any                     # SB3 model object (saveable)
    config_hash: str                     # D-32
    status: str                          # 'completed' | 'capped' | 'stuck' | 'skipped' | 'crashed' | 'denied'
    reason: Optional[str]               # D-27, D-31, D-56: plain-language reason
    
    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk (D-23)."""
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ModelResult':
        """Load model from disk (D-23)."""
```

**Files touched:** NEW `interceptor-training/src/api.py`, NEW `interceptor-training/src/results.py`  
**Decision trace:** D-10, D-21, D-22, D-23, D-26, D-29, D-32, D-34, D-36

### B-3. Create `interceptor-training/run.py` — the one-file runner

```python
#!/usr/bin/env python3
"""run.py — one-file training runner.

Usage:
    python run.py
    
Edit model.yaml and config.yaml to configure training.
"""
from src.api import train_model

# Train ppo_baseline through stages 1-4
model1 = train_model(
    source="configs/model.yaml",
    stages=[1, 2, 3, 4],
    name="experiment_1",
    config="configs/config.yaml",
)

# Continue training on stages 5-7 from model1
model2 = train_model(
    source=model1,
    stages=[5, 6, 7],
    name="experiment_2",
)

# Save final model
model2.ppo_baseline.save("my_model/")
```

**Files touched:** NEW `interceptor-training/run.py`  
**Decision trace:** D-9, D-22

### B-4. Refactor `train.py` to use `train_model()`

Keep `train.py` as the Docker ENTRYPOINT but refactor it to call `train_model()` internally. Keep CLI args for smoke mode.

**Files touched:** MODIFY `interceptor-training/scripts/train.py`  
**Decision trace:** D-9

### B-5. Refactor `orchestrator.py` — internal engine for `train_model()`

The Orchestrator becomes an internal implementation detail called by `train_model()`. Key changes:
- Accept stages as argument (not from config) — D-21
- Accept name for log folder — D-22
- Accept seed override — D-36
- Return `TrainResult` instead of summary dict
- Per-model error isolation (D-31, D-55)

**Files touched:** MODIFY `interceptor-training/src/training/orchestrator.py`  
**Decision trace:** D-10, D-21, D-22, D-31, D-36, D-55

---

## Phase C — Training Loop Refactor

> **Decisions:** D-24, D-25, D-27, D-28, D-30, D-31, D-32, D-33, D-35, D-45, D-46, D-47, D-55, D-56

### C-1. Capped-stage handling (D-24, D-25, D-30)

**Current:** Capped stage always continues to next stage.  
**After:** Read `on_capped` from config:
- `continue` → proceed (current behavior, D-24: all snapshots saved)
- `stop` → stop training that model, mark last stage as `capped`, return result

**In `orchestrator.py`:** After `CurriculumScheduler` returns `CAPPED`:
```python
if self.config.on_capped == 'stop':
    return StageOutcome.CAPPED  # stop this model
else:
    # save final, advance to next stage (current behavior)
```

**Files touched:** MODIFY `orchestrator.py`, MODIFY `curriculum.py`  
**Decision trace:** D-24, D-25, D-30

### C-2. Skip logic (D-27, D-35, D-56)

Before training each model:
1. If source is a `TrainResult` with a capped model → **deny** with plain message (D-56)
2. If source model's last completed stage does not fit the first requested stage → **skip** with reason (D-27)
3. Record skipped/denied models in result with `reason` string

```python
def _check_model_eligibility(model_def, source_result, stages):
    if source_result and source_result.status == 'capped':
        return 'denied', f"Model '{model_def.model_id}' has a capped stage; retrain with a new model id"
    if source_result and source_result.last_stage not in valid_predecessors(stages[0]):
        return 'skipped', f"Model '{model_def.model_id}' last completed stage {source_result.last_stage} does not fit requested stages {stages}"
    return 'eligible', None
```

**Files touched:** MODIFY `orchestrator.py`  
**Decision trace:** D-27, D-35, D-56

### C-3. Continuation settings (D-28)

When continuing from a prior result, the training settings (algo, hyperparameters, net_arch) are loaded from the saved model's metadata, NOT from the model file. No settings can be overridden.

**Implementation:** Save a `_training_meta.json` alongside each model checkpoint:
```json
{
  "model_id": "ppo_baseline",
  "algo": "PPO",
  "hyperparameters": {...},
  "net_arch": {...},
  "config_hash": "..."
}
```

On continuation, load this and verify it matches. If mismatch → error.

**Files touched:** MODIFY `checkpoint_manager.py`, MODIFY `orchestrator.py`  
**Decision trace:** D-28

### C-4. Weight transfer across observation size change (D-33)

When moving from Stage 1 (obs=14) to Stage 2 (obs=76):
- Extract weights from the old network
- Create new network with the Stage 2 observation size
- Copy weights for matching dimensions (input layer: first 12 columns match)
- Initialize new columns (13–75) with small random values
- Copy all other layers as-is (hidden and output layers match exactly)

```python
def _transfer_weights(old_model, new_obs_dim, old_obs_dim):
    old_params = old_model.policy.state_dict()
    # ... copy matching weights, init new with small random
    new_model.policy.load_state_dict(new_params, strict=False)
```

**Files touched:** NEW `interceptor-training/src/training/weight_transfer.py`, MODIFY `orchestrator.py`  
**Decision trace:** D-33

### C-5. Crash isolation (D-31, D-55)

Wrap each model's training in a try/except:
```python
for model_id in model_ids_to_train:
    try:
        result = self._train_single_model(model_id, stages)
        results[model_id] = result
    except Exception as e:
        results[model_id] = ModelResult(
            model_id=model_id,
            status='crashed',
            reason=f"Training crashed: {e}",
        )
        logger.error(f"Model {model_id} crashed: {e}", exc_info=True)
```

STUCK also stops only that model (D-55):
```python
if outcome == 'stuck':
    results[model_id] = ModelResult(status='stuck', reason=...)
    break  # break the stage loop, not the model loop
```

**Files touched:** MODIFY `orchestrator.py`  
**Decision trace:** D-31, D-55

### C-6. Rollback counts persist across calls (D-45)

Save rollback retry counts in the resume state JSON. When a new `train_model()` call loads a prior result, carry over the retry counts.

**Files touched:** MODIFY `checkpoint_manager.py`, MODIFY `orchestrator.py`  
**Decision trace:** D-45

### C-7. Mid-stage resume (D-46)

If the program dies mid-stage, the next `train_model()` call:
1. Finds the `run_state.json` for the given `name`
2. Validates `config_hash` match
3. Loads the latest periodic checkpoint for the current stage
4. Resumes training from there with remaining budget

**Files touched:** MODIFY `orchestrator.py`, MODIFY `checkpoint_manager.py`  
**Decision trace:** D-46

### C-8. SAC/TD3 replay buffer persistence (D-47)

For off-policy algorithms, save and reload the replay buffer on continuation:
```python
if algo in ('SAC', 'TD3'):
    model.save_replay_buffer(path + '_buffer.pkl')
    # On load:
    model.load_replay_buffer(path + '_buffer.pkl')
```

**Files touched:** MODIFY `checkpoint_manager.py`  
**Decision trace:** D-47

### C-9. Stage outcome in result (D-32)

Each stage records its outcome:
```python
@dataclass
class StageOutcome:
    stage: int
    result: str        # 'advance' | 'capped' | 'stuck'
    steps: int
    success_rate: float
```

These are collected in `ModelResult.stages`.

**Files touched:** MODIFY `orchestrator.py`, NEW `interceptor-training/src/results.py`  
**Decision trace:** D-32

---

## Phase D — Environment Changes

> **Decisions:** D-3, D-4, D-5, D-6, D-13, D-14, D-37, D-48, D-50, D-51, D-52

### D-1. Config-driven target altitude (D-3, D-50)

**Current:** `InterceptorBaseEnv._target_alt` is a property returning `5.0`.  
**After:** Read `target_alt` from config. If range `{min, max}`, draw at each reset.

```python
# In base_env.py __init__:
self._target_alt_range = parse_range(config.target_alt)  # (lo, hi)

# In reset():
self._current_target_alt = self._rng.uniform(*self._target_alt_range)
```

Stage 1 only (D-50). Stages 2+ target z comes from the target generator.

**Parity note:** If `target_alt == 5.0` (fixed), Stage 1 parity with `hover_env.py` is preserved. If randomized, the new draw must be added AFTER the existing Stage 1 draws (xy, tilt, axis) to preserve RNG ordering for backwards compatibility.

**Files touched:** MODIFY `base_env.py`, MODIFY `stage_config.py`  
**Decision trace:** D-3, D-50

### D-2. Fixed polynomial targets for Stages 5-7 (D-4, D-13, D-51)

**Current:** `target_generator.py` uses Euler integration with periodic re-sampling.  
**After for Stages 5-7:**

1. At episode reset, draw a polynomial of the stage's order (1, 2, or 3)
2. Compute the polynomial backwards from start point, end point, and episode length:
   - `pos(t) = start + (end - start) * p(t/T)` where `p` is a polynomial
   - For order 1: `p(τ) = τ` (linear)
   - For order 2: `p(τ) = aτ² + bτ` where coefficients ensure `p(0)=0, p(1)=1`
   - For order 3: `p(τ) = aτ³ + bτ² + cτ` similarly constrained
3. The path is independent of the drone (D-5)
4. No mid-episode re-draws (D-4)

**Free parameters (D-51):** Start point, end point, speed cap — each is fixed or `{min, max}` range, drawn at reset.

```python
class PolynomialTargetPath:
    """Fixed polynomial path computed at episode start."""
    
    def __init__(self, order, start, end, duration_s, rng):
        self.coeffs = self._compute_coefficients(order, start, end, duration_s)
    
    def position_at(self, t: float) -> np.ndarray:
        """Return target position at time t seconds."""
    
    def velocity_at(self, t: float) -> np.ndarray:
        """Return target velocity at time t seconds (derivative)."""
    
    def acceleration_at(self, t: float) -> np.ndarray:
        """Return target acceleration at time t (2nd derivative)."""
```

**Stage 8 (evasive):** Stays as-is but disabled in current scope (D-6). Only exercised by the smoke test under `pred` mode (D-8, D-42).

**Files touched:** MODIFY `target_generator.py`, MODIFY `stage_config.py`  
**Decision trace:** D-4, D-5, D-6, D-13, D-51

### D-3. Episode length in seconds (D-13, D-14, D-52)

**Current:** `max_episode_steps` is an integer count.  
**After:** Episode length is in seconds, converted to steps via `steps = round(T / PH_DT)`.

- Fixed value or `{min, max}` range (D-14)
- Stage 1: always fixed (D-52) — preserves reward-sum threshold of 60.0
- Other stages: drawn at each reset if range is given

```python
# In stage_config.py:
episode_length_s: float | dict  # seconds, or {min: ..., max: ...}

# In base_env.py reset():
if isinstance(self.stage.episode_length_s, dict):
    T = self._rng.uniform(self.stage.episode_length_s['min'],
                          self.stage.episode_length_s['max'])
else:
    T = self.stage.episode_length_s
self._max_episode_steps = round(T / PH_DT)
```

**Files touched:** MODIFY `stage_config.py`, MODIFY `base_env.py`  
**Decision trace:** D-13, D-14, D-52

### D-4. Yaw facing reference (D-37)

The drone should face the target — its nose aims at the target direction.

**Implementation:** Compute desired yaw from the line-of-sight vector:
```python
los = target_pos[:2] - drone_pos[:2]  # horizontal direction
desired_yaw = np.arctan2(los[1], los[0])
yaw_error = wrap_angle(desired_yaw - current_yaw)
```

This is used in:
1. The facing reward (D-57, Phase G)
2. The privileged critic reference (D-12)

**Files touched:** MODIFY `base_env.py` (yaw error computation), MODIFY `reward.py`  
**Decision trace:** D-37

---

## Phase E — Observation System

> **Decisions:** D-11, D-12, D-15, D-16, D-18, D-33, D-38, D-39, D-40, D-41, D-53, D-54

### E-1. Retire shipped lookahead (D-39)

**Current:** `lookahead_enabled` in stage config; `predicted_pos()` in target generator.  
**After:** Remove `lookahead_enabled` from all stage configs. Remove `predicted_pos()` call from `base_env.py`. The future samples system (D-16–D-18) replaces it.

**Files touched:** MODIFY `stage_config.py`, MODIFY `base_env.py`, MODIFY `target_generator.py`  
**Decision trace:** D-39

### E-2. Target always visible (D-40)

**Current:** `target_visible` can be False in Stage 1 (hides target fields).  
**After:** The drone always receives target info (with noise). No perception model. `target_visible` is always `True` for stages with targets. The `[18]` visible flag becomes constant `1.0` (keep it for now for obs dimension stability; removing it is PROPOSED but not decided).

**Files touched:** MODIFY `base_env.py`, MODIFY `obs_builder.py`  
**Decision trace:** D-40

### E-3. History + future observation stacking (D-18, D-38)

**Current:** History only: `[frame_t, frame_{t-s}, ..., frame_{t-m·s}]`.  
**After:** History + future: `[frame_t, hist_{t-p}, ..., hist_{t-m·p}, fut_{t+q}, ..., fut_{t+n·q}]`.

Parameters n, m, p, q from config file (D-38):
- `m` = history_frames, `p` = history_skip (past)
- `n` = future_samples, `q` = future_skip (future)

Actor observation: `frame_dim × (1 + m + n)`.

```python
class ObsBuilder:
    def __init__(self, obs_cfg, future_cfg):
        self.m = future_cfg.history_frames   # past count
        self.p = future_cfg.history_skip     # past spacing
        self.n = future_cfg.future_samples   # future count
        self.q = future_cfg.future_skip      # future spacing
```

**Files touched:** MODIFY `obs_builder.py`  
**Decision trace:** D-18, D-38

### E-4. Future samples from ground truth or predictor (D-16)

Based on `future_source` in config:
- `true` → get future target positions from the simulator's polynomial path (exact)
- `pred` → get future from the predictor module (Phase F)

Both produce the same shape: `n` frames of target fields, each in the drone's current body frame (D-54).

**Implementation in `obs_builder.py`:**
```python
def _build_future_frames(self, env_state, target, future_source):
    frames = []
    for i in range(1, self.n + 1):
        t_future = current_time + i * self.q * PH_DT
        if future_source == 'true':
            pos, vel = target.position_at(t_future), target.velocity_at(t_future)
        else:
            pos, vel = self.predictor.predict(t_future)
        # D-54: convert to current body frame relative to current position
        rel_pos = R_WB.T @ (pos - drone_pos)
        rel_vel = R_WB.T @ (vel - drone_vel)
        frame = np.array([*rel_pos, *rel_vel, 1.0])  # 7 fields
        frames.append(frame)
    return np.concatenate(frames)
```

**Files touched:** MODIFY `obs_builder.py`, MODIFY `base_env.py`  
**Decision trace:** D-16, D-41, D-54

### E-5. Body-frame transform for past and future target fields (D-54)

All past and future target fields are converted to the drone's **current** body frame, measured from its **current** position. This prevents a turning drone from making a stationary target look like it's moving.

```python
# For each past/future frame:
rel_pos = R_WB_current.T @ (target_pos_at_t - drone_pos_current)
rel_vel = R_WB_current.T @ (target_vel_at_t - drone_vel_current)
```

Only target fields are transformed; drone body states (rot6, v, omega) in past frames stay in their original frame.

**Files touched:** MODIFY `obs_builder.py`  
**Decision trace:** D-54

### E-6. Store past drone poses for predictor (D-53)

The predictor needs the drone's own measurement history (target observations since episode start). Store a ring buffer of past target measurements:

```python
class ObsBuilder:
    def reset(self):
        self._measurement_history = []  # D-53: cleared at every reset
    
    def on_step(self, target_obs):
        self._measurement_history.append(target_obs)
```

**Files touched:** MODIFY `obs_builder.py`  
**Decision trace:** D-53

### E-7. Privileged critic observations (D-11, D-12, D-15)

The critic receives additional inputs NOT available to the actor:

| Critic field | Source | Decision |
|---|---|---|
| `time_remaining` | `(max_steps - current_step) / max_steps` | D-15 |
| `facing_error` | `yaw_error` (body frame, from D-37) | D-57 |
| `target_true_pos` | `R_WB.T @ (true_target_pos - drone_pos)` error | D-12 |
| `target_true_vel` | `R_WB.T @ (true_target_vel - drone_vel)` error | D-12 |

The critic input list is defined in the model file (D-11).

**SB3 asymmetric critic implementation:**
SB3 does not natively support different actor/critic observation spaces. Implementation options:
1. Custom policy class that masks critic-only features from actor
2. Use SB3's `features_extractor` to split the observation

```python
class AsymmetricMlpPolicy(ActorCriticPolicy):
    """MlpPolicy with privileged critic inputs."""
    
    def __init__(self, obs_space, action_space, lr_schedule, 
                 actor_obs_dim, critic_obs_dim, **kwargs):
        self.actor_obs_dim = actor_obs_dim
        super().__init__(obs_space, action_space, lr_schedule, **kwargs)
    
    def extract_features(self, obs):
        actor_obs = obs[:, :self.actor_obs_dim]
        critic_obs = obs  # full observation including privileged
        return actor_obs, critic_obs
```

**Files touched:** NEW `interceptor-training/src/training/asymmetric_policy.py`, MODIFY `obs_builder.py`, MODIFY `orchestrator.py`  
**Decision trace:** D-11, D-12, D-15

---

## Phase F — Predictor Module

> **Decisions:** D-16, D-19, D-42, D-43, D-53

### F-1. Create predictor interface

```python
# interceptor-training/src/prediction/__init__.py

class Predictor(ABC):
    """Base class for target predictors."""
    
    @abstractmethod
    def reset(self) -> None:
        """Clear history at episode start (D-53)."""
    
    @abstractmethod
    def update(self, measurement: np.ndarray, timestamp: float) -> None:
        """Feed a new target measurement (D-53: own measurements only)."""
    
    @abstractmethod
    def predict(self, t_future: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return predicted (position, velocity) at time t_future."""
```

### F-2. `const_vel` predictor (D-19)

```python
class ConstVelPredictor(Predictor):
    """Constant-velocity predictor: assumes target moves at its last observed velocity."""
    
    def predict(self, t_future):
        dt = t_future - self._last_t
        pos = self._last_pos + self._last_vel * dt
        vel = self._last_vel
        return pos, vel
```

### F-3. `linear_ridge` predictor (D-19, D-43)

Analytic (no ML, no fitted artifact — D-43). Closed-form ridge regression on the measurement window.

```python
class LinearRidgePredictor(Predictor):
    """Analytic linear ridge predictor on measurement history."""
    
    def __init__(self, ridge_strength: float = 1e-3):
        self.alpha = ridge_strength
    
    def predict(self, t_future):
        # Build design matrix from stored measurements
        # Fit ridge: w = (X^T X + alpha I)^-1 X^T y
        # Extrapolate to t_future
```

### F-4. Predictor factory

```python
def make_predictor(name: str, **kwargs) -> Predictor:
    if name == 'const_vel':
        return ConstVelPredictor()
    elif name == 'linear_ridge':
        return LinearRidgePredictor(**kwargs)
    raise ValueError(f"Unknown predictor: {name}")
```

**Files touched:** NEW `interceptor-training/src/prediction/__init__.py`, NEW `src/prediction/const_vel.py`, NEW `src/prediction/linear_ridge.py`  
**Decision trace:** D-16, D-19, D-42, D-43, D-53

---

## Phase G — Reward Changes

> **Decisions:** D-39, D-40, D-57

### G-1. Facing reward (D-57)

New reward term from Stage 2 onwards:

```python
def _compute_facing_reward(self, drone_pos, drone_yaw, target_pos):
    """Facing reward: penalize not looking at target."""
    los = target_pos[:2] - drone_pos[:2]
    horiz_dist = np.linalg.norm(los)
    if horiz_dist < 0.5:  # switch off under ~0.5m
        return 0.0
    desired_yaw = np.arctan2(los[1], los[0])
    facing_error = abs(wrap_angle(desired_yaw - drone_yaw))
    return -self.cfg.k_facing * facing_error
```

Add `r_facing` to the reward components dict. Add `k_facing` to `RewardConfig` and `StageConfig`.

**Note:** `k_facing` value is OPEN in the Session Handoff. Use a reasonable default (e.g., `0.15`) and make it configurable in stage overrides.

**Files touched:** MODIFY `reward.py`, MODIFY `stage_config.py`  
**Decision trace:** D-57

### G-2. Remove lookahead from reward (D-39)

Remove all references to `lookahead_enabled` and `predicted_pos()` from reward computations. The reward uses the actual target position (with noise, per D-40).

**Files touched:** MODIFY `reward.py`, MODIFY `base_env.py`  
**Decision trace:** D-39

### G-3. Update reward summation to include facing

```python
r_step = r_alive + r_alt + r_tilt + r_angvel + r_thrust + r_smooth
       + r_velocity_alignment + r_progress_delta + r_time_penalty
       + r_facing  # NEW (D-57)
       + r_kill_bonus + r_crash + r_oob
```

**Files touched:** MODIFY `reward.py`  
**Decision trace:** D-57

---

## Phase H — Smoke Test & Bug Fixes

> **Decisions:** D-2, D-7, D-8, D-58

### H-1. Update `smoke_test.py` to use smoke files (D-2, D-7)

**Current:** Tests load `default_run.yaml`.  
**After:** Tests load `smoke_model.yaml` + `smoke_config.yaml`.

Update every test that constructs a `RunConfig` to use the new loader:
```python
models = load_models("configs/smoke_model.yaml")
config = load_config("configs/smoke_config.yaml")
```

Update `config_loader_validation` test to validate:
- `config_stages("ppo_baseline") == [1, 2, 3, 4, 5, 6, 7]` (stage list is now passed to `train_model`, not in the file; update test accordingly)
- `config_obs_history("ppo_5layer_deep") == {"frames": 5, "skip": 3}`

### H-2. Smoke test Stage 8 under `pred` only (D-8, D-42)

The `all_stage_envs_run` test runs Stage 8 with `future_source: pred` (not `true`, since there's no polynomial path for evasive targets).

### H-3. Update observation stacking test

Update `obs_history_stacking` to test the new history + future format: `(1 + m + n) × frame_dim`.

### H-4. Update reward test

Add `r_facing` to the reward sanity test.

### H-5. Update curriculum test

Update `curriculum_advance_cap_rollback` to test:
- `on_capped: stop` behavior
- STUCK stops only that model (D-55)
- Denied capped source (D-56)

### H-6. Bug fix pass (D-58)

Per Session Handoff "FIX pending" list:
- BUGS items 12, 11 (test), 2/1, 8, 3, 5, 10 — review and fix each
- README mass discrepancy (§13.6: 0.04085 kg, not 27g)
- Monitor CSV overwrite guard (§13.5)
- `steps_added` undercount on rollback (§13.4)

### H-7. R-1 investigation (ESC polynomial)

Per Session Handoff: "motor speed falls as thrust command rises." Review `pipeline.py:L99-L110` and `constants.py:L121-L122`. If confirmed, the ESC coefficients need correction. This is critical for sim-to-real transfer but non-blocking for training.

**Action:** Investigate and report. Fix if coefficients are clearly wrong; otherwise document as known limitation.

**Files touched:** MODIFY `smoke_test.py`, MODIFY `BUGS.md`, MODIFY `README.md`, potentially `pipeline.py` and `constants.py`  
**Decision trace:** D-2, D-7, D-8, D-42, D-58

---

## Phase I — Repo Sync, Clean, Push, Docker

### I-1. Clean the github_repos folder completely

```powershell
cd "C:\Users\ADMIN\Desktop\projects\github_repos\rl-drone-flight-simulator"
# Remove ALL files except .git/
Get-ChildItem -Force | Where-Object { $_.Name -ne '.git' } | Remove-Item -Recurse -Force
```

### I-2. Copy updated codebase from working directory

Copy ONLY the clean files from `interceptor-training/` and essential root files:

```
rl-drone-flight-simulator/
├── .gitattributes
├── .gitignore
├── BUGS.md
├── CURRENT_WORK.md
├── README.md
├── context.md                              ← updated context reflecting D-2 to D-58
├── Interceptor_Complete_Project_Scope.docx  ← per user request
├── docs/
│   └── coefficient_fitting.md
├── hover_env.py                            ← Stage-1 reference (parity)
├── interceptor-training/
│   ├── Dockerfile
│   ├── .dockerignore
│   ├── run.py                              ← NEW one-file runner
│   ├── configs/
│   │   ├── model.yaml                      ← NEW
│   │   ├── config.yaml                     ← NEW
│   │   ├── smoke_model.yaml                ← NEW
│   │   └── smoke_config.yaml               ← NEW
│   ├── data/
│   │   ├── .gitkeep
│   │   ├── checkpoints/.gitkeep
│   │   ├── curriculum_state/.gitkeep
│   │   ├── results/.gitkeep
│   │   └── tb_logs/.gitkeep
│   ├── docker-compose.yml
│   ├── requirements.txt
│   ├── scripts/
│   │   ├── evaluate.py
│   │   ├── smoke_test.py                   ← MODIFIED
│   │   └── train.py                        ← MODIFIED
│   └── src/
│       ├── __init__.py
│       ├── api.py                          ← NEW
│       ├── results.py                      ← NEW
│       ├── envs/
│       │   ├── __init__.py
│       │   ├── base_env.py                 ← MODIFIED
│       │   ├── obs_builder.py              ← MODIFIED
│       │   ├── reward.py                   ← MODIFIED
│       │   ├── stage_config.py             ← MODIFIED
│       │   └── target_generator.py         ← MODIFIED
│       ├── physics/
│       │   ├── __init__.py
│       │   ├── aero.py
│       │   ├── constants.py
│       │   ├── pipeline.py
│       │   └── quaternion.py
│       ├── prediction/                     ← NEW
│       │   ├── __init__.py
│       │   ├── const_vel.py
│       │   └── linear_ridge.py
│       ├── training/
│       │   ├── __init__.py
│       │   ├── asymmetric_policy.py        ← NEW
│       │   ├── callbacks.py
│       │   ├── checkpoint_manager.py       ← MODIFIED
│       │   ├── curriculum.py               ← MODIFIED
│       │   ├── orchestrator.py             ← MODIFIED
│       │   └── weight_transfer.py          ← NEW
│       └── utils/
│           ├── __init__.py
│           ├── config_loader.py            ← MODIFIED
│           ├── model_loader.py             ← NEW
│           └── logger.py
└── requirements.txt
```

**Note:** Root-level legacy files (`swift_*.py`, `compare_*.py`, `demo.py`, notebooks, etc.) are NOT synced to the clean repo. They stay in the messy working directory only. The clean repo contains only the active training pipeline + `hover_env.py` (parity reference) + essential docs.

### I-3. Include `Interceptor_Complete_Project_Scope.docx`

Per user request — copy from working directory to repo root.

### I-4. Git operations

```powershell
cd "C:\Users\ADMIN\Desktop\projects\github_repos\rl-drone-flight-simulator"
git add -A
git commit -m "feat: implement D-2 to D-58 design decisions

- Three-artifact architecture (model.yaml + config.yaml + run.py)
- train_model() public API with TrainResult
- Privileged critic with asymmetric observations
- History + future observation stacking
- Polynomial target paths for Stages 5-7
- const_vel and linear_ridge predictors
- Facing reward from Stage 2
- Capped/stuck/skip/crash/deny handling
- Resume with rollback persistence
- Weight transfer across observation size change
- Smoke test updated with own model/config files"

git tag -a v6.0 -m "D-2 to D-58 design decisions implemented"
git push origin main
git push origin v6.0
```

### I-5. Docker build and smoke test

```powershell
cd "C:\Users\ADMIN\Desktop\projects\github_repos\rl-drone-flight-simulator\interceptor-training"
docker build -t interceptor-drone-rl:v6.0 .
docker run --rm interceptor-drone-rl:v6.0 --smoke
```

### I-6. Fresh-clone verification

```powershell
$tmp = New-TemporaryFile | ForEach-Object { Remove-Item $_; mkdir $_ }
git clone https://github.com/siddhmehta5131/interceptor-drone-rl.git "$tmp\repo"
cd "$tmp\repo\interceptor-training"
docker build -t interceptor-drone-rl:fresh .
docker run --rm interceptor-drone-rl:fresh --smoke
```

---

## Decision Traceability Matrix

| Decision | Phase | Steps | Status |
|---|---|---|---|
| D-2 | H | H-1 | Smoke test uses own files |
| D-3 | D | D-1 | Config-driven target_alt |
| D-4 | D | D-2 | Fixed polynomial per episode |
| D-5 | D | D-2 | Paths independent of drone |
| D-6 | D | D-2 | Stage 8 disabled |
| D-7 | H | H-1 | Smoke runs all models/stages |
| D-8 | H | H-2 | Smoke Stage 8 under pred |
| D-9 | A | A-1 to A-5 | Three-artifact design |
| D-10 | B | B-1 | One call trains all models |
| D-11 | E | E-7 | Critic inputs in model file |
| D-12 | E | E-7 | Critic reference as error |
| D-13 | D | D-2, D-3 | Polynomial backwards + ep length |
| D-14 | D | D-3 | Episode length range |
| D-15 | E | E-7 | Actor hides time remaining |
| D-16 | E, F | E-4, F-1 | Future source selector |
| D-17 | — | — | Superseded by D-53 |
| D-18 | E | E-3 | History + future (n, m, p, q) |
| D-19 | F | F-2, F-3 | const_vel + linear_ridge |
| D-20 | A | A-5 | Retire default_run.yaml |
| D-21 | B | B-1 | Stages as argument |
| D-22 | B | B-1, B-3 | name argument + log folder |
| D-23 | B | B-2 | Save/load model result |
| D-24 | C | C-1 | Capped is not error |
| D-25 | C | C-1 | on_capped setting |
| D-26 | B | B-1, B-2 | Result keyed by model_id |
| D-27 | C | C-2 | Skip with reason |
| D-28 | C | C-3 | Continuation reuses settings |
| D-29 | B | B-1 | Write final to disk |
| D-30 | A, C | A-2, C-1 | on_capped in config |
| D-31 | C | C-5 | Crash isolation |
| D-32 | C | C-9 | Stage outcomes in result |
| D-33 | C | C-4 | Weight transfer 14→76 |
| D-34 | A, B | A-4, B-2 | Model_id is Python name |
| D-35 | C | C-2 | Capped not continued |
| D-36 | B | B-1 | Optional seed argument |
| D-37 | D | D-4 | Yaw faces target |
| D-38 | A, E | A-2, E-3 | n, m, p, q in config file |
| D-39 | E, G | E-1, G-2 | Retire shipped lookahead |
| D-40 | E | E-2 | Target always visible |
| D-41 | E | E-4 | Future has same fields |
| D-42 | H | H-2 | Smoke S8 pred only |
| D-43 | F | F-3 | linear_ridge is analytic |
| D-44 | A | A-1 | Names kept |
| D-45 | C | C-6 | Rollback counts persist |
| D-46 | C | C-7 | Mid-stage resume |
| D-47 | C | C-8 | SAC/TD3 buffer persist |
| D-48 | A | A-6 | Range syntax {min, max} |
| D-49 | A | A-3 | Smoke model + config files |
| D-50 | D | D-1 | target_alt Stage 1 only |
| D-51 | D | D-2 | Polynomial params ranged |
| D-52 | D | D-3 | Stage 1 fixed length |
| D-53 | E, F | E-6, F-1 | Predictor uses own history |
| D-54 | E | E-5 | Body-frame transform |
| D-55 | C | C-5 | STUCK stops one model |
| D-56 | C | C-2 | Capped source denied |
| D-57 | G | G-1 | Facing reward |
| D-58 | H | H-6 | FIX pass last |

---

## Execution Order

```
Phase A (config split)         ← foundation, everything depends on this
  ↓
Phase B (train_model API)      ← depends on A
  ↓
Phase C (training loop)        ← depends on B
  ↓
Phase D (environment)          ← depends on A (config), parallel with E/F
Phase E (observations)         ← depends on A, parallel with D
Phase F (predictors)           ← depends on E
  ↓
Phase G (rewards)              ← depends on D, E
  ↓
Phase H (smoke test + fixes)   ← depends on all above
  ↓
Phase I (repo sync + push)     ← final stage
```

**Estimated file changes:** 13 modified, 10 new, 1 deleted = 24 files total.

---

## Open Items (require user input before implementation)

| # | Item | Default if not answered |
|---|---|---|
| 1 | `k_facing` weight value | `0.15` |
| 2 | Facing cutoff distance | `0.5 m` |
| 3 | Fix R-1 (ESC polynomial) before implementation? | Investigate and report, don't block |
| 4 | `linear_ridge` strength parameter | `1e-3` |
| 5 | Remove visible flag `[18]` or keep as constant 1? | Keep as constant 1 |
| 6 | Stage 5-7 polynomial free parameter values | Use current stage_config defaults |
| 7 | Seed semantics (D-36): one seed per call? | Default to `global.seed` |

---

**GATE: Awaiting APPROVED before execution begins.**
