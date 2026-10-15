# Antigravity Interceptor Drone Context

This document tracks the complete context for the Antigravity Interceptor Drone task, including the visualiser codebase overview, identified issues, known fixes, unsolved issues, and the overarching project goals.

## 1. Project Goal
The user provided a set of trained Reinforcement Learning (RL) models (`models and meta data stage 1-8.zip`) trained over a ~63M step curriculum to control an interceptor quadcopter. The goal is to generate 9 high-quality, full-length demonstration videos:
* 8 videos showing the PPO models completing Curriculum Stages 1 through 8.
* 1 video demonstrating the base physics simulator using a PID instruction-based controller (`physics_simulator_demo.mp4`).

**Requirements**:
* **Complete length**: Videos must capture the entire duration of the episodes (e.g., 3.0s or 6.0s depending on the stage), not just a fraction of a second.
* **Execution Environment**: We are shifting to running tests and diagnostics **locally** rather than exclusively inside the Docker container to speed up iteration.
* **Presentation**: A final presentation document detailing the physics simulator, RL models (SAC, TD3, PPO), observation structures, and the 8-stage curriculum.

## 2. Visualiser Codebase Overview
The project is split into the core training environment and a separate stage-runtime visualizer.
* **`interceptor-training/src/envs/base_env.py`**: The core Gymnasium environment. It integrates the quadcopter physics, aerodynamic effects, targets, and RK4 integration step (`_physics_step`).
* **`interceptor-training/src/physics/pipeline.py`**: The physics pipeline. It translates `Box([-1, 1])` actions to thrust and body rate commands via `command_from_action()`, applies PID rate control, and computes RK4 derivations.
* **`interceptor-training/src/envs/obs_builder.py`**: Handles stacking history frames, future trajectory samples, and privileged critic fields into a flat 1D observation vector. It applies sensor noise (`obs_noise_std`) before stacking.
* **`stage-runtime/stage_runtime/runner.py`**: The evaluation loop. It loads `stage_XX.json` configs, builds the environment, loads the controller (instructions or ML model), and steps the environment to capture Matplotlib frames for `ffmpeg`.
* **`stage-runtime/stage_runtime/controllers.py`**: The controller abstraction. The `ModelController` loads SB3 `.zip` files and performs `model.predict(obs, deterministic=True)`.

## 3. Issues & Known Fixes

### Issue 1: Stable Baselines 3 Observation Shape Mismatch (Fixed)
* **Problem**: When running `PPO_stage_2_final.zip` (which expects an 84-dim observation) on `stage_02.json` (which configures the environment to yield a 105-dim observation due to `future_samples=3`), SB3 immediately threw `ValueError: Unexpected observation shape (105,)`.
* **Root Cause**: The JSON stage configuration and the training metadata defining the checkpoint's internal observation space were misaligned.
* **Fix**: Modified `controllers.py:resolve_observation()` to actively parse the checkpoint zip's `data` file to extract `_shape` (e.g., 84). It then uses this to dynamically update the `cfg.observation` (e.g., forcing `future_samples = 0`) to perfectly match the model's dimensions.

### Issue 2: Matplotlib/Docker FFmpeg Dependency (Fixed)
* **Problem**: Generating `.mp4` videos raised an error because `imageio-ffmpeg` was missing in the rendering pipeline.
* **Fix**: Adapted the script `generate_final_videos.ps1` to loop through all stages and successfully write out `.mp4` files using the existing container dependencies.

## 4. Unsolved Issues (Crashing Drone)

### Issue: Early Episode Termination (The "1-second Video" Bug)
* **Problem**: The generated RL videos are only ~0.2 seconds long because the episodes are terminating prematurely. The `info` dictionary reports that the drone's tilt is exceeding the 60-degree crash threshold (`max_tilt_deg` > 60.0) within the first 10-30 simulation steps.
* **Diagnostics Performed**:
    1.  **Physics Engine Integrity**: The physics engine is **stable**. The `physics_simulator_demo.mp4` uses the same `base_env.py` and RK4 step with an `instructions` controller and hovers perfectly for the full duration.
    2.  **Observation Dimensions**: The environment now successfully supplies exactly the 14-dim (Stage 1) or 84-dim (Stage 2) observations the models expect.
    3.  **Action Outputs**: The `PPO_stage_1_final` model (evaluated locally with the exact same initial state) produces highly correlated rate commands (e.g., `roll = -0.16, pitch = 0.16, yaw = 0.16`). In a 100Hz simulation, these constant rate commands saturate the PID controller, causing the drone to quickly pitch/roll over and trigger the termination condition `crashed_flip`.
* **Potential Root Causes to Investigate**:
    *   **Missing Normalization**: The most common cause of out-of-distribution neural network behavior is missing `VecNormalize` wrapper application. However, a repository-wide search found zero instances of `VecNormalize`, and there are no `vecnormalize.pkl` files. Did the training script implement manual scaling?
    *   **Noise Bug**: `obs_builder.py` applies Gaussian noise to elements of the observation. Testing revealed that `obs[8]` (`v_z` in world frame) receives noise, but `obs[13]` (a duplicate of `v_z` at the end of the 14-dim array) remains exactly zero-noise, potentially confusing a highly sensitive policy.
    *   **Action Mapping Scaling**: The model outputs `[-1, 1]` which maps to angular rates via `PH_W_SCALE` (2.0 rad/s). If the curriculum training utilized a different scale or frequency, the test-time execution will multiply the intended rates, resulting in instant flips.
    *   **Trained Checkpoint Validity**: The zip contains checkpoints with `num_timesteps: ~3,000,000`. Are we sure these are the successful finalized models, or did the curriculum training diverge?

## 5. Next Steps / Code Usage (Local Execution)
1. **Local Evaluation Script**: Stop using `docker compose run`. Write a standalone Python script in the project root to load `base_env.py`, load the PPO model natively on Windows, and step the environment.
2. **Observation Inspection**: Dump the exact observations and actions step-by-step and compare the statistical distribution against a typical RL stable state to definitively rule out normalization bugs.
3. **Action Scaling Ablation**: Experiment with dynamically downscaling the model's action outputs before passing them to the environment (e.g., `action * 0.1`) to see if the policy was trained on a much smaller rate scale.
