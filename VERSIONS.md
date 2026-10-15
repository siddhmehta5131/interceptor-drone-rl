# Version Registry
**Project:** RL-Based Autonomous Interceptor Drone
**Last updated:** 2024-09-10

---

## Core Files

### `hover_env.py` — v2.0.0
**Purpose:** Gymnasium environment for Stage 1 altitude-hold training.
**Dependencies:** `numpy`, `gymnasium` (optional — stub fallback), `scipy` (optional — for `brentq`)
**Observation:** 14-dim float32 (6D rotation, velocity, angular rate, altitude error, vertical velocity)
**Action:** Box([-1,1]^4) → [thrust, roll, pitch, yaw]
**Physics:** RK4 integrator, Crazyflie 2.1 parameters from `swift_live_demo_fitted.py`

| Version | Date       | Changes |
|---------|------------|---------|
| 2.0.0   | 2024-09-10 | Full rewrite: altitude-plane reward (no XY penalty), alive bonus, `render()`/`close()` stubs, uniform `PH_*` naming, `_ph_*` function prefix, random tilt/XY jitter at reset, 14-dim obs with explicit Δz and v_z, proper `bool`/`float` casts, backward-compatible aliases `HoverEnv`/`HoverRewardConfig` |
| 1.0.0   | —          | Original: 3D position penalty, no alive bonus, no render(), mixed naming conventions |

---

### `SB3_AltitudeHold_Benchmark.ipynb` — v2.0.0
**Purpose:** Multi-algorithm RL benchmark notebook (Colab-ready).
Trains **PPO, A2C, SAC, TD3, DDPG** on `AltitudeHoldEnv` and produces a professional comparison dashboard.
**Dependencies:** `stable-baselines3[extra]>=2.0`, `gymnasium`, `scipy`, `matplotlib`, `pandas`

| Version | Date       | Changes |
|---------|------------|---------|
| 2.0.0   | 2024-09-10 | Full rewrite: trains all 5 SB3 continuous-action algorithms, Monitor CSVs saved to per-algo checkpoint dirs, convergence curves from Monitor logs, 3D trajectory gallery with checkpoint colour progression, radar chart, matplotlib summary table, best-algorithm deep-dive, portable Colab/local paths, version-stamped header |
| 1.0.0   | —          | Original `PPO_Hover_Training.ipynb`: PPO-only, hardcoded Colab paths, no Monitor filename → broken convergence plots, truncated Stage 1b cell |

---

## Physics Reference Files

### `swift_live_demo_fitted.py` — v1.0.0
**Purpose:** Primary physics reference — complete gamepad-driven flight demo with validated Crazyflie 2.1 parameters.
**Role:** Source-of-truth for all physical constants, aerodynamic coefficients, and the RK4 integrator.
All other physics files derive from this one.

### `swift_physics_headless.py` — v1.0.0
**Purpose:** Headless (no pygame/matplotlib) extraction of the physics pipeline.
Importable as a module with a structured `params` dict.
Contains both RK4 and Euler integrators.

### `swift_physics_headless_v2.py` — v1.0.0
**Purpose:** RL-friendly wrapper around `swift_physics_headless.py`.
Adds `step()` / `step_batch()` API and adaptive scipy integrator backends (RK45, DOP853).

---

## Analysis / Validation Files

### `Coeff solver.ipynb` — v1.0.0
**Purpose:** Least-squares regression to fit aerodynamic polynomial coefficients from NanoBench / PX4 flight-log data.

### `phy_sim_val.ipynb` — v1.0.0
**Purpose:** Physics simulator validation — compares simulated trajectories against real flight data.

### `pybullet_vs_swift_comparison.ipynb` — v1.0.0
**Purpose:** Head-to-head comparison between this custom simulator and PyBullet's Crazyflie model.

### `compare_pybullet.py` — v1.0.0
**Purpose:** Script version of the PyBullet comparison (no notebook dependency).

### `validate.py` / `validate_flight_only.py` / `validate_onestep.py` — v1.0.0
**Purpose:** Validation scripts comparing simulator output to real NanoBench flight logs at various granularities.

### `self_test.py` — v1.0.0
**Purpose:** Internal physics sanity checks (hover equilibrium, motor response, etc.).

---

## Other Files

### `demo.py` / `default_demo.py` — v1.0.0
**Purpose:** Interactive flight demos with pygame gamepad input and live matplotlib visualisation.

### `identify_params.py` — v1.0.0
**Purpose:** System identification — estimates physical parameters from flight data.

### `references.txt`
**Purpose:** URLs and descriptions of external resources that were removed during cleanup (can be re-downloaded in <2 min).

### `BTP plan preliminary.docx`
**Purpose:** Bachelor Thesis Project preliminary plan document.
