# Implementation Plan — Layered Interceptor System

Version 2.0 · written 2026-11-12 · supersedes `updation_plan.md` (v1.0)

This plan takes the project from its current state (three partly overlapping
codebases) to the agreed submission scope: a validated quadrotor simulator, an
RL flight controller, an APN+RL path planner for three interceptor types, a
closed-loop demonstration, a visualiser, and full mathematical documentation.

How to read it:

- Section 1 is the verified starting point, including defects found while
  reading the code.
- Sections 2–3 fix the target architecture and the decisions.
- Section 4 is the work, in twelve phases. Each task has an ID (`P3.4`), the
  files it touches, and an acceptance check.
- Sections 5–9 are the cross-cutting rules: Colab workflow, run matrix,
  statistics, risks, and the final checklist that defines "done".

Stopping rule: the project is finished when every box in section 9 is ticked.
It is defined by artefacts existing, not by RL winning any comparison.

---

## 1. Verified starting point

### 1.1 What exists

| Codebase | Location | Size | State |
|---|---|---|---|
| A. Guidance package v1.0 | this repo: `src/`, `visualiser/`, `scripts/`, `tests/`, `configs/` | ~7,100 lines | 52 fast tests pass (re-run 2026-11-12). Never trained: `data/runs/` is empty. Single interceptor, z-up frame, 3 stages (static, polynomial, random-jerk "evasive"). |
| B. Manoeuvres + 3-platform notebooks | this repo: `archive/path_planning_updated_project/` | ~3,000 unique lines | `evasive_maneuvers.py` (14 laws, NED frame), v2 Colab notebook (3 platforms, seeker noise, 3-way benchmark), Three.js web visualiser. Trained in Colab; checkpoints were saved to `checkpoints_unified_noisy_v2/` in the Colab session and are **not** in the repo. |
| C. Crazyflie physics simulator | `C:\Users\ADMIN\Desktop\projects\github_repos\rl-drone-flight-simulator` (commit `493d370`) | ~20,500 lines, of which the physics core is ~750 | 6-DOF RK4 pipeline, rate PID, mixer, ESC curve, polynomial aero, OU wind, ground effect, domain randomisation. Its RL pipeline was only smoke-run (≈800 steps per stage, 0 % success). |

Local environment: Python 3.14, numpy 2.5, gymnasium 1.3, stable-baselines3
2.9, torch 2.14 (CPU), pyvista 0.49. `pybullet` and `gym-pybullet-drones` are
not installed. This repo has **no git remote** and one commit, with a large
uncommitted working tree.

### 1.2 Defects and gaps found in the code

These are the items a detailed reviewer will find. Each is assigned to a task.

| # | Finding | Evidence | Fixed in |
|---|---|---|---|
| F1 | The three codebases use different world frames. A and C are z-up; B is NED (z-down). | `intercept_env.py` checks `p[2] < 0` for ground; notebook checks `p_M[2] > 0`; `PH_G_W = (0,0,-9.81)`. | P1.6 |
| F2 | The benchmark's "pure APN" arm uses raw measurements and no axial term, while APN+RL uses filtered measurements plus an axial speed boost. The RL contribution is confounded. | Notebook cell 6, `step()`. | P5.7 |
| F3 | Results rest on 5 episodes per cell, one training seed. Pure RL had 300k steps against 400k. | Notebook cells 7–8. | P7, §6 |
| F4 | The filter named "Kalman" is a fixed-gain α-β-γ filter with hand-set gains 0.65 / 0.55 / 0.35. | `SeekerSensorFilter.observe`. | P5.4 |
| F5 | Platform limits, target limits, noise levels, manoeuvre constants, reward weights, kill radii and the 0.35 residual authority have no source. | `InterceptorLimits.from_preset`, `GenericUnconstrainedTarget`. | P2, P5 |
| F6 | The quad class (45 m/s, 5.2 g) is far outside a Crazyflie. Simulator constants give thrust-to-weight 2.05; the gym-pybullet-drones Crazyflie file implies a maximum total thrust of 0.595 N, which at 40.85 g is thrust-to-weight ≈ 1.49. | `PH_OMEGA_MAX = 2800`, `PH_C_L = 2.618e-8`; `cf2x.urdf` `thrust2weight = 2.25` at 27 g. | P2.3, P4.9 |
| F7 | `PH_OMEGA_MAX = 2800 rad/s` has no source and disagrees with the 2,273 rad/s implied by `cf2x.urdf`. | as above | P2.3 |
| F8 | Yaw torque coefficient: simulator uses `c_d = c_l/480`; `cf2x.urdf` gives `kf/km ≈ 39.8`. A factor of ~12 difference in yaw authority. | `constants.py`; `cf2x.urdf`. | P2.3 |
| F9 | The ESC curve is a two-point anchored fit `Ω = 1170 + 1630·√cmd`, not fitted to data. The earlier fitted curve was monotonically decreasing and was replaced. | `constants.py` "BLOCKING FIX" note. | P3.4 |
| F10 | Two aerodynamic coefficients had their signs flipped by hand because the fit used collinear features. Magnitudes are unverified. | `constants.py` "BLOCKING FIX" note. | P3.5 |
| F11 | The existing PyBullet comparison applies the simulator's own forces to a bare sphere. It checks the integrator, not an independent drone model. | `compare_pybullet.py`, `compare_v3_vs_pybullet.py`. | P3.7 |
| F12 | Validation predates the October fixes to ESC, aero and PID, so earlier plots no longer describe the current model. NanoBench data is not on disk. | `references.txt`; fix dates. | P3 |
| F13 | Physics timestep is hard-coded (`PH_DT = 0.01` used inside `deriv` and `rk4`). | `pipeline.py`. | P3.1 |
| F14 | Rate-PID derivative gain had to be cut to avoid a limit cycle at 100 Hz. The control rate is too low for the 20 ms motor lag. | `constants.py` note on `PH_KD`. | P3.1, P4.3 |
| F15 | Rate command range is ±2 rad/s, too small for interception manoeuvres. | `PH_W_SCALE = 2.0`. | P4.2 |
| F16 | With `γ = 0.995` and episodes up to 1,100 steps, a terminal kill is discounted to ≈ 0.007 at the first step. | `model.yaml`; notebook cell 7. | P5.9 |
| F17 | The oracle is recomputed every step in codebase A (stage 3 ran at ~280 steps/s against ~1,280 for stages 1–2). | `docs/presentation.md` §8.1. | P5.10 |
| F18 | Observation scales are hard-coded for one vehicle (`rel_pos: 100`, `rel_vel: 60`). | `obs_builder.py` `NORM_SCALES`. | P5.8 |
| F19 | Kill detection in codebase A is sampled once per step; a fast pass-through can be missed. The notebook has a sub-step closest-approach check. | `intercept_env.py` step 7. | P5.6 |
| F20 | The gym-pybullet-drones `DSLPIDControl` is a position + attitude controller with no body-rate loop. It cannot be compared like-for-like with a rate controller. | Source read 2026-11-12. | P4.5, P4.6 |
| F21 | Drone meshes have no recorded licence; no fixed-wing or missile mesh exists. | `models/README.md`. | P10.5 |
| F22 | `.gitignore` excludes `*.png`, `*.zip`, `*.mp4`, so results and figures cannot be committed. | `.gitignore`. | P0.4 |
| F23 | Stale code and docs: `weight_transfer.py` (old observation layout), `download_models.py`, statements that meshes are placeholders. | `docs/BUGS.md` #6, #8. | P1.9 |

### 1.3 Facts verified from external sources (2026-11-12)

- `gym-pybullet-drones` 2.2.0 requires Python ≥ 3.12, numpy ≥ 2.5, gymnasium
  ≥ 1.3, stable-baselines3 ≥ 2.9, pybullet ≥ 3.2.7. This matches the local
  versions.
- `cf2x.urdf`: mass 0.027 kg, inertia (1.4e-5, 1.4e-5, 2.17e-5) kg·m², arm
  0.0397 m, `kf = 3.16e-10` N/RPM², `km = 7.94e-12` N·m/RPM², thrust-to-weight
  2.25, max speed 30 km/h, drag, ground-effect and downwash coefficients.
  Converted: `kf = 2.88e-8` N/(rad/s)², close to the simulator's `2.618e-8`.
- Crazyflie firmware (`platform_defaults_cf2.h`): rate PID roll/pitch
  `Kp 250, Ki 500, Kd 2.5`, yaw `Kp 120, Ki 16.7`; attitude PID roll/pitch
  `Kp 6, Ki 3`, yaw `Kp 6, Ki 1, Kd 0.35`. Units are degrees and 16-bit motor
  counts. Loop rates were not in that file and must be read from the firmware
  (believed 500 Hz; verify).
- Fixed-wing reference, Aerosonde UAV: `mavsim_public` (companion code to
  Beard & McLain, *Small Unmanned Aircraft*), file
  `mavsim_python/parameters/aerosonde_parameters.py`. It provides mass 11.0 kg,
  wing area 0.55 m², span 2.8956 m, chord 0.18994 m, full longitudinal and
  lateral coefficient sets, a stall model (`M = 50`, `α₀ = 0.47 rad`),
  parasitic drag, Oswald factor, and a complete propeller and motor model.
  It does not state a structural load limit.
- Missile reference, AIM5: the generic air-intercept missile from Zipfel's
  CADAC++ (book supplement to *Modeling and Simulation of Aerospace Vehicle
  Dynamics*), public at `github.com/SwaggyTyrion/CADAC` under `example/AIM5/`.
  It provides mass against time (63.8 → 38.6 kg), thrust against time (peak
  39.2 kN, burnout at 3.02 s), specific impulse 267 s, reference area
  0.01767 m², diameter 0.15 m, lift and drag tables over angle of attack
  (0–35°) and Mach (0.7–5) for motor on and off, angle-of-attack limit 35°,
  the code that derives the g limit from it, autopilot constants (rate-loop
  time constant 0.1 s, acceleration-loop gain 40 rad/s²), navigation gain 4,
  launch condition (269 m/s at 10 km), and a 3-DOF manoeuvring target
  aircraft with its own limits.
- Preliminary values computed from those files (to be redone properly in
  P2.4): Aerosonde stall speed ≈ 11 m/s using the file's stall model, whose
  peak lift coefficient (2.42) is optimistic; AIM5 aerodynamic limit ≈ 38 g at
  Mach 2 and 10 km after burnout, ≈ 4 g at launch speed and full mass.

---

## 2. Target architecture

### 2.1 Repository layout

```
interceptor/                  installable package (pip install -e .)
  common/     frames.py, quaternion.py, units.py, config.py, logging.py, seeding.py
  sim/        params.py, motor.py, aero.py, wind.py, rigid_body.py, integrator.py, quadrotor.py
  fc/         env.py, setpoints.py, pid.py, firmware_pid.py, mixer.py, attitude.py, metrics.py
  guidance/   geometry.py, apn.py, platform.py, kinematics.py, maneuvers.py, targets.py,
              seeker.py, tracker.py, oracle.py, obs.py, reward.py, env.py
  bridge/     setpoint.py, closed_loop.py
  training/   algos.py, policy.py, callbacks.py, runner.py, checkpoints.py
  eval/       stats.py, sim_validation.py, fc_eval.py, guidance_eval.py, closed_loop_eval.py,
              sweeps.py, tables.py
  visualiser/ schema.py, scene.py, renderer.py, animator.py, compose.py, cli.py
configs/      platforms.yaml, quadrotor.yaml, wind.yaml, seeker.yaml, maneuvers.yaml,
              guidance.yaml, fc.yaml, training.yaml, eval.yaml, visualiser.yaml
notebooks/    01 … 07 (Colab, thin)
scripts/      build_source_table.py, check_equation_refs.py, make_fixtures.py, run_eval.py
tests/        unit/, integration/, fixtures/
docs/         mathematics.md, source_table.md, algorithm_selection.md, sim_validation.md,
              report.md, presentation.md, figures/
results/      tables/, figures/, videos/, manifest.json
archive/      untouched historical material
```

### 2.2 Conventions (fixed for the whole project)

| Item | Convention |
|---|---|
| World frame | Right-handed, x north, y west, z up. Gravity `(0, 0, −9.81)`. Ground at `z = 0`. |
| Body frame (quad) | x forward, y left, z up (thrust axis). Matches `PH_R_P`. |
| Quaternion | `[w, x, y, z]`, rotates body vectors into world. |
| NED legacy code | Converted with `(x, y, z)_NED → (x, −y, −z)`. |
| Units | SI throughout. Angles in radians internally; degrees only at I/O. |
| Rates | Quad physics 1000 Hz (RK4), flight controller 500 Hz, guidance decisions 50 Hz. The guidance kinematic model integrates at 200 Hz (4 sub-steps per decision) so the actuator lag is resolved. |
| Randomness | One `numpy.random.Generator` per environment, seeded explicitly. No global RNG. |
| Config values | Every leaf is `{value, unit, source}`. A test fails if `source` is missing. |
| Equations | Each has an ID in `docs/mathematics.md` (for example `G-3`). Implementing functions cite the ID in their docstring. A script checks every cited ID exists. |

### 2.3 Interfaces between layers

```
guidance  →  GuidanceCommand { a_des[3], v_des[3], omega_des[3] }      world frame, 50 Hz
bridge    →  RateSetpoint    { collective ∈ [0,1], omega_cmd[3] rad/s } body frame, 500 Hz
fc        →  MotorCommand    { cmd[4] ∈ [0,1] }                         500 Hz
sim       →  QuadState       { p, q, v, omega, Omega[4] }               1000 Hz
```

Every environment writes the same trajectory-log schema
(`visualiser/schema.py`), so one renderer serves all layers.

---

## 3. Decisions

### 3.1 Settled

| ID | Decision |
|---|---|
| S1 | Flight controller interface: collective thrust + three body rates in, four motor commands out. |
| S2 | Reference quadrotor: Crazyflie 2.x. |
| S3 | Simulator, flight controller and closed loop are quad only. Path planning covers quad, fixed-wing and missile. |
| S4 | One RL algorithm for both models, chosen by derivation plus a three-way experiment. |
| S5 | Visualiser is a Python script that renders videos on chosen or random seeds. A web app is out of scope. |
| S6 | All training runs in Colab; the laptop runs tests, small evaluations and low-resolution renders. |
| S7 | All extras are in scope: robustness sweeps, ablations, residual analysis, domain randomisation, actuator lag, reproducibility. |
| S8 | Rates: quad physics 1000 Hz, flight controller 500 Hz, guidance 50 Hz with 200 Hz kinematic integration. Reasons: the motor time constant is 20 ms, so 100 Hz control leaves two samples per time constant and forced the PID derivative cut (F14); 500 Hz matches the firmware loop so its gains can be used unchanged; 1000 Hz physics gives two RK4 steps per control step; 50 Hz guidance matches the existing environments and a realistic seeker update rate; 200 Hz integration keeps the step below a quarter of the shortest actuator lag. |
| S9 | Reference vehicles: quad = Crazyflie 2.x; fixed-wing = Aerosonde UAV (Beard & McLain, data from `mavsim_public`); missile = AIM5 generic air-intercept missile (Zipfel, data from CADAC++). All three have openly available parameter files (§1.3). |
| S10 | Hosting: `github.com/siddhmehta5131/interceptor-drone-rl` (public) is the project repository and what Colab clones. `github.com/siddhmehta5131/colab-workspace` (private) is not needed by this plan. Checkpoints live on Google Drive and in GitHub Releases. |

### 3.2 Open — recommendation given, confirm before the phase listed

Open decisions are numbered Q1–Q12 (Q3, Q4 and Q10 are settled above). Deliverables are numbered D1–D8.

| ID | Decision | Recommendation | Needed by |
|---|---|---|---|
| Q1 | Guidance model for fixed-wing and missile | Keep one unified kinematic model with per-platform constraints; use the point-mass dynamics only to derive the constraint values. | P2 |
| Q2 | Quad parameter set | Primary: NanoBench-fitted set (40.85 g). Secondary: stock `cf2x.urdf` set (27 g), used only for the PyBullet cross-check. | P2 |
| Q5 | Target definition | Define the target by two ratios relative to the interceptor: speed ratio and acceleration ratio. Sweep both in P9. | P2 |
| Q6 | Kill radius | One rule for all platforms, tied to vehicle size, and always report the full miss-distance distribution so no conclusion depends on the radius. | P2 |
| Q7 | Tracking filter | Replace the fixed-gain filter with a constant-acceleration Kalman filter whose Q and R come from the noise model. | P5 |
| Q8 | Discount and reward | Raise `γ` so the horizon covers an episode, and make the progress term potential-based. | P5 |
| Q9 | Package name | Rename `src` to `interceptor` and add `pyproject.toml`. | P1 |
| Q11 | Fixed-wing load limit | The Aerosonde file has no structural limit and its stall model gives an optimistic peak lift coefficient. Adopt a cited structural load factor for small UAVs and take the lower of the structural and aerodynamic limits. | P2 |
| Q12 | Missile flight phase | The AIM5 limit varies from about 4 g at launch to about 38 g after burnout. Define the missile platform at one stated condition (recommended: post-burnout coast at Mach 2, 10 km) and show the variation in the mathematics document. | P2 |

---

## 4. Phases

Legend: **[L]** runs on the laptop, **[C]** runs in Colab. Size: S (hours),
M (days), L (a week or more of part-time work). Sizes are rough estimates.

Dependency order:

```
P0 → P1 → P2 → P3 → P4 ─┐
                └→ P5 ──┼→ P6 → P7 → P8 → P9 → P10 → P11
```

P4 and P5 can proceed in parallel after P3 and P2 respectively.

---

### P0 — Safeguard and baseline  [L] (S)

Goal: nothing can be lost, and the starting point is recorded.

| ID | Task | Acceptance |
|---|---|---|
| P0.1 | Check the Colab session and Google Drive for `checkpoints_unified_noisy_v2/`. If found, download into `archive/checkpoints_v2/`. | Present, or recorded as lost in `docs/CURRENT_WORK.md`. |
| P0.2 | Commit the current working tree as-is on `master` ("snapshot before consolidation"). Tag `v1.0-pre-consolidation`. | `git status` clean. |
| P0.3 | In `interceptor-drone-rl`, tag the existing `main` (old simulator, `493d370`) as `v6.0-physics-archive`. Add the repository as `origin` here and push this project to a new branch `v2`, leaving `main` untouched. Make `v2` the default branch once P1 is complete. The repository is public and the licence of `models/*.obj` is unknown (F21), so the meshes are **not** published: add `models/*.obj` to `.gitignore`, keep the files on disk, and push `v2` as a fresh root commit of the current tree without them (the existing local commit `bb9c119` contains the meshes and stays local, tagged `v1.0-local`). The visualiser falls back to primitive shapes until licensed meshes are added in P10.5. | `git remote -v` shows origin; `v2` visible on GitHub with no `.obj` files in its history; old code reachable by tag. |
| P0.4 | Fix `.gitignore`: keep ignoring `data/` and checkpoints, but allow `results/**` and `docs/figures/**`. | A test PNG under `results/figures/` is tracked. |
| P0.5 | Create `Drive/interceptor/` with `runs/`, `data/`, `results/`. | Folder exists. |
| P0.6 | Record the baseline: run the 52 fast tests and the slow test; save output to `docs/baseline_2026-11.txt`. | File committed. |

---

### P1 — Consolidate into one package  [L] (L)

Goal: one tested package containing everything worth keeping. Behaviour is
preserved exactly; improvements come in later phases.

**Golden fixtures first.** Before moving any code, capture its behaviour.

| ID | Task | Acceptance |
|---|---|---|
| P1.1 | `scripts/make_fixtures.py`: from the old simulator, record state trajectories for six fixed motor-command sequences (hover, climb, roll step, pitch step, yaw step, mixed) with default parameters, no wind. Save as `tests/fixtures/sim_*.npz`. | Six files, each with inputs and full state history. |
| P1.2 | Same script: from `archive/.../evasive_maneuvers.py`, record target trajectories for all 16 behaviours × 3 platforms from fixed initial conditions against a straight-flying interceptor. Save as `tests/fixtures/maneuver_*.npz`. | 48 files. |
| P1.3 | Same script: from the v2 notebook code (extracted to a temporary module), record 3 full episodes per platform under zero action with fixed seeds, for the raw-APN mode. | 9 files. |

**Then move code.**

| ID | Task | Source → destination | Acceptance |
|---|---|---|---|
| P1.4 | Create `pyproject.toml` (Python ≥ 3.12), rename `src/` to `interceptor/`, update imports, move `visualiser/` inside. | all of codebase A | 52 tests pass under the new import path; `pip install -e .` works. |
| P1.5 | Port the physics core. | C: `physics/{constants,quaternion,aero,pipeline}.py` → `interceptor/sim/` | Replays of P1.1 fixtures match to 1e-12. |
| P1.6 | Port manoeuvres with frame conversion NED → z-up. Re-derive the path-frame normals `n_h`, `n_v` and every altitude test in the new frame. | B: `evasive_maneuvers.py` → `guidance/maneuvers.py` | Replays of P1.2 fixtures match after applying `(x, −y, −z)`, to 1e-9. |
| P1.7 | Port platform limits, seeker noise, constraint enforcement, lag and jerk limit, closest-approach check, noise schedule callback. Values are copied unchanged for now. | B: notebook cells 2–7 → `guidance/platform.py`, `seeker.py`, `kinematics.py`, `env.py`, `training/callbacks.py` | Replays of P1.3 fixtures match to 1e-9. |
| P1.8 | Merge the two environments into one `guidance/env.py` with a `mode` argument (`apn_raw`, `apn_filtered`, `apn_rl`, `pure_rl`) and a `target_set` argument (`static`, `polynomial`, `evasive`). Keep `PolynomialTargetPath` from codebase A as the non-evasive target set. | A + B | Old stage-1 and stage-2 behaviour reproduced (existing tests adapted and passing). |
| P1.9 | Delete `weight_transfer.py`, `download_models.py`, and the stale statements in docs. | A | No references remain (`grep`). |
| P1.10 | Bring over the old simulator's domain-randomisation sampler and ground-effect function. Do not port the old 8-stage environment, its training pipeline, `stage-runtime/`, or the probe scripts. | C | Listed in `docs/CURRENT_WORK.md` as intentionally left behind. |
| P1.11 | Unify logging, seeding and config loading in `interceptor/common/`. | A | One config loader; all environments accept `seed`. |

Exit: all tests pass, fixtures match, tag `v2.0-consolidated`.

---

### P2 — Parameters, sources and derivations  [L] (L)

Goal: every number has a source or a derivation. This phase is mostly reading
and arithmetic, and it produces the first half of the mathematics document.

| ID | Task | Acceptance |
|---|---|---|
| P2.1 | Convert all configs to the `{value, unit, source}` schema. Write `scripts/build_source_table.py` (generates `docs/source_table.md`) and the test `test_all_constants_sourced`. Unsourced values are marked `source: TODO` and listed by the test. | Table generates; test reports the TODO count. |
| P2.2 | Write `docs/mathematics.md` §1 (engagement geometry, APN derivation from zero-effort miss) and §2 (unified kinematic model). Assign equation IDs. Write `scripts/check_equation_refs.py`. | Script passes. |
| P2.3 | **Quadrotor parameters.** For each of mass, inertia, arm, thrust coefficient, torque coefficient, maximum motor speed, motor time constant, find the primary source (Förster 2015; Bitcraze thrust and PWM measurements; NanoBench paper; `cf2x.urdf`). Resolve findings F6, F7, F8 explicitly: state which value is adopted and why. | `configs/quadrotor.yaml` has zero TODO; a comparison table of candidate values is in `docs/sim_validation.md`. |
| P2.4a | **Fixed-wing: Aerosonde.** Copy the parameter values from `aerosonde_parameters.py` into `configs/airframes/aerosonde.yaml`, each citing the file and the textbook table. Write `docs/mathematics.md` §3: point-mass equations and the derivation of stall speed (`v_min`), maximum level speed from thrust = drag using the propeller model (`v_max`), `a_t,max` and `a_t,min` from excess thrust and drag, `a_n,max` and `ω_max` from the load factor (decision Q11), and the acceleration lag `τ` from the roll and pitch response. Evaluate at sea level and the file's trim speed (25 m/s), and plot each limit against speed. | Fixed-wing entry in `configs/platforms.yaml` has zero TODO; each value cites an equation ID. |
| P2.4b | **Missile: AIM5.** Copy the mass, thrust, lift and drag tables from the CADAC++ `aim5_prop_deck.asc` and `aim5_aero_deck.asc` into `configs/airframes/aim5.yaml`, citing Zipfel. Write `docs/mathematics.md` §4: point-mass equations with thrust and variable mass, normal-force limit from the angle-of-attack limit (as in the CADAC++ aerodynamics module), speed envelope from the boost profile and drag, axial acceleration from thrust and drag, and `τ` from the autopilot constants. Evaluate at the condition chosen in decision Q12 and plot the limits over the flight. Reproduce the CADAC++ g-limit calculation in a unit test. | Missile entry has zero TODO; unit test matches the CADAC++ formula. |
| P2.4c | Record the licences: `mavsim_public` (listed as GPL-3.0; confirm in the repository) and the CADAC repository (MIT, added by the repository owner; the code and data originate from Zipfel's book supplement). Only numerical data is used, with citation; no source code is copied. | Note in `docs/source_table.md`. |
| P2.5 | **Target definition** (decision Q5): speed ratio and acceleration ratio per platform, with a cited or argued baseline value. | In `configs/platforms.yaml`. |
| P2.6 | **Kill radius and engagement geometry** (decision Q6): kill radius rule, initial range, altitude and episode length per platform, each derived from platform speed and a stated time-to-intercept. | In `configs/guidance.yaml`. |
| P2.7 | **Seeker noise model**: choose and cite a range-dependent angular noise model, velocity and acceleration noise, dropout probability. | `configs/seeker.yaml` zero TODO. |
| P2.8 | **Manoeuvre constants**: trace each formula in `context evasive.txt` to a primary textbook. Express each law's constants as non-dimensional parameters (for example weave frequency relative to the interceptor time constant, break time relative to `τ`). Write `docs/mathematics.md` §5 with all 14 laws. | `configs/maneuvers.yaml` zero TODO. |
| P2.9 | **Navigation constant**: document the admissible range of `N′` and plan the tuning sweep (run in P5.11). | Section in mathematics document. |

The quad's guidance-level limits are **not** set here. They are measured from
the simulator in P4.9.

Exit: `test_all_constants_sourced` passes for platforms, quadrotor, seeker and
manoeuvres.

---

### P3 — Simulator completion and validation (D1)  [L develop, C run] (L)

Goal: a simulator whose every model term is justified, compared against an
independent simulator and against real flights.

**Model work**

| ID | Task | Acceptance |
|---|---|---|
| P3.1 | Make the timestep a parameter everywhere (fix F13). Add sub-stepping so one control step runs N physics steps. Adopt the rates in S8. | Fixture replay at 100 Hz still matches; convergence study at 100, 250, 500, 1000, 2000 Hz shows fourth-order error decay. |
| P3.2 | Analytic verification tests: free fall; hover equilibrium; torque-free rotation conserves angular momentum and rotational energy; constant torque about a principal axis; single-axis thrust acceleration. | Each within 1e-6 relative error at 1000 Hz. |
| P3.3 | Write `docs/mathematics.md` §6: full 6-DOF equations as implemented (translation, rotation, quaternion kinematics, torque assembly, rotor gyroscopic terms, motor lag), with equation IDs cited in `sim/`. | Equation-reference check passes. |
| P3.4 | Motor and ESC model (fix F9): replace the anchored curve with a fit to published Crazyflie PWM-to-thrust and PWM-to-speed data, including battery-voltage dependence if the data supports it. Document the fit and residuals. | Fit plot in `docs/figures/`; coefficients sourced. |
| P3.5 | Aerodynamic model (fix F10): refit the drag polynomial on NanoBench data with a train/test split by trajectory, with sign constraints that force dissipation, and regularisation. If the constrained fit is no better than a linear rotor-drag model, adopt the simpler model. | Held-out error reported; all coefficients dissipative; decision recorded. |
| P3.6 | Wind (D1 requirement): implement the Dryden turbulence model per MIL-HDBK-1797 (low-altitude intensities and length scales, forming filters for u, v, w) plus mean wind and discrete gusts. Note in the document that the existing Ornstein–Uhlenbeck process is the first-order Dryden longitudinal filter, and map its parameters. Handle the low-airspeed limit with a stated minimum airspeed. | Simulated spectra match the analytic Dryden spectra; variances match within 5 % over 10 minutes of simulated wind. |

**Validation**

| ID | Task | Acceptance |
|---|---|---|
| P3.7 | Cross-check against gym-pybullet-drones (replaces F11). Load the `cf2x.urdf` parameter set into our simulator. Feed identical motor-speed sequences to both for hover, vertical step, roll, pitch and yaw doublets, and an aggressive mixed sequence. Three levels: (a) rigid body only, motor lag and aero off; (b) with each simulator's drag model; (c) near ground. | Table of position, attitude, velocity and rate error over time per case; level (a) agrees to integrator tolerance; differences at (b), (c) explained by named model terms. |
| P3.8 | Real-flight validation on NanoBench with the primary parameter set: one-step prediction error, and open-loop rollout error at 0.1, 0.5 and 1.0 s horizons, on trajectories not used for fitting. Run the same test through gym-pybullet-drones for comparison. | Table: our simulator against PyBullet against real data. |
| P3.9 | Wind response: mean wind, gust, and turbulence cases; show the drone's open-loop drift and the closed-loop hold under the PID stack (after P4.4). | Plots in `docs/figures/`. |
| P3.10 | Speed: environment steps per second for both simulators, single process, same machine. | One table. |
| P3.11 | Write `docs/sim_validation.md` and notebook `01_sim_validation`. | Notebook regenerates every table and figure from a clean Colab runtime. |

Exit: D1 complete.

---

### P4 — Flight controller environment and classical baselines  [L] (L)

Goal: the task definition and all non-learned controllers, before any RL.

| ID | Task | Acceptance |
|---|---|---|
| P4.1 | `fc/env.py`. Observation (actor): three frames of [body rates (3), rate error (3), commanded collective (1), previous action (4)] = 33 values. Privileged critic input: mass, inertia, thrust and torque coefficients, motor speeds, wind, body velocity. Action: four values in [−1, 1] mapped to motor commands `(a+1)/2`. Episode: free flight from a random attitude and velocity, no ground, 4 s. | Gymnasium `check_env` passes; random policy runs. |
| P4.2 | `fc/setpoints.py`: generators for steps, ramps, doublets, sinusoids, chirps and band-limited random sequences on all three axes plus collective. Rate range set from the requirement measured in P4.9 (fix F15). | Unit tests on each generator. |
| P4.3 | Reward: `−w_ω·‖e_ω‖² − w_T·(thrust error)² − w_s·‖Δaction‖² − w_sat·(saturation)`, each term normalised to its range. Write `docs/mathematics.md` §8 with the justification of each term. Termination on body rate above a safety limit. | Reward unit tests; weights in `configs/fc.yaml` with sources or a planned sensitivity run. |
| P4.4 | Baseline B1: the project's own rate PID plus mixer, re-tuned at 500 Hz by a documented method (for example model-based loop shaping from the linearised rate dynamics, then verified by step response). | Step response meets stated rise-time and overshoot targets; derivation in the mathematics document. |
| P4.5 | Baseline B2: Crazyflie firmware rate PID and motor mixing, gains verbatim from the firmware with explicit unit conversion (degrees and 16-bit counts to radians and [0, 1]). Confirm loop rate and mixer from the firmware source. | Unit test reproduces the firmware formula on sample inputs. |
| P4.6 | Common outer loop for whole-stack tests: quaternion attitude controller (firmware attitude gains) producing rate setpoints, and a simple position loop. Any inner controller (B1, B2, RL) plugs in underneath. | Hover and attitude-step tests stable with B1 and B2. |
| P4.7 | Baseline B3: gym-pybullet-drones `DSLPIDControl`, run in its own simulator with the stock parameter set, on the same attitude-step and position-step tasks (fix F20). This is a whole-stack reference, not a rate-loop comparison. | Results table produced in Colab. |
| P4.8 | `fc/metrics.py`: rise time, overshoot, settling time, integral absolute error, RMS tracking error, control effort, action smoothness, saturation fraction; disturbance rejection under wind gust and under parameter error of ±10 % mass and inertia. | Unit tests on synthetic responses. |
| P4.9 | **Measure the quad's guidance-level limits** with the B1 stack on the validated simulator: maximum sustained lateral and vertical acceleration, maximum speed, acceleration time constant and jerk limit (from acceleration-command step responses), maximum turn rate. Write these into `configs/platforms.yaml` with `source: measured, P4.9`. Compare against the analytic values from thrust-to-weight. Record the body-rate range the attitude loop demands and feed it back to P4.2. | Quad entry complete; analytic and measured values agree within a stated tolerance or the gap is explained. |
| P4.10 | Evaluate B1 and B2 on the full metric set. | Classical results table exists before any RL training. |

Exit: the flight-controller task is frozen; changing it later invalidates
trained models.

---

### P5 — Guidance environment and classical arms  [L] (L)

Goal: the path-planning task and its two non-learned arms, for all three
platforms, with sourced values.

| ID | Task | Acceptance |
|---|---|---|
| P5.1 | Load platforms from `configs/platforms.yaml` (replacing the hard-coded presets). | Three platforms instantiate; values match the source table. |
| P5.2 | `guidance/kinematics.py`: unified model with tangential and normal acceleration limits, speed limits, first-order lag, jerk limit, turn-rate limit, integrated at 200 Hz with the command held for each 50 Hz decision. Cite equation IDs. | Unit tests: each limit binds when it should; a non-holonomic platform cannot drop below `v_min`; halving the integration step changes the trajectory by less than a stated tolerance. |
| P5.3 | `guidance/seeker.py`: measurement model from P2.7 with dropouts. | Sample statistics match configured values. |
| P5.4 | `guidance/tracker.py` (decision Q7): constant-acceleration Kalman filter, Q and R derived from the noise model, coasting through dropouts. Keep the fixed-gain filter as an option for reproducing the old benchmark. | Filter consistency test (normalised innovation within bounds); estimate error below raw measurement error. |
| P5.5 | `guidance/apn.py`: keep the zero-effort-miss form; add the axial speed-management term as a separate, documented function with its equation. | Existing APN tests pass; new tests for the axial term. |
| P5.6 | Kill detection by closest point of approach within each step (fix F19), against the true target position. Miss distance recorded as the episode minimum, independent of the kill radius. | Test: a pass-through between two samples is detected. |
| P5.7 | Four arms as environment modes, differing in exactly one factor at each step (fix F2): **A1** raw measurements, textbook APN. **A2** filtered measurements, APN plus axial term. **A3** A2 plus the RL residual. **A4** RL alone on the same filtered observation. | A3 with zero action equals A2 exactly (test). |
| P5.8 | Observation: keep the 16-value frame, history and prediction blocks; normalise by per-platform scales from config (fix F18). Privileged critic block unchanged. | Dimensions 85 / 16 / 101 asserted for all platforms; values within ±10 on random episodes. |
| P5.9 | Reward and discount (decision Q8, fix F16): terminal kill bonus, time cost, potential-based progress term, small penalties on residual magnitude and action change. Choose `γ` from the episode horizon. Write `docs/mathematics.md` §9. | Reward tests; a short note showing the discounted terminal bonus at the first step. |
| P5.10 | Oracle cost (fix F17): profile; use the closed-form boundary-value oracle per step and keep the full trajectory oracle for analysis only. | ≥ 1,000 environment steps per second on the laptop in evasive mode. |
| P5.11 | Tune the classical baseline so it is at its best: sweep `N′` over the admissible range for A1 and A2 on each platform, pick the best by kill rate then median miss. | Chosen `N′` per platform in config with `source: sweep P5.11`. |
| P5.12 | Guidance output: each step logs `a_des`, `v_des`, `ω_des` in the shared schema. | Present in the trajectory log. |
| P5.13 | Evaluate A1 and A2 on the full protocol (§6) for both target sets: non-evasive (static and polynomial orders 1–3) and evasive (16 behaviours). | Two tables per platform. The non-evasive table demonstrates the brief's claim that APN alone suffices there. |

Exit: the guidance task is frozen.

---

### P6 — Algorithm derivation and selection  [L write, C run] (M)

| ID | Task | Acceptance |
|---|---|---|
| P6.1 | Write `docs/algorithm_selection.md`: common setting; PPO from the policy-gradient theorem through the trust-region bound to the clipped objective and GAE; TD3 from the deterministic policy gradient with clipped double-Q, target smoothing and delayed updates; SAC from the maximum-entropy objective and soft Bellman equation. Then the criteria table scored separately for each model. | Document complete with references to the original papers. |
| P6.2 | State the decision rule before running anything: the analytically preferred algorithm is adopted unless another beats it with non-overlapping confidence intervals on final performance. | Rule written in the document. |
| P6.3 | `training/algos.py`: uniform construction of PPO, SAC and TD3 from Stable-Baselines3 with the same network sizes, symmetric (non-privileged) policies for fairness, and hyperparameters taken from a cited source (RL Baselines3 Zoo defaults). | Each algorithm trains for 2,000 steps on both environments in a test. |
| P6.4 | Notebook `02_algorithm_selection`: 3 algorithms × 2 tasks (flight controller; quad guidance arm A3) × 3 seeds, equal environment-step budget. Record learning curves, final performance, area under the curve, wall-clock time, seed variance, action smoothness. | 18 runs complete; figures and table generated. |
| P6.5 | Record the outcome and the chosen algorithm in the document. | Decision stated with evidence. |

The remaining phases assume PPO. If another algorithm is chosen, only
`configs/training.yaml` and the asymmetric-critic wiring change.

---

### P7 — Final training  [C] (L)

| ID | Task | Acceptance |
|---|---|---|
| P7.1 | `training/runner.py`: one function `train(run_id, config)` that is idempotent and resumable. It saves a checkpoint and state every N steps to Drive, resumes from the latest, and writes a `DONE` marker with the final model. Each run folder stores the resolved config, git commit hash, `pip freeze`, seed and learning curves. | Kill the process mid-run and re-run: training continues from the last checkpoint. |
| P7.2 | Benchmark Colab throughput (steps per second, with `SubprocVecEnv`) for both environments. Set step budgets so learning curves have plateaued; equal budgets for A3 and A4. | Budgets recorded in `configs/training.yaml` with the benchmark as source. |
| P7.3 | Notebook `03_train_flight_controller`: asymmetric actor-critic, domain randomisation on mass, inertia, thrust and torque coefficients and aerodynamic coefficients, wind enabled. 3 seeds. | 3 completed runs; training curves saved. |
| P7.4 | Notebook `04_train_guidance`: per platform, A3 with near-zero initial residual and the noise schedule, and A4 with the same budget. Manoeuvres sampled with the documented weighting. 3 platforms × 2 arms × 3 seeds. | 18 completed runs. |
| P7.5 | Copy final checkpoints to a GitHub Release; write `results/manifest.json` (run id, file, SHA-256, commit, config hash). | Manifest committed; every checkpoint loads on the laptop. |

---

### P8 — Evaluation and closed loop (D2, D3, D4)  [C run, L develop] (L)

| ID | Task | Acceptance |
|---|---|---|
| P8.1 | `eval/stats.py`: Wilson intervals for kill rate; median and inter-quartile range plus empirical distribution for miss distance; paired comparison on common seeds (McNemar for kills, bootstrap for miss distance); mean and spread across training seeds. | Unit tests against known values. |
| P8.2 | Flight controller evaluation: RL against B1 and B2 on every metric in P4.8, nominal and under wind and parameter error; whole-stack comparison including B3. | `results/tables/fc_*.csv`, step-response and tracking figures. |
| P8.3 | Guidance evaluation per §6: four arms × 3 platforms × 16 behaviours, plus the non-evasive set. | `results/tables/guidance_*.csv`; one summary table per platform; miss-distance distribution plots. |
| P8.4 | `bridge/setpoint.py`: desired thrust vector `m(a_des − g)`; collective by inverting the thrust curve; desired attitude from the thrust direction and a stated yaw rule; rate setpoint from the quaternion error. Write `docs/mathematics.md` §7. | Unit tests: hover command gives hover collective and zero rates; a lateral command tilts the correct way. |
| P8.5 | `bridge/closed_loop.py`: guidance at 50 Hz, flight controller at 500 Hz, physics at 1000 Hz, with wind. Target and seeker identical to the kinematic run. | One episode runs end to end. |
| P8.6 | Closed-loop evaluation on the quad: arms A2 and A3, inner controller RL and B1, same seeds as the kinematic runs. Report kill rate and miss distance for kinematic against full physics, and the acceleration-tracking error that explains the gap. | `results/tables/closed_loop.csv`; comparison figure. |
| P8.7 | Notebook `06_evaluate_all`: regenerates every table and figure from the checkpoints named in the manifest. | Runs on a clean Colab runtime without edits. |

---

### P9 — Extras (D8)  [C] (M)

| ID | Task | Acceptance |
|---|---|---|
| P9.1 | Robustness sweeps (evaluation only): kill rate and median miss against seeker noise scale, against target acceleration ratio, and against target speed ratio, for arms A2 and A3 on each platform. | Three figures per platform showing where each method breaks. |
| P9.2 | Ablations on the quad, 3 seeds each: no privileged critic; no history block; no prediction block. | Table against the full model. |
| P9.3 | Design-choice sensitivity on the quad: residual authority at three levels; `γ` at two levels. | Table; chosen values justified. |
| P9.4 | Residual analysis: for one hard manoeuvre per platform, plot the APN command, the RL residual and the target acceleration against time, and the residual's component along and across the line of sight. | Three figures with a written interpretation. |
| P9.5 | Flight-controller robustness: performance with and without domain randomisation in training, evaluated at ±20 % parameter error. | Table. |
| P9.6 | Kill-radius sensitivity: recompute kill rates from stored miss distances at several radii. | One figure per platform. |
| P9.7 | Notebook `05_ablations_and_sweeps`. | Regenerates P9 outputs. |

---

### P10 — Visualiser (D5)  [L develop, C or L render] (M)

| ID | Task | Acceptance |
|---|---|---|
| P10.1 | `visualiser/schema.py`: one trajectory-log schema for guidance, flight-controller and closed-loop runs. Adapt the existing renderer to it. | Existing stage videos still render. |
| P10.2 | Guidance mode: platform, arm, behaviour, seed or `--random-seeds N`; trails, line of sight, command arrows, kill sphere, heads-up display. | One video per platform renders. |
| P10.3 | Flight-controller mode: close view of the quad attitude with commanded and actual rate traces drawn alongside. | One video renders. |
| P10.4 | Closed-loop mode, and side-by-side composition of two or three runs on the same seed. | APN against APN+RL video renders. |
| P10.5 | Meshes (fix F21): record source and licence of the two existing meshes or replace them; add fixed-wing and missile meshes with clear licences, or build them from primitives. | `models/README.md` lists source and licence for every mesh. |
| P10.6 | Rendering path for the weak laptop: 1280×720 default locally; notebook `07_render_videos` for full-resolution renders in Colab with an off-screen display. | Both paths produce a valid MP4. |
| P10.7 | Render the delivered set into `results/videos/`: per platform, one hard manoeuvre, A2 beside A3; one flight-controller tracking clip, RL beside PID; one closed-loop clip. | Five videos. |

---

### P11 — Documents and freeze (D6, D7)  [L] (L)

| ID | Task | Acceptance |
|---|---|---|
| P11.1 | Complete `docs/mathematics.md`: §1 geometry and APN, §2 unified kinematics, §3 fixed-wing, §4 missile, §5 manoeuvres, §6 quadrotor dynamics and wind, §7 bridge, §8 flight-controller reward, §9 guidance observation and reward, §10 algorithms (summary of P6.1). Every equation has an ID and names its implementing function. | `check_equation_refs.py` passes in both directions. |
| P11.2 | Final `docs/source_table.md`. | `test_all_constants_sourced` passes with zero TODO across all configs. |
| P11.3 | `docs/report.md`: problem, architecture, methods, results for D1–D4 and D8, limitations, reproduction instructions. Update `docs/presentation.md`. | All figures and tables referenced exist in `results/`. |
| P11.4 | Rewrite `README.md`, `docs/CURRENT_WORK.md`, `docs/VERSIONS.md`, `docs/BUGS.md` for the final state. | No statement contradicts the code. |
| P11.5 | Clean-room reproduction: fresh clone, install, run tests; in Colab run `06_evaluate_all` from the Release checkpoints. | Both succeed; tag `v3.0-submission`. |

---

## 5. Colab workflow

1. Each notebook begins with the same three cells: mount Drive; clone the
   repository at a pinned commit and `pip install -e .`; print versions and
   the commit hash.
2. Notebooks contain no model logic. They call package functions with a run
   matrix and display results.
3. A run matrix is a list of run IDs: `d3-quad-a3-ppo-s0`,
   `d2-fc-ppo-s1`, and so on. The runner skips runs with a `DONE` marker and
   resumes partial ones, so a disconnected session is restarted by re-running
   the same cell.
4. Environments are CPU-bound; request a CPU runtime with the most cores
   available and use `SubprocVecEnv`. A GPU gives no benefit for these small
   networks.
5. `pybullet` and `gym-pybullet-drones` are installed only in the notebooks
   that need them (01, and the B3 part of 06). Tests that need them are marked
   and skipped on the laptop.
6. Datasets (NanoBench) are downloaded once to `Drive/interceptor/data/`.
7. Checkpoint compatibility: train and evaluate with the same pinned package
   versions; each run stores `pip freeze`.

---

## 6. Evaluation protocol

| Item | Rule |
|---|---|
| Training seeds | 0, 1, 2 for every learned model. |
| Evaluation seeds | A fixed list disjoint from training seeds, identical across arms (paired). |
| Guidance episodes | 100 per (platform, behaviour) cell, per model. |
| Guidance cells | 3 platforms × 16 behaviours (evasive set) + 3 platforms × 4 target types (non-evasive set). |
| Noise | Full noise scale 1.0 for headline tables; other scales only in sweeps. |
| Headline metrics | Kill rate with Wilson 95 % interval; miss distance median and inter-quartile range; time to intercept. |
| Across seeds | Report mean and range over the 3 training seeds; never the best seed alone. |
| Comparisons | Paired tests on common evaluation seeds; state effect size and interval. |
| Flight controller | 50 setpoint sequences per type, per controller, nominal and perturbed. |
| Closed loop | Same evaluation seeds as the kinematic quad runs. |

### Run matrix (learned models)

| Purpose | Runs |
|---|---|
| Algorithm selection | 3 algorithms × 2 tasks × 3 seeds = 18 (reduced budget) |
| Flight controller final | 3 |
| Guidance final | 3 platforms × 2 arms × 3 seeds = 18 |
| Ablations | 3 variants × 3 seeds = 9 |
| Sensitivity | (2 extra authority levels + 1 extra γ) × 3 seeds = 9 |
| Domain-randomisation control | 3 |
| **Total** | **60** |

---

## 7. Test plan

| Layer | Tests |
|---|---|
| Fixtures | Ported code reproduces recorded behaviour of the three original codebases (P1.1–P1.3). |
| Simulator | Analytic cases (P3.2), integrator convergence, wind statistics, parameter loading. |
| Flight controller | Environment API check, setpoint generators, reward terms, PID and firmware-PID formulas, metric functions. |
| Guidance | Geometry and APN identities, constraint enforcement per platform, tracker consistency, closest-approach kill detection, arm equivalence (A3 with zero action equals A2), observation dimensions and ranges, each manoeuvre law. |
| Bridge | Hover and tilt cases, round trip acceleration → setpoint → simulated acceleration. |
| Training | Each algorithm trains briefly on each environment; save, load and resume. |
| Evaluation | Statistics functions against known values; table generation on a tiny run. |
| Governance | All constants sourced; all equation references resolve. |
| Slow, marked | End-to-end short training and evaluation; PyBullet-dependent tests. |

All non-slow tests must pass on the laptop in under two minutes.

---

## 8. Risks

| Risk | Response |
|---|---|
| Old Colab checkpoints are gone. | They are superseded by P7; record the loss and move on. |
| The Crazyflie is weaker than assumed (F6), making the quad a slow interceptor. | The target is defined by ratios (Q5), so the engagement stays well-posed at any scale. |
| RL flight controller does not beat the PID. | Report it. The deliverable is the comparison. |
| Pure RL stays near zero at equal budget. | Report with learning curves; optionally add one longer-budget run, labelled as such. |
| Large gap between kinematic and full-physics results. | P8.6 measures the acceleration-tracking error; P4.9 already sets kinematic limits from the simulator to keep the gap small. |
| Airframe data for fixed-wing or missile is incomplete. | Both chosen files were read and cover mass, aerodynamics and propulsion (§1.3). The known gaps are the fixed-wing structural limit (Q11) and the choice of missile flight condition (Q12). |
| The three platforms now span very different scales (about 8, 40 and 600 m/s). | Engagement geometry is derived per platform from its speed (P2.6) and the target is defined by ratios (Q5), so the same environment serves all three. |
| NanoBench data unavailable or lacks needed channels. | Fall back to Bitcraze published measurements for thrust and to gym-pybullet-drones as the only cross-check; state the limitation. |
| Colab disconnects during long runs. | Resumable runner (P7.1); budgets sized from measured throughput (P7.2). |
| Package version drift between laptop and Colab. | Pinned versions; `pip freeze` stored per run. |
| Changing a frozen task definition after training. | Tasks are frozen at the end of P4 and P5; any later change requires retraining and is logged in `docs/VERSIONS.md`. |

---

## 9. Final checklist — definition of done

**D1 Simulator**
- [ ] Analytic verification tests pass.
- [ ] Cross-check against gym-pybullet-drones at three model levels.
- [ ] Validation against NanoBench flights on held-out trajectories.
- [ ] Dryden wind implemented, verified against analytic spectra, and demonstrated.
- [ ] Speed comparison table.

**D2 Flight controller**
- [ ] Trained RL policy, 3 seeds.
- [ ] Compared with own PID and firmware PID on all metrics, nominal and perturbed.
- [ ] Whole-stack comparison including the gym-pybullet-drones controller.

**D3 Path planner**
- [ ] APN+RL and pure-RL policies for quad, fixed-wing and missile, 3 seeds each.
- [ ] Four-arm tables with intervals for all 16 behaviours.
- [ ] Non-evasive tables showing APN alone.

**D4 Closed loop**
- [ ] Quad guidance through the flight controller on full physics, compared with the kinematic result.

**D5 Visualiser**
- [ ] One command renders any model, platform, behaviour and seed.
- [ ] Five delivered videos.

**D6–D7 Documents**
- [ ] Mathematics document complete, every equation linked to code.
- [ ] Source table with no missing sources.
- [ ] Algorithm-selection document with derivations and experiment.
- [ ] Report and presentation.

**D8 Extras**
- [ ] Robustness sweeps, ablations, sensitivity, residual analysis, domain-randomisation control, kill-radius sensitivity.

**Reproducibility**
- [ ] Checkpoints released with a manifest.
- [ ] `06_evaluate_all` regenerates every table and figure on a clean runtime.
- [ ] Fresh clone passes all tests. Tagged `v3.0-submission`.

**Out of scope:** hardware flights; 6-DOF dynamics or a flight controller for
fixed-wing and missile; an interactive web visualiser; adaptive or learning
targets; multiple interceptors or targets.
