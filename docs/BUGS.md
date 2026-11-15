# BUGS — known issues

This project is a fresh implementation of `updation_plan.md` v1.0 (APN + RL
interceptor). Findings below are grouped by status. Per task instructions,
**non-blocking** defects are logged here rather than fixed mid-task.

Date: 2026-10-16.

---

## Fixed during implementation

### 1. Plan §2.3 APN cross-product is not dimensionally consistent — FIXED
**Area:** `src/guidance/apn.py` (`compute_apn_command`)
**Impact:** Would have produced a wrong guidance command. The plan's literal form
`a_M = N'·(V_rel × ω_LOS) + (N'/2)·a_T⊥` has a non-zero component **along the
line of sight** (a cross product of two vectors is perpendicular to both, but
`V_rel` is not generally perpendicular to the LOS) and an inconsistent lateral
magnitude. Plan §2.4 then derives the correct equivalent, `a_M = N'·ZEM⊥/t_go²`,
which is what PN standard theory gives.
**Fix:** implemented the mathematically equivalent **zero-effort-miss** form
`a_APN = N'·perp(V_rel)/t_go + (N'/2)·perp(a_T)` with
`perp(x) = x − (x·R̂)R̂`. Verified against static, crossing and accelerating
targets (see `tests/test_apn.py`).

### 2. `t_go` unbounded explosion when the range opens — FIXED
**Area:** `src/guidance/apn.py` (`compute_time_to_go`, `compute_apn_command`)
**Impact:** High. When the interceptor opens the range (`V_c ≤ 0`),
`t_go = R / max(V_c, 0.1)` collapses to `10·R` and the zero-effort-miss
(∝ `V_rel · t_go`) exploded to ≈ 2.6 × 10⁵ per step, driving the reward to
≈ −1.75 × 10⁶.
**Fix:** clamp `t_go = clip(R / max(V_c, 0.1), t_go_min, t_go_max)` with a new
`t_go_max = 20 s` (configurable in `configs/config.yaml`).

### 3. Raw ZEM reward term was unbounded — FIXED
**Area:** `src/envs/reward.py`
**Impact:** Medium. Plan §10.3 specified `r_ZEM = −k_ZEM · ZEM`, but ZEM is a
metric distance, so the term dominated the per-step reward as the range grew.
**Fix:** normalise to `r_ZEM = −k_ZEM · |ZEM| / max(distance, 1.0)`.

### 4. `stage_cfg.apn` does not exist (config-access gotcha) — test fixed
**Area:** `tests/test_integration.py`
**Impact:** Low (test only). A `StageConfig` exposes the APN values as **flat**
fields (`stage_cfg.apn_N_prime`, `stage_cfg.apn_t_go_min`); there is no
`stage_cfg.apn`. The global `cfg.apn` **is** an `APNSpec`. An early test used the
wrong attribute and failed to construct the env; corrected to the flat fields.
**Note for future code:** use `cfg.apn.*` for the global spec and
`stage_cfg.apn_N_prime` / `stage_cfg.apn_t_go_min` for per-stage overrides.

---

## Non-blocking (logged, not fixed)

### 5. VTK GLES shader warnings during off-screen render — RESOLVED
**Area:** `visualiser/renderer.py`
**Impact:** Cosmetic only. On this headless, GPU-less Windows host, VTK printed
`vtkShaderProgram` / `vtkOpenGLPolyDataMapper` ERROR messages during every
off-screen render. The MP4 output was written correctly and was valid.
**Root cause (found):** the spam came from **shadow-map shaders** — the
unconditional `enable_shadows()` call from plan §14.3. A headless GLES context
cannot compile them, so VTK errored on every frame.
**Resolution:** shadows are now **opt-in** via `visualiser.yaml`
(`lighting.shadows: false` by default); `enable_shadows()` is only called when
true. With shadows off the log drops to ~10 lines and `vtkShaderProgram` hits = 0.
This is a deliberate deviation from plan §14.3 (documented in `CURRENT_WORK.md`).

### 6. `weight_transfer.py` constants tied to the old (physics) observation layout
**Area:** `src/training/weight_transfer.py`
**Impact:** None today. `SHARED_STAGE1_PREFIX = 12` and `STAGE1_OBS_DIM = 14`
reflect the previous physics observation, not the new 16-dim relative-kinematics
frame (actor dim 85). The module is retained per the plan ("KEEP AS-IS") but is
**not imported** by the new orchestrator, so the stale constants are inert.
**Suggested fix (future):** update the prefix/dim to the new observation or delete
the module if weight transfer is not used.

### 7. Full training budgets not exercised on this host
**Area:** `configs/stages.yaml`
**Impact:** Low (verification gap, not a defect). The 2 M / 5 M / 10 M step
budgets were only validated with smoke-scale `--steps` overrides on CPU. The
curriculum scheduler advance/cap logic is covered by unit tests, but no full
kill-rate threshold has been reached here.
**Suggested fix:** run the real curriculum on a GPU host.

### 8. `models/*.obj` meshes — placeholder origin (superseded)
**Area:** `models/interceptor_drone.obj`, `models/target_drone.obj`
**Impact:** None now. Originally `scripts/download_models.py` could not fetch
real assets and wrote generated placeholder quadcopters. The user has since
supplied **real** `interceptor_drone.obj` (748,744 B) and `target_drone.obj`
(841,970 B). `scripts/download_models.py` will overwrite these only with
`--force`, so they are preserved.
**Note:** the two models use different raw units/bounds (interceptor
12.4×3.4×8.8, target 3.1×2.0×3.9), so **per-model scale** is required — see
item 9.

### 9. Per-model mesh scale + orientation needed for the visualiser
**Area:** `configs/visualiser.yaml`, `visualiser/scene_builder.py`
**Impact:** Visual only. The single `models.scale: 0.01` from the plan §14.3
reference assumed one asset unit system. The two supplied OBJ files differ in
scale and axis convention, so each needs its own transform:
`interceptor_scale: 0.1612`, `target_scale: 0.5114` (each normalises its mesh to
~2 m max dimension), plus `interceptor_rotation: I08` and `target_rotation: T21`
(proper cube-rotation codes) to make each model upright and front-facing.
**Fix:** `scene_builder` gained `cube_rotations()`, `rotation_matrix_from()` and
`apply_mesh_rotation()`; `renderer._setup_drones` bakes the rotation into the
actor base mesh and applies the per-frame pose matrix on top.
**Status:** Resolved for the two supplied models. New assets need fresh codes —
see the orientation-shot workflow in `CURRENT_WORK.md`.

### 10. Off-screen render is slow on CPU (software rasteriser)
**Area:** `visualiser/renderer.py`
**Impact:** Convenience only. With no GPU, renders run ~**5.4 s/frame** at
1920×1080 (stage-1 = 68 frames ≈ 6 min; stage-2 = 105 frames ≈ 14 min). Shadows
doubled this to ~11 s/frame and are now disabled by default.
**Mitigation:** pass `--resolution 1280x720` (or lower) to the CLI; use a
GPU-backed OpenGL context for production renders.

---

# Update 2026-11-13 (plan v2.0 audit)

Items 1-10 above date from v1.0 and use old `src/` paths (now `interceptor/`). Items 6 and 8 were resolved by P1.9 / P0.3
(stale module deleted; meshes are local-only and gitignored).

## Open

### 19. Colab-only runs are pending
The algorithm comparison (P6.4), FC/guidance training (P7.3/P7.4), GitHub Release
upload (P7.5) and the clean-Colab run of notebook 06 (P8.7) require a Colab
runtime and, for B3, gym-pybullet-drones. The code, notebooks and budgets are
in place; results sections say "provisional"/"pending" rather than reporting
numbers that were never produced. Rerun the notebooks in Colab and replace the
pending markers.

### 11. NanoBench data missing -> P3.4 / P3.5 / P3.8 blocked
ESC curve is a two-anchor fit and the aero polynomial is inherited with hand-flipped signs; no held-out error exists.
`sim_validation.md` refers to "BUGS R-1 / R-2" but those entries did not exist; they are this item (R-1 = non-monotone ESC fit,
replaced; R-2 = two sign-flipped aero coefficients). Mitigation: linear rotor-drag model added as fallback (commit 5741b8b).

### 12. PyBullet / gym-pybullet-drones not installed -> P3.7, P4.7 pending
Run in Colab. Tests are marked and skipped locally.

### 13. Seeker filter is a fixed-gain alpha-beta-gamma filter (F4)
`guidance/seeker.py` gains 0.65/0.55/0.35 unsourced. Fix in P5.4 (Kalman).

### 14. `docs/mathematics.md` section 10
Resolved 2026-11-13: section 10 now points to `docs/algorithm_selection.md`;
the selected algorithm is recorded in `configs/training.yaml`.

### 15. fc env observation-space bounds are +-inf
`check_env` warns. Harmless; tighten when the FC task is frozen.

### 16. No `notebooks/` directory; Colab workflow untested
Resolved 2026-11-13 for authoring: notebooks 02, 03, 04 and 06 exist. They have
not been executed because no Colab runtime was available; the results sections
are explicitly marked provisional/pending.

### 17. Docs stale: `docs/presentation.md`, `docs/VERSIONS.md` still describe v1.0
Rewrite in P11.4.

### 18. Fixed-wing presets in `platform.py` disagree with `configs/platforms.yaml`
Resolved 2026-11-14: `PLATFORMS` (the env default) now loads the sourced `configs/platforms.yaml`;
the notebook presets survive as `LEGACY_PLATFORMS` / `from_preset` and are used only by the fixture-parity test.
Consequence: fixed-wing and missile default episodes changed; no fixed-wing/missile results existed. Original note:
`PLATFORMS[FIXED_WING]` keeps the v2 notebook values (`v_max=255`, `r_init_max=2800`,
`max_time=22`), while `configs/platforms.yaml` (sourced, EQ-FW-*) says `40`, `900`,
`36`. `UnifiedInterceptEnv` defaults to the presets, so a default fixed-wing episode
uses a different envelope from the documented one. The golden fixtures were recorded
against the presets, so changing the defaults now would break P1 parity. P5.8's
`observation_scales` block uses the config values and callers can pass
`limits=load_platforms(...)` explicitly; align the presets when the fixtures are
regenerated (P6/P7).


### 19. Compute-gated results (training, learned-arm evaluation, B3, P3.7, P3.8)
Checkpoints, learned-arm evaluation, B3, P3.7 PyBullet cross-check and P3.8 NanoBench
need Colab or external data. Code and notebooks exist; none has been executed.

### 20. Closed-loop residual authority was hard-coded 0.35 in two places
`UnifiedInterceptEnv` now takes `residual_authority` (default 0.35, unchanged); the
closed-loop bridge (`bridge/closed_loop.py`) still uses the literal 0.35. Resolved 2026-11-14: `run_closed_loop(residual_authority=0.35)` is now a parameter; pass the P9.3 value.

### 21. Mesh licences unknown (F21)
`models/README.md`: the two OBJ meshes have no recorded source. Fixed-wing and missile
meshes are generated from primitives and have no licence issue.

### 22. Visualiser side-by-side labels are slow
`render.side_by_side` draws each label with matplotlib; about 3 minutes per platform at
640x360 on the laptop. Replace with Pillow text if full-resolution local renders are needed.

### 23. Results produced before 2026-11-14 are stale
The platform envelopes changed (quad to the measured P4.9 closed-loop limits, fixed-wing to
14 CFR 23.337 load factors and the TB2 target speed, missile target to 9 g, kill radii to the
contact rule). `results/tables/guidance_quadcopter_*.csv`, `nav_sweep.csv` and
`nav_sweep_best.json` were computed with the old quad envelope (45 m/s, 5.2 g, 3 m) and must
be regenerated (`scripts/sweep_nav.py`, `scripts/evaluate_guidance.py`). The navigation
constant, reward weights and seeker clamp were tuned for the old envelope and need re-checking.
`fc_evaluation.csv` and `quad_limits.json` are unaffected (simulator unchanged).

**Extended 2026-11-15: every platform changed again, so every result table is stale.**
Quad: rotor drag kz 0.7 -> 0.5, envelope re-measured (v_max 22.0 -> 27.7 m/s, a 14.8 -> 15.4,
a_down 9.5, jerk 88.3), target 14.5 -> 18.3 m/s and 7.404 m/s^2, spawn 92-277 m; so
`quad_limits.json` and `fc_evaluation.csv` are no longer "unaffected". Fixed-wing: target is the
Orlan-10 (30.6 m/s, kill radius 3.00 m). All platforms: seeker angular noise 2.5 -> 8.727 mrad.
30-episode APN check on the new values (seeds 10000-10029, evasive set, zero residual):

| Platform | apn_raw | apn_filtered | Check of 2026-11-14 (old values) |
|---|---|---|---|
| quad | 15/30 | 25/30 | 17/30 |
| fixed-wing | 8/30 | 11/30 | 2/30 |
| missile | 8/30 | 5/30 | 12/30 |

Which mode the 2026-11-14 check used was not recorded, so compare with care. Open question raised
by this check: the missile case got worse. The 8.727 mrad term is a ground counter-UAS radar figure;
at the missile's 3.5-7 km spawn range it gives a 31-62 m position sigma, and the fixed-gain filter
does worse than raw APN there (5/30 against 8/30). Decide whether one angular figure should apply
to all three platforms, and rerun `sweep_nav.py` (N' was chosen on the old envelope).

### 24. Fixed-wing target is faster than the fixed-wing interceptor (FIXED 2026-11-14)
The TB2 (46.3 m/s) was faster than the 40 m/s Aerosonde and gave 2/30 pure-APN kills. The target
is now the Orlan-10: cruise 110 km/h = 30.6 m/s, span 3.1 m (Airforce Technology profile,
24 Mar 2023; trade press, no manufacturer datasheet), speed ratio 1.31, kill radius 3.00 m.
The 2026-11-15 check gives 8/30 (raw APN) and 11/30 (filtered APN): winnable, still the hardest
case (equal 3.8 g limits on both sides). The TB2 and Shahed-136 speeds are covered by the P9.1
`target_speed_ratio` sweep.

### 25. `configs/seeker.yaml` is never loaded (FIXED 2026-11-14)
The env now builds its noise model with `load_sensor_noise_config()` (`guidance/seeker.py`);
unknown or missing YAML keys raise `KeyError`. Two keys the code never implemented
(`coast_acceleration_decay`, `noise_floor`) were removed from the YAML. The `SensorNoiseConfig()`
dataclass defaults are still the notebook-v2 values (2.5 mrad) and are passed explicitly by the
fixture-parity tests only. Tests: `tests/unit/test_seeker_config.py` (5 tests).

### 26. Learner discount did not match the shaping discount (FIXED 2026-11-14)
The guidance reward shapes with `gamma = 0.1**(1/1100) = 0.99791` (EQ-GUIDE-8), but
`training/runner.py` passed the SB3 default `gamma = 0.99` from `configs/training.yaml` to
PPO/SAC/TD3; the horizon gamma lived only in the legacy `configs/model.yaml`. That breaks the
policy-invariance condition of potential-based shaping (Ng, Harada & Russell 1999) and undoes
the P5.9 horizon fix. The runner now sets the learner's gamma to `env.gamma` (an explicit
`algo_overrides` gamma still wins, so the P9.3 `gamma_0.99` ablation works) and writes the
resolved values to `<run>/hyperparameters.json`. Test: `test_learner_gamma_matches_shaping_gamma`.
No checkpoints existed, so nothing has to be retrained.

### 27. NanoBench reference was the wrong paper; two quad constants are unconfirmed
The audit named arXiv:2512.14450 (Busetto et al., 2025) as the candidate NanoBench reference.
That paper is a different data set: a 45 g Crazyflie 2.1 Brushless with `k_F = 3.72e-8`.
Our 40.85 g flying mass is stated in NanoBench itself, Ullah & Baca (2026), arXiv:2603.09908
(Sec. IV-B1) and in its repository README (github.com/syediu/nanobench-iros2026). The reference
was corrected on 2026-11-14. Still open: the thrust coefficient `PH_C_L = 2.618e-8` and the
motor time constant `0.02 s` are attributed to NanoBench but appear in neither the paper text
nor the README. They may come from the data set's metadata files or from our own fit; until
that is checked, treat them as "our fit to NanoBench data", not as values quoted from the paper.
Also open: `ROTOR_DRAG_XYZ` kz = 0.7 has no published support (Faessler et al. 2018 identify
only d_x and d_y and set d_z = 0).

**Update 2026-11-15 (traced in the predecessor repository `rl-drone-flight-simulator`).**
- `PH_C_L = 2.618e-8` is our own fit: median of `m g / (4 Omega^2)` over 9,179 hover samples from
  10 NanoBench trajectories (`identify_params.py`, `swift_physics_headless.py:351`). NanoBench logs
  PWM, not rotor speed; `Omega` came from the ESC map later rejected (R-1). The fit must be repeated.
- Motor tau 0.02 s is labelled "best guess" in `default_demo.py:600`: PROJECT_CHOICE.
- The torque coefficient is an assumed `c_l / 480` ratio: PROJECT_CHOICE.
- kz is now 0.5 (equal to kx, ky). Top speed is drag-limited: kz 0.7 gives 22.0 m/s, kz 0 gives
  513 m/s (`results/tables/quad_limits_repeatability.csv`).
- Ground effect is a linear ramp, not the Cheeseman & Bennett formula (report read 2026-11-15:
  ARC R&M 3021, eq. 2, p. 2).
- `docs/sim_validation.md` section 1 had a "Bitcraze measurement" column and a "step-response
  identification" claim that trace to nothing; both removed.
- The `wind` sources in `configs/quadrotor.yaml` called OU-process parameters "Dryden turbulence
  intensity, MIL-HDBK-1797"; relabelled PROJECT_CHOICE.
- **Traced 2026-11-15, all relabelled PROJECT_CHOICE** (predecessor repo): maximum motor speed
  2800 rad/s ("best guess, typical small-quad max motor speed", `default_demo.py:601`), motor
  efficiency 0.7 ("best guess", `default_demo.py:615`), the five domain-randomisation ranges
  (bare literals, `swift_rl_env.py:149-156`) and the aero clips (+-50 N, +-5 N, bare literals).
  The earlier "Bitcraze", "NanoBench spread" and "measured peak" source text was unsupported.
  Max motor speed fixes the thrust ceiling (thrust-to-weight 2.05), so the quad acceleration
  limits inherit that assumption; the repeatability study does not vary it.

### 28. Bridge rate limit of 12 rad/s can never bind
`attitude_to_rate` returns `clip(2 Kp q_err, +-limit)` with Kp = 6 1/s and |q_err| <= 1, so the
command never exceeds 12 rad/s. A sweep at 6 / 12 / 24 rad/s gave bit-identical episodes
(2026-11-15); 1.5 and 3 rad/s do change the outcome. `scripts/sweep_bridge_rate.py` therefore
sweeps 1.5 / 3 / 6 / 12. The limit is redundant at its nominal value, not wrong. Note also that
the measured peak body rate in a hard attitude step is 4.4 rad/s.

### 29. Closed-loop `accel_tracking_rms` is implausibly large (OPEN)
The bridge-rate smoke test reported a median acceleration-tracking RMS of 31-78 m/s^2 on a
vehicle whose limit is 15.4 m/s^2. Likely causes: the APN command is compared before the bridge's
thrust/tilt saturation, and the "actual" acceleration is a 50 Hz finite difference of velocity.
Not investigated; do not quote this metric until it is.

### 30. RL hyperparameter sources were mislabelled (FIXED 2026-11-15)
`configs/training.yaml`, `training/algos.py`, `docs/algorithm_selection.md`, decision D024 and
the audit regex called the PPO/SAC/TD3 hyperparameters the defaults of the per-environment tuned
hyperparameter files of the RL Baselines3 collection. They are the plain Stable-Baselines3 2.9.0
constructor defaults. Every value was then checked against the installed SB3 source and the
papers (PDFs read 2026-11-15) and now names its table/section and SB3 file:line. Errors found
while checking the interrupted 2026-11-15 rewrite:
- Andrychowicz et al. (2021) was cited as Sec. 3.5 for network width; that section is "Training
  setup". The network findings are in Sec. 3.2, pp. 5-6. Fixed in `training.yaml`,
  `ablations.yaml`, `algorithm_selection.md`.
- SAC `train_freq: 1` was cited to "Table 1 target update interval 1", which is a different SB3
  argument (`target_update_interval`). Now cited to Haarnoja et al. (2018) Sec. 4.2 (one
  environment step per iteration).
- TD3 `learning_rate: 3e-4` was labelled as cited; it matches neither the paper (1e-3, Sec. 6.1)
  nor SB3 (1e-3, td3.py L84). Now PROJECT_CHOICE (authors' code sfujim/TD3 TD3.py L83/L87 uses
  3e-4; equal to PPO/SAC so P6.4 compares algorithms at one step size).
- TD3 `learning_starts: 100` (SB3) differs from the paper's 1000/10000 random steps; now stated.
`tests/unit/test_algos.py::test_values_equal_sb3_defaults_except_project_choices` now fails if a
YAML value drifts from the installed SB3 default. No result is stale: only labels changed.

### 31. TD3 explored without action noise (FIXED 2026-11-15)
SB3's `TD3` adds exploration noise only when an `ActionNoise` object is passed; `build_model`
passed none, so after the 100 random warm-up steps TD3 acted deterministically. Fujimoto et al.
(2018) Sec. 6.1 add N(0, 0.1) to every action (sfujim/TD3 main.py L44, L122). `training.yaml`
now has `td3.action_noise_std: 0.1` and `build_model` builds `NormalActionNoise` with
sigma = 0.1 x half the action range (= 0.1 for our [-1, 1] actions). Test:
`test_td3_gets_gaussian_exploration_noise`. Stale: any TD3 run made before this commit. None
exists (P6.4 not yet run; `results/` holds no algorithm-selection tables), so nothing has to be
re-run; `results/tables/training_throughput.json` TD3 steps/s may shift slightly (noise sampling
is cheap) but it is a laptop benchmark only.

### 32. Fixed-wing interceptor limits do not follow from the cited model (OPEN, owner decision)
Item 3 of the 2026-11-15 sourcing pass compared every Aerosonde value with mavsim_public
(`docs/aerosonde_mavsim_check.md`): the 53 airframe numbers are correct. The derived limits are
not. With the book propulsion model (Beard & McLain uavbook draft 2025-10-02, eqs. 4.17-4.22)
and the mavsim parameters, full throttle at sea level gives:

| limit | configured | model | source text before this fix |
|---|---|---|---|
| `v_max` | 40 m/s | 32.4 m/s (T = D root) | "thrust = drag root" |
| `a_t_max` | 41.2 m/s^2 (4.2 g) | 2.5 m/s^2 at 25 m/s; <= 7.7 m/s^2 static | "excess thrust" |
| `a_t_min` | -19.6 m/s^2 (-2 g) | -3.0 m/s^2 at 25 m/s | "idle thrust / drag" |
| `tau_accel` | 0.12 s | not in the book | "Beard & McLain control chapter fit" |

The 40 m/s is hard-coded in the v1 notebook (`archive/path_planning_updated_project/
apn_rl_interceptor_path_planning.ipynb`, `v_max=40.0`); nothing derived it. `v_min = 15 m/s`
uses CL_max = 1.45, a project choice (the book lift curve peaks at 2.42, giving 11.3 m/s).
The audit had labelled the `platforms.yaml` copies CITED because its Beard regex matched the word
"aerosonde" in a file path; fixed. Values are **unchanged** and relabelled PROJECT_CHOICE
because they are platform values (owner decision). If they are changed to the model values, the
fixed-wing interceptor (32.4 m/s) is only 1.06x faster than the Orlan-10 target (30.6 m/s), and
every fixed-wing result and the `platforms.yaml` observation scales (40 m/s) become stale.
Also fixed: `docs/equations_reference.md` still showed n_max = 10.5 and j = 858 m/s^3.

### 33. Secondary-only data rows: primary sources searched (2026-11-15)
Item 4 of the sourcing pass. Upgraded to the Bharat Dynamics manufacturer PDF
(bdl-india.in Detailsofbusiness.pdf): Akash (I01; speed Mach 1.8-2.5, range 3-25 km, ceiling
20 km) and Astra Mk1 (I04, S03; Mach 4.5, ceiling 20 km). Correction: the Astra active-radar
homing range is **13 km** (BDL), not the 25 km taken from Wikipedia; the missile `r_init_max`
cross-check (7 km) still holds. The audit now sets CITED when every data row a value is tagged
with has a primary tier. Still secondary, with the reason recorded per row: IRIS-T SLM speed
(Diehl's brochure gives no speed), JF-17 (pac.org.pk refused connections), J-20 (no published
figures), Orlan-10 (no manufacturer datasheet online), Shahed-136 (IRGC poster via Janes),
Akash radar ranges (no LRDE datasheet). No model value changed, so nothing is stale.

