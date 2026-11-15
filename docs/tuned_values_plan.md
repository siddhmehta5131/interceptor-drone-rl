# Tuned values: what needs defending and how

Generated from `docs/full_value_audit.csv`; last updated 2026-11-15. "Tuned
value" = audit status `PROJECT_CHOICE`: our own design or tuning number with no
external source. No citation exists for these, so the defence is a sweep, a
statistical argument, or a clear statement that the value is part of the
experiment's definition.

## Count history (743 → 435 → 415 → 449)

- **743 → 435.** `scripts/trace_liveness.py` ran tiny versions of every v2 entry
  point and recorded every executed line (`results/tables/liveness.json`). Whole
  files that executed nothing (the superseded 3-stage pipeline, plan P1.10) held
  320 values.
- **435 → 415.** Those files were moved to `archive/legacy_3stage/` and are no
  longer audited, so the audit has no legacy rows. The rules that referred to
  them were removed from `scripts/reference_library.py` on 2026-11-15.
- **415 → 449.** Not new tuning of the system: the four defence studies below
  were written, and their own grids and settings are counted (reward-weight
  ablation group, manoeuvre-shape sweep, bridge-rate sweep, repeatability
  settings). In addition several simulator constants that used to carry a
  citation they did not deserve are now honestly labelled project choices
  (motor time constant, torque coefficient, ground-effect ramp, the three
  Ornstein-Uhlenbeck wind parameters).

## The 449, by family

| Family | Values | Defence | Status |
|---|---|---|---|
| **A. Target manoeuvres and scenarios** (`guidance/maneuvers.py`, `guidance/targets.py`) | 144 | Amplitudes are fractions of `a_max`, so the P9.1 `target_accel_ratio` sweep covers them. Shape constants (frequencies, switch times, mixing fractions) are scaled together by 0.5 / 1 / 1.5 in the `maneuver_shape` sweep (`eval/sweeps.py`, env kwarg `maneuver_shape_scale`). | Code done; run in Colab (notebook 05) |
| **B. Guidance env, observation, reward** (`guidance/env.py`, `guidance/observation.py`, `guidance/polynomial_path.py`) | 52 | `N′`: P5.11 `sweep_nav` (rerun for the new envelope). Residual authority 0.35 and γ: P9.3. Reward weights: `reward_weights` group in `configs/ablations.yaml` (6 variants, env kwarg `reward_scale`). Observation divisors: any scale that keeps \|obs\| ≤ 10 is equivalent up to network initialisation; the bound is tested (P5.8). | Code done; reward sweep needs training (Colab) |
| **C. Flight controller, bridge, simulator constants and domain randomisation** (`sim/constants.py`, `fc/*`, `bridge/closed_loop.py`, `configs/fc.yaml`, `configs/quadrotor.yaml`) | 104 | DR ranges: P9.5. Setpoint and test-signal definitions are part of the evaluation protocol (state them). Bridge rate limit: `scripts/sweep_bridge_rate.py` at 1.5 / 3 / 6 / 12 rad/s. The limit cannot act above 12 rad/s because the rate command is `2 Kp q_err` with Kp = 6 1/s (a 24 rad/s run is bit-identical to 12), so only lower values are swept. Motor lag: repeatability rows ×0.5 / ×2. | Code done; bridge sweep smoke-tested only |
| **D. Seeker noise** (`configs/seeker.yaml`) | 7 | P9.1 `noise_scale` ∈ {0, 0.25, 0.5, 1, 2, 4}. The angular term is now sourced (EchoShield < 0.5° = 8.727 mrad, used as 1σ); ×2 is EchoGuard, ×0.25 is close to the old 2.5 mrad (`docs/mathematics.md` §5A.1). | Covered once the sweep is rerun |
| **E. Evaluation protocol and budgets** (episode counts, seeds, sweep grids, training budgets, ablation grid) | 110 | Not tuned. Episode counts follow from the Wilson 95 % half-width (100 episodes gives at most ±9.6 points, at p = 0.5); budgets come from the throughput benchmark; grids define the experiment. Stated in `docs/mathematics.md` §11. | Argument written |
| **F. Measurement procedure** (`scripts/measure_quad_limits.py`) | 32 | Repeatability check done (`scripts/check_quad_limits_repeatability.py`, `results/tables/quad_limits_repeatability.csv`): two alternative settings move the envelope by 5.1 % and 5.4 % (top speed, through the hold fraction), slightly over the 5 % target. | Done; result reported as found |

## Result of the repeatability check (family F)

| Case | v_max (m/s) | lateral a (m/s²) | Largest change vs baseline |
|---|---|---|---|
| baseline (kz = 0.5) | 27.7 | 15.4 | - |
| procedure settings A | 26.2 | 15.4 | 5.1 % |
| procedure settings B | 29.2 | 15.4 | 5.4 % |
| motor lag ×0.5 | 27.3 | 15.1 | 4.1 % |
| motor lag ×2 | 28.9 | 16.0 | 5.9 % |
| kz = 0.7 (previous value) | 22.0 | 14.8 | 20.5 % |
| kz = 0 | 513 | 17.1 | unbounded |

The top speed is drag-limited and the rotor drag is an assumption (Faessler et
al. 2018 identify `d_x`, `d_y` on a 610 g vehicle and no `d_z`). The absolute
quad speed should not be presented as a measured Crazyflie property; the study
keeps the interceptor/target speed ratio tied to real vehicles and sweeps it.

## Still to run (all in Colab)

1. `sweep_robustness.py` (noise, speed ratio, accel ratio, manoeuvre shape, interceptor lag, axial limit, jerk limit, seeker gains) on the new envelope.
2. `sweep_bridge_rate.py` at full episode count.
3. Reward-weight group (training).
4. `sweep_nav.py` rerun: `N′` was chosen on the old envelope.

## Defence of every project choice (item 6, 2026-11-15)

Every one of the 475 PROJECT_CHOICE rows (values and decisions) now carries a
`defence_category` and a `defence` text in `docs/full_value_audit.csv`, assigned by
`scripts/defence_rules.py`; `scripts/build_full_value_audit.py` exits with an error if any
PROJECT_CHOICE row has none. The text names the exact sweep grid, ablation group or
repeatability case, or states the non-sweep argument.

| Category | Rows | Meaning |
|---|---|---|
| SWEEP | 48 | an evaluation sweep varies it (`interceptor/eval/sweeps.py` SWEEPS, `sweep_nav.py`, `sweep_bridge_rate.py`) |
| ABLATION | 15 | a training group in `configs/ablations.yaml` varies it |
| REPEATABILITY | 31 | `scripts/check_quad_limits_repeatability.py` varies it |
| PENDING | 14 | disagrees with a source (BUGS #32, #34); owner decision, plus the sweep that tests it meanwhile |
| PROTOCOL | 25 | evaluation protocol: episode counts (Wilson half-width), disjoint seeds, budgets |
| DEFINITION | 176 | defines the experiment (scenario, test signal, grid, interface, convention); stated, not tuned |
| BOUND | 14 | a physical / logical bound shows no result depends on the exact value |
| NUMERICAL | 20 | numerical guard or resolution far from the signal scale |
| UNUSED | 97 | not executed by any v2 entry point (notebook-v2 manoeuvre library, OU wind) |
| PRESENTATION | 35 | plot, file output or worker count |

New evaluation-only sweeps added for this (run automatically by `scripts/sweep_robustness.py`,
notebook 05): `interceptor_lag_scale` (tau_accel x0.5 / 1 / 2), `axial_limit_scale` (a_t_max, a_t_min
x0.5 / 1 / 2, spanning the speed dependence of the fixed-wing model values, BUGS #32),
`jerk_limit_scale` (x0.5 / 1 / 2) and `seeker_gain_scale` (alpha-beta-gamma gains x0.5 / 1 / 1.5,
capped at 1). Not swept and stated as such: the domain-randomisation widths (the on/off ablation
exists), the FC observation layout (tested end to end against B1 / B2), the TD3 learning rate
(authors' code value). Open owner decisions: BUGS #32 (fixed-wing limits), #34 (quad motor
constants vs Forster 2015), #35 (which N' the RL arms use).
