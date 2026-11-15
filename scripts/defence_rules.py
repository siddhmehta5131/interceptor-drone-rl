"""Defence of every PROJECT_CHOICE value (sourcing pass item 6, 2026-11-15).

A project choice has no external source by definition, so the audit must say what
tests it.  ``defence_for(loc, name, value)`` returns ``(category, text)``; the audit
writes it to the ``defence`` column and refuses to finish if a PROJECT_CHOICE row
has none.  Categories:

  SWEEP          an evaluation sweep varies this value: script / grid named
  ABLATION       a training ablation group in configs/ablations.yaml varies it
  REPEATABILITY  scripts/check_quad_limits_repeatability.py varies it
  PROTOCOL       evaluation protocol (episode counts, seeds, budgets); statistical argument
  DEFINITION     defines the experiment (scenario, test signal, grid, interface); stated, not tuned
  BOUND          a physical or logical bound shows results do not depend on the exact value
  NUMERICAL      numerical guard / resolution far from the signal scale
  UNUSED         not executed by any v2 entry point, so no result depends on it
  PRESENTATION   plotting / file output / parallelism only
  PENDING        disagrees with a source; owner decision open (BUGS #) - plus what tests it meanwhile

Every sweep or group named here exists: interceptor/eval/sweeps.py SWEEPS (run by
scripts/sweep_robustness.py, notebook 05), scripts/sweep_nav.py N_GRID,
scripts/sweep_bridge_rate.py RATE_LIMITS, configs/ablations.yaml groups,
scripts/check_quad_limits_repeatability.py cases.
"""
from __future__ import annotations

import re

W100 = "100 episodes per cell: Wilson 95 % half-width at most 9.6 percentage points (p = 0.5; docs/mathematics.md section 11.1)"
W80 = ("5 episodes per manoeuvre x 16 manoeuvres = 80 per grid point (notebook 05 passes --episodes 5): "
       "Wilson 95 % half-width at most 10.7 points")
SEEDS = ("evaluation seeds start at 10000 with stride 1000 so they never overlap the training seeds 0, 1, 2 "
         "(plan section 6); any fixed disjoint value is equivalent")
SHAPE = ("SWEEP maneuver_shape (interceptor/eval/sweeps.py, x0.5 / 1 / 1.5): this constant is multiplied by "
         "shape_scale s in targets.py; manoeuvre amplitudes are fractions of a_max, covered by the "
         "target_accel_ratio sweep (0.1-0.6 of the interceptor a_n_max)")
LEGACY_MAN = ("not executed by v2: v2 uses only the ManeuverType / PlatformType enums of maneuvers.py; this "
              "notebook-v2 manoeuvre library (EvasiveTargetGenerator, presets, PN interceptor) is used only by "
              "scripts/make_fixtures.py for the fixture-parity tests (checked by reference search and "
              "results/tables/liveness.json, 2026-11-15); the live target model is guidance/targets.py")
FW32 = ("owner decision open (BUGS #32: the book propulsion model gives 32.4 m/s, 0.25 g, -0.31 g). Tested "
        "meanwhile by sweeps.py axial_limit_scale (x0.06 = the model a_t values), interceptor_lag_scale "
        "(x0.5 / 1 / 2), jerk_limit_scale (x0.5 / 1 / 2) and target_speed_ratio (0.3-1.1; the model case "
        "30.6 / 32.4 = 0.94 lies inside the grid)")

# (category, path prefix, regex on "name = value" or None, text); first match wins
RULES = [
    # --- presentation / parallelism -------------------------------------------------
    ("PRESENTATION", "", r"savefig|\.grid\(|set\(ylim|subplots|\.bar\(|DictWriter|json\.dumps|fieldnames|"
                         r"i % 10 == 0|'--workers'", "plot, file output or worker count; each episode's seed is fixed by its job index, so results are identical for any value"),
    # --- open owner decisions ---------------------------------------------------------
    ("PENDING", "configs/airframes/aerosonde", r"v_max|a_t_max|a_t_min|tau_accel", FW32),
    ("PENDING", "configs/platforms", r"fixed_wing\.(v_max|a_t_max|a_t_min|tau_accel|max_jerk)", FW32),
    ("PENDING", "configs/quadrotor", r"torque_coefficient",
     "owner decision open (BUGS #34: Forster 2015 eq. 3.5 gives ~3x the yaw torque per newton); affects yaw authority only; varied +-15 % by the c_d_frac domain randomisation, ablated by configs/ablations.yaml fc_randomisation"),
    ("PENDING", "configs/quadrotor", r"motor_time_constant",
     "owner decision open (BUGS #34: Forster 2015 pole gives 65-78 ms); REPEATABILITY motor-lag x0.5 / x2 rows of results/tables/quad_limits_repeatability.csv (envelope moves 4.1 % / 5.9 %)"),
    ("PENDING", "interceptor/sim/constants", r"^PH_K_MOT\b",
     "owner decision open (BUGS #34); REPEATABILITY motor-lag x0.5 / x2 rows (results/tables/quad_limits_repeatability.csv)"),
    ("PENDING", "interceptor/sim/constants", r"^PH_(OMEGA_MAX|ETA)\b",
     "'best guess' in the predecessor simulator (BUGS #27); sets the thrust ceiling; owner decision open (BUGS #34: Forster 2015 gives max 0.160 N per motor at 3,052 rad/s vs our 0.205 N); target_accel_ratio sweep reports results relative to the resulting a_n_max"),
    # --- configs -----------------------------------------------------------------------
    ("ABLATION", "configs/training", r"^network\.",
     "ABLATION configs/ablations.yaml network group (2 x 64 tanh, 2 x 256 ReLU vs nominal), notebook 05"),
    ("DEFINITION", "configs/training", r"td3\.learning_rate",
     "convention: the TD3 authors' own code value (sfujim/TD3 TD3.py L83, L87) and equal to the PPO / SAC step size, so P6.4 compares algorithms at one step size; not swept (training cost)"),
    ("PROTOCOL", "configs/training", r"^budgets\.",
     "equal environment-step budget for every algorithm (P6.4) sized from the throughput benchmark results/tables/training_throughput.json to fit Colab; learning curves (notebook 02) show whether it saturates"),
    ("PROTOCOL", "configs/training", r"^seeds\.training",
     "3 seeds set by the Colab budget; Agarwal et al. 2021 caveats recorded at decision D030; results reported per seed with bootstrap CIs"),
    ("PROTOCOL", "configs/training", r"^seeds\.", SEEDS),
    ("DEFINITION", "configs/training", r"^selected_algo",
     "decision rule P6.2 stated before the runs (docs/algorithm_selection.md section 6); provisional until notebook 02 (Colab)"),
    ("DEFINITION", "configs/ablations", None,
     "this row is a variant of the ablation grid itself (on/off, or 0.5x / 1.5x around the nominal guidance_ablations.full); it defines the experiment"),
    ("BOUND", "configs/fc", r"safety_rate",
     "termination threshold 5x the rate limit: only ends episodes that have already diverged, far above any tracked setpoint"),
    ("DEFINITION", "configs/fc", r"init_", "initial-condition distribution of the flight-controller task (30 deg tilt, quarter of rate_max), stated in docs; the same for every controller compared"),
    ("DEFINITION", "configs/fc", r"overshoot_max", "acceptance target for the B1 baseline design (Ogata step-response definitions); a pass/fail criterion, not a model parameter"),
    ("SWEEP", "configs/guidance", r"N_prime",
     "SWEEP scripts/sweep_nav.py N_GRID 3, 3.5, 4, 4.5, 5 (P5.11; results/tables/nav_sweep.csv, rerun in Colab); lower bound cited (Palumbo et al. 2010)"),
    ("BOUND", "configs/quadrotor", r"ground_effect",
     "linear ramp active only below 0.15 m; guidance episodes start at alt_init and a crash (z < 0) ends them, so no interception result depends on it (BUGS #27)"),
    ("UNUSED", "configs/quadrotor", r"^wind\.",
     "the Ornstein-Uhlenbeck wind (sim/aero.py sample_wind_params, wind_ou_step) has no caller in v2: the FC env and all evaluations use the Dryden model (sim/wind.py)"),
    ("SWEEP", "configs/seeker", r"sigma|dropout",
     "SWEEP noise_scale (interceptor/eval/sweeps.py, x0, 0.25, 0.5, 1, 2, 4): seeker.py _sigmas multiplies this term and the dropout probability by noise_scale"),
    ("SWEEP", "configs/seeker", r"ema_alpha",
     "SWEEP seeker_gain_scale (interceptor/eval/sweeps.py, x0.5 / 1 / 1.5, capped at 1): the env uses this alpha-beta-gamma filter (seeker.py default filter_type 'abg'); the P5.4 Kalman tracker is not wired into the env (BUGS #13)"),
    ("PENDING", "configs/airframes/aerosonde", None, FW32),
    # --- modelling decisions (docs/decision_register.csv) --------------------------------
    ("SWEEP", "guidance", r"Navigation constant", "SWEEP scripts/sweep_nav.py N_GRID 3-5 (P5.11), per platform and arm"),
    ("SWEEP", "guidance", r"Seeker noise", "SWEEP noise_scale x0-4 and seeker_gain_scale x0.5-1.5 (interceptor/eval/sweeps.py)"),
    ("SWEEP", "guidance", r"manoeuvre library", SHAPE),
    ("DEFINITION", "guidance", r"Missile flight condition", "engagement condition of the missile case (Q12), inside the CADAC AIM5 envelope; the target speed relative to it is varied by target_speed_ratio"),
    ("ABLATION", "rl", r"Reward weights", "ABLATION configs/ablations.yaml reward_weights (kill_bonus, w_progress, w_res_rl at 0.5x / 1.5x)"),
    ("PROTOCOL", "rl", r"Seeds and episodes", "3 seeds (Colab budget, Agarwal 2021 caveats at D030); " + W100),
    ("DEFINITION", "fc", r"interface", "collective thrust + body-rate setpoints = the attitude-rate controller level of the Crazyflie firmware (cf_controllers); B1, B2 and the RL controller all use it"),
    # --- training / evaluation scripts ------------------------------------------------
    ("PROTOCOL", "scripts/", r"'--episodes'.*= 100$", W100),
    ("PROTOCOL", "scripts/sweep_robustness", r"'--episodes'", W80),
    ("PROTOCOL", "scripts/sweep_bridge_rate", r"'--episodes'",
     "episodes per rate value per arm, manoeuvres cycled and the same seeds at every rate (paired comparison); the notebook-05 value 5 is a coarse check (Wilson half-width up to 33 points), the full-count run is Colab work (docs/tuned_values_plan.md)"),
    ("PROTOCOL", "scripts/", r"EVAL_SEED0|'--seed'.* = (EVAL_SEED0|10000)$|seed0", SEEDS),
    ("PROTOCOL", "scripts/sweep_nav", r"'--seed'|5000", "nav-sweep seeds 5000+ are disjoint from the training seeds 0-2 and the evaluation seeds 10000+, so N' is not selected on the evaluation episodes"),
    ("DEFINITION", "scripts/sweep_nav", r"N_GRID", "the sweep grid itself: brackets the cited lower bound 3 to the project upper bound 5 (docs/mathematics.md section 1.5)"),
    ("DEFINITION", "scripts/sweep_bridge_rate", r"RATE_LIMITS", "the sweep grid itself: nominal 12 rad/s and 1/2, 1/4, 1/8 of it; higher values cannot act (BUGS #28)"),
    ("DEFINITION", "scripts/sweep_robustness", r"radii|linspace", "kill-radius grid of the P9.6 sensitivity plot (25 radii from 0.25x to 3x the nominal radius); recomputed from stored miss distances, no new episodes"),
    ("NUMERICAL", "scripts/sweep_nav", r"percentile", "75th and 25th percentiles define the interquartile range (a definition)"),
    ("DEFINITION", "scripts/evaluate_fc", r"DT|N_STEPS|rate_max|seed|sign|sequences|range\(50\)|setpoints",
     "flight-controller evaluation protocol: 500 Hz control (Crazyflie ATTITUDE_RATE, stabilizer_types.h), 1.5 s sequences, 50 sequences per signal kind on fixed seeds 7000+, identical for B1 / B2 / RL; stated in docs/implementation_plan.md P4"),
    ("DEFINITION", "scripts/evaluate_fc", r"DrydenWind", "evaluation wind: Dryden model at 100 m altitude (MIL-F-8785C form); the same disturbance for every controller compared"),
    ("DEFINITION", "scripts/", r"noise_scale=\) = 1\.0|done = False",
     "nominal setting (noise x1) / loop flag; the noise level itself is tested by the noise_scale sweep"),
    ("PROTOCOL", "scripts/", r"'--episodes'", W80),
    ("REPEATABILITY", "scripts/measure_quad_limits", None,
     "measurement-procedure setting: REPEATABILITY scripts/check_quad_limits_repeatability.py reruns the P4.9 measurement with two other procedure settings (envelope moves 5.1 % / 5.4 %; results/tables/quad_limits_repeatability.csv)"),
    ("PROTOCOL", "scripts/", None, "evaluation / analysis setting of a reporting script; does not enter the simulated dynamics"),
    # --- library code ------------------------------------------------------------------
    ("DEFINITION", "interceptor/guidance/maneuvers", r"^[A-Z_]+ = '[a-z_]+'$",
     "enum identifier (names a platform or manoeuvre law); the 16 laws are listed in docs/mathematics.md section 5 and each is evaluated separately"),
    ("DEFINITION", "interceptor/guidance/maneuvers", r"^G_ACCEL", "gravity 9.81 m/s^2 (standard gravity rounded, as mavsim L44)"),
    ("UNUSED", "interceptor/guidance/maneuvers", None, LEGACY_MAN),
    ("BOUND", "interceptor/guidance/targets", r"floor_altitude|\[2\] = -0\.5|p_W\[2\] > 100",
     "altitude floor 35 m with a damped bounce keeps the target airborne (it is not a ground target); the dive law only acts above 100 m; a geometric guard, not a performance parameter"),
    ("NUMERICAL", "interceptor/guidance/targets", r"max\(kin\.r, 50\.0\)|max\(speed_T, 5\.0\)|1e-06|1e-6",
     "guard against division by a small range / speed in the manoeuvre law; active only inside 50 m / below 5 m/s"),
    ("DEFINITION", "interceptor/guidance/targets", r"shape_scale=\)=1\.0",
     "nominal point of the maneuver_shape sweep (x0.5 / 1 / 1.5)"),
    ("DEFINITION", "interceptor/guidance/targets", r"rng\.random\(\) > 0\.5|split_s_completed",
     "random turn direction (symmetric 50 / 50) and the split-S completion flag: definitions of the manoeuvre, not magnitudes"),
    ("DEFINITION", "interceptor/guidance/targets", r"pos = self\.p_W \+ self\.v_W",
     "constant-acceleration extrapolation p + v dt + a dt^2 / 2 (kinematics, not a tuning value)"),
    ("SWEEP", "interceptor/guidance/targets", r"1\.3 \* self\.v_nominal",
     "speed-change law tops out at 1.3 v_nominal; v_nominal itself is varied by the target_speed_ratio sweep (0.3-1.1 of v_max)"),
    ("SWEEP", "interceptor/guidance/targets", None, SHAPE),
    ("SWEEP", "interceptor/eval/sweeps", r"SWEEPS",
     "the sweep grid itself (defines the experiment); each grid brackets the nominal value 1 (or the nominal ratio)"),
    ("DEFINITION", "interceptor/eval/sweeps", r"noise_scale=|shape_scale=",
     "nominal point / swept value passed through; defined by the SWEEPS grid"),
    ("PROTOCOL", "interceptor/eval/sweeps", r"episodes=\) = 5", W80),
    ("DEFINITION", "interceptor/eval/sweeps", r"replace\(ema_alpha", "filter gains capped at 1.0 (1 = raw measurement, no smoothing): the admissible range of an alpha-beta-gamma gain"),
    ("PROTOCOL", "interceptor/eval/sweeps", r"seed0", SEEDS),
    ("SWEEP", "interceptor/guidance/env", r"noise_scale=\)|maneuver_shape_scale",
     "nominal point of the noise_scale / maneuver_shape sweeps (interceptor/eval/sweeps.py)"),
    ("SWEEP", "interceptor/guidance/env", r"N_nav=\)",
     "SWEEP scripts/sweep_nav.py (P5.11); NOTE BUGS #35: training, evaluate_guidance_arms and the robustness sweeps use this default 4.0, not the per-platform sweep winner"),
    ("DEFINITION", "interceptor/guidance/env", r"mode=\)|target_set=\)|polynomial_order|legacy_reward",
     "default arm / target set / polynomial order selection; every arm and target set is evaluated explicitly (scripts/evaluate_guidance*.py)"),
    ("DEFINITION", "interceptor/guidance/env", r"weights",
     "training curriculum: hard manoeuvres sampled 2.5x as often; every evaluation runs each of the 16 manoeuvres separately (fixed_maneuver), so reported per-manoeuvre kill rates do not depend on it"),
    ("DEFINITION", "interceptor/guidance/env", r"action_space|observation_space|act = |a_rl",
     "action normalised to [-1, 1] and scaled by a_n_max (residual authority tested by ablations.yaml sensitivity.authority_0.20 / 0.35 / 0.50); observation clipped to +-10 (P5.8 bound test)"),
    ("DEFINITION", "interceptor/guidance/env", r"\baz\b|\bel\b|pitch|v_mag|init_speed|p_T0|end",
     "engagement scenario (spawn cone +-35 deg az, +-15 deg el; target speed 0.75-1.15 v_nominal; interceptor 0.65-0.85 v_max; altitude floor 50 m), identical for every arm compared; the speed ratios are varied by the target_speed_ratio sweep"),
    ("BOUND", "interceptor/guidance/env", r"diverged",
     "stops an episode only after closest approach has passed (closing speed below -0.2 v_max and range 60 m beyond the recorded minimum), so the minimum miss distance is already final"),
    ("NUMERICAL", "interceptor/guidance/env", r"r_min_cl|kin_cmd|tf_p",
     "t_go guard: range clamped at 2 m in the APN time-to-go (BUGS #2), active only inside the last 2 m"),
    ("DEFINITION", "interceptor/guidance/observation", r"los_rate=|target_accel=|load_observation_scales",
     "observation normalisation divisor; any divisor that keeps |obs| <= 10 is equivalent up to network initialisation (Andrychowicz 2021 Sec. 3.3: normalise observations); bound tested in P5.8"),
    ("ABLATION", "interceptor/guidance/observation", r"m=\)|p=\)|n=\)|q=\)|maxlen",
     "history / prediction lengths of the observation; ABLATION configs/ablations.yaml guidance_ablations.no_history and no_prediction remove these blocks"),
    ("NUMERICAL", "interceptor/guidance/observation", None,
     "guard (range floor 15 m for the LOS rate, closing-speed floor 0.1 m/s for t_go) and the +-10 clip of the observation bound"),
    ("DEFINITION", "interceptor/guidance/polynomial_path", r"speed_cap=\)|shape=\)",
     "constructor default; the env always passes speed_cap = 1.15 v_target_nominal and shape 'random'"),
    ("NUMERICAL", "interceptor/guidance/polynomial_path", r"_PEAK_SAMPLES|range\(64\)",
     "sampling resolution (201 points) / rejection attempts (64) for drawing a monotone path; a fallback shape is used if all fail"),
    ("DEFINITION", "interceptor/guidance/polynomial_path", None,
     "defines the non-evasive target family (monotone cubic / quadratic paths, coefficients in [-0.8, 0.8] and [-1, 1]) and its exact derivatives; stated in docs/mathematics.md"),
    ("DEFINITION", "interceptor/bridge/closed_loop", r"CONTROL_DT|PHYSICS_DT",
     "500 Hz control / 1000 Hz physics = Crazyflie ATTITUDE_RATE / RATE_MAIN_LOOP (stabilizer_types.h, cf_rates); RK4 order verified by tests/unit/test_sim_timestep.py"),
    ("BOUND", "interceptor/bridge/closed_loop", r"GUIDANCE_DT",
     "50 Hz guidance step (decision S8): 20 ms is at most 0.2 of the smallest interceptor lag tau_accel in configs/platforms.yaml (0.1 s missile), so the guidance update is faster than the vehicle can respond"),
    ("SWEEP", "interceptor/bridge/closed_loop", r"rate_max|bridge_rate_limit",
     "SWEEP scripts/sweep_bridge_rate.py RATE_LIMITS 1.5 / 3 / 6 / 12 rad/s (BUGS #28: higher values cannot act)"),
    ("NUMERICAL", "interceptor/bridge/closed_loop", r"r_min_cl|^kin",
     "t_go guard: range clamped at 2 m in the APN time-to-go (BUGS #2), active only inside the last 2 m"),
    ("DEFINITION", "interceptor/bridge/closed_loop", r"np\.clip\(np\.asarray\(act",
     "policy action clipped to its [-1, 1] box and mapped to motor command [0, 1] (same map as the FC env)"),
    ("SWEEP", "interceptor/bridge/closed_loop", r"N_nav=\)", "SWEEP scripts/sweep_nav.py; the evaluation scripts pass resolve_n_prime() (per-platform sweep winner)"),
    ("SWEEP", "interceptor/bridge/closed_loop", r"noise_scale", "nominal point of the noise_scale sweep"),
    ("DEFINITION", "interceptor/bridge/closed_loop", r"DrydenWind|\bw\b",
     "closed-loop disturbance: Dryden wind at 100 m altitude, mean (1, 0.5, 0) m/s, W20 = 3 m/s, fixed per seed; identical for every arm compared"),
    ("DEFINITION", "interceptor/bridge/closed_loop", None,
     "default argument / frame count of the closed-loop runner; the evaluation scripts pass the arm, inner controller, target set and seeds explicitly"),
    ("DEFINITION", "interceptor/fc/setpoints", None,
     "test-signal definition of the flight-controller evaluation (step, ramp, doublet, sinusoid, chirp 0.2-5 Hz, band-limited random); part of the evaluation protocol and identical for every controller"),
    ("DEFINITION", "interceptor/fc/rollout", None,
     "rollout defaults: 500 / 1000 Hz rates (Crazyflie firmware), full battery 4.2 V (U_BAT_FULL), start at 10 m; the evaluation protocol"),
    ("NUMERICAL", "interceptor/fc/thrust", r"omega", "square-root guard for zero or negative thrust"),
    ("DEFINITION", "interceptor/fc/thrust", r"g=\)", "gravity 9.81 m/s^2 (standard gravity rounded, as mavsim L44); same constant as the simulator"),
    ("ABLATION", "interceptor/fc/env", r"domain_randomization",
     "ABLATION configs/ablations.yaml fc_randomisation.with_dr / without_dr (P9.5)"),
    ("DEFINITION", "interceptor/fc/env", r"FRAME|N_FRAMES|_push|om =|e =|\bf\b|Om / PH_OMEGA",
     "flight-controller observation layout (3 frames of rate, rate error, collective error, previous action; privileged block normalised by nominal values, body velocity by 5 m/s); adequacy tested end to end against the B1 / B2 PID baselines (scripts/evaluate_fc.py)"),
    ("DEFINITION", "interceptor/fc/env", r"wind|mean",
     "training disturbance: Dryden wind with per-episode mean N(0, 1 m/s); the evaluation uses fixed Dryden wind (scripts/evaluate_fc.py)"),
    ("BOUND", "interceptor/fc/env", r"rew",
     "terminal penalty applied only when |omega| exceeds safety_rate, i.e. to an episode that has already failed"),
    ("DEFINITION", "interceptor/fc/env", None,
     "flight-controller task definition (initial state 100 m with N(0, 1) m/s velocity, hover rotor speed, action map [-1, 1] -> [0, 1], thrust-error term); identical for every controller trained"),
    ("ABLATION", "interceptor/sim/constants", r"DR_RANGES",
     "ABLATION configs/ablations.yaml fc_randomisation.with_dr / without_dr tests whether randomisation helps; the widths themselves are not swept (note: our c_l differs from Forster 2015 by 44 %, outside the +-15 % range; BUGS #34)"),
    ("UNUSED", "interceptor/sim/constants", r"WIND_", "OU wind parameters: sample_wind_params has no caller in v2 (Dryden wind is used everywhere)"),
    ("BOUND", "interceptor/sim/constants", r"GROUND_EFFECT",
     "inactive above 0.15 m; no interception episode flies that low (BUGS #27)"),
    ("REPEATABILITY", "interceptor/sim/constants", r"ROTOR_DRAG",
     "REPEATABILITY kz = 0 / 0.7 rows of results/tables/quad_limits_repeatability.csv (top speed is drag-limited: 513 / 22.0 m/s vs 27.7); the speed ratio is kept by tying the target to the interceptor and swept by target_speed_ratio"),
    ("BOUND", "interceptor/sim/constants", r"TILT_CRASH|GROUND_CONTACT",
     "crash classification thresholds (60 deg tilt, 1 m/s ground contact) used only to end flight-controller episodes that have already failed"),
    ("DEFINITION", "interceptor/sim/constants", r"PH_DT\b|PH_CMD|PH_C_SCALE|PH_W_SCALE|PH_CUT_THR|_hover_cmd|brentq|om = PH_BAT|PH_OMEGA_HOVER|PH_R_P|PH_ZETA|PH_SPIN",
     "simulator convention: normalised command range [0, 1], stick scales and throttle cut of the predecessor firmware interface, 100 Hz fixture-replay default dt (v2 passes 500 / 1000 Hz explicitly), X-frame rotor geometry and spin signs, hover command solved by root finding on [0.02, 1]"),
]
RULES = [(c, p, re.compile(rx) if rx else None, t) for c, p, rx, t in RULES]


def defence_for(loc: str, name: str, value: str):
    """(category, text) for a PROJECT_CHOICE row, or ('', '') if no rule applies."""
    path = loc.rsplit(":", 1)[0].split("#")[0]
    key = f"{name} = {value}"
    for cat, pre, rx, text in RULES:
        if path.startswith(pre) and (rx is None or rx.search(key) or rx.search(name)):
            return cat, text
    return "", ""
