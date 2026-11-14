# Simulator validation

This log records every parameter-resolution table, fit, cross-check and speed
measurement required by P2/P3. Values that are adopted in code point at the
source table and the equation IDs in `docs/mathematics.md`.

## 1. Quadrotor parameter resolution (P2.3; F6, F7, F8)

Corrected 2026-11-15. An earlier version of this table had a "Bitcraze
measurement" column and described the motor time constant as a "step-response
identification". Neither could be traced to any document or data file, so both
were removed, together with a `cf2x.urdf` column whose coefficient entries were
never checked against the file. The table below states only what was traced.

| Parameter | Adopted | Where the number comes from | Status |
|---|---|---|---|
| mass (kg) | 0.04085 | NanoBench, Ullah & Baca (2026), arXiv:2603.09908, Sec. IV-B1 and repository README: 27 g airframe plus markers and charging deck. | Cited (read 2026-11-14) |
| `Jxx = Jyy`, `Jzz` (kg m^2) | 2.3951e-5, 3.2347e-5 | Attributed to Förster (2015), ETH BSc thesis. | Thesis record confirmed; values not re-read. The thesis vehicle is a 27 g Crazyflie 2.0, so using its inertia at 40.85 g is an approximation. |
| arm (m) | 0.0397 | Attributed to the Bitcraze frame geometry and `cf2x.urdf`. | Not re-checked |
| thrust coefficient (N s^2/rad^2) | 2.618e-8 | **Our own fit**: median of `m g / (4 Omega^2)` over 9,179 hover samples from 10 NanoBench trajectories (`identify_params.py` and `swift_physics_headless.py:351` in the predecessor repository `rl-drone-flight-simulator`). | Own fit, to be repeated: NanoBench logs motor PWM, not rotor speed, so `Omega` came from the ESC map later rejected as non-monotone (BUGS R-1). The value appears nowhere in the NanoBench paper, README, code or metadata. |
| torque coefficient (N m s^2/rad^2) | 5.45e-11 | Assumed ratio `c_l / 480`; the yaw identification was too noisy to fit it. | Project choice; varied ±15 % by domain randomisation |
| motor time constant (s) | 0.02 | Labelled "best guess" in the predecessor simulator (`default_demo.py:600`). No step-response identification was done. | Project choice; defended by the motor-lag ×0.5 / ×2 rows of `results/tables/quad_limits_repeatability.csv` (largest envelope change 4.1 % / 5.9 %) |
| max motor speed (rad/s) | 2800 | Labelled "best guess, typical small-quad max motor speed" in the predecessor simulator (`default_demo.py:601`). The earlier "Bitcraze" text was unsupported. | Project choice; sets the thrust ceiling (thrust-to-weight 2.05) |
| ground effect | ramp to +20 % below 0.15 m | Linear thrust ramp `1 + 0.20 (1 - z/0.15)` (`EQ-SIX-10`). It is not the Cheeseman & Bennett formula `T_g/T_inf = 1/(1 - R^2/(16 Z^2))` (ARC R&M 3021, Sept. 1955, eq. 2, p. 2; scan read 2026-11-15), which is cited only for the existence of the effect. | Project choice; inactive above 0.15 m, so no interception result depends on it |

**What this means for the results.** The mass is cited. The thrust coefficient
sets thrust-to-weight (2.05) and so the acceleration limits; it rests on a fit
that used a rejected ESC map and must be repeated once rotor speed is available
(BUGS #27, #11). Top speed is set by the assumed rotor drag (section 5 and
`docs/mathematics.md` 6.1). The adopted values are in `configs/quadrotor.yaml`,
each with an honest `source` string.

## 2. Motor/ESC fit (P3.4; F9)

The anchored two-term fit is

```
Omega_ss(c) = 1170.36749799968 + 1629.63250200032 sqrt(c)   [rad/s]
```

It is monotone increasing over `c in [0, 1]`, matches the measured hover speed
`Omega_hover = 1956.211 rad/s` at `c = 0.232537`, and matches the full-throttle
speed 2800 rad/s at `c = 1`. The fit has two parameters for two measured anchor
points, so the residual at the anchors is zero by construction. The earlier
five-term fit was rejected because it was non-monotone (BUGS R-1). A
battery-voltage term is retained in the schema but is zero: the available data
does not separate voltage sag from throttle at fixed charge.

Figure: `docs/figures/esc_fit.png` (regenerate with
`python scripts/make_sim_figures.py`).

**Blocked sub-item.** The plan asks for a fit to a published PWM-to-thrust and
PWM-to-speed table with residuals. No such table is present in the repository
and the NanoBench export is not available on this host, so the fit remains the
two-anchor monotone calibration. This is a documented limitation, not a
silently accepted value: solving it needs the NanoBench per-trajectory PWM and
thrust channels (P3.8 dataset). The residual plot therefore cannot yet be
produced.

## 3. Aerodynamic refit (P3.5; F10)

The polynomial structure is the NanoBench fit inherited from the archived
simulator, with the two non-dissipative coefficients sign-corrected (BUGS R-2).
**The constrained refit and its train/test split are blocked** on the same
missing NanoBench trajectories. The decision rule is fixed in advance:

1. fit the polynomial with sign constraints that force dissipation,
2. fit a linear rotor-drag model on the same split,
3. keep the polynomial only if its held-out normalised RMS error is lower.

Until the data exists, no held-out number is claimed. `scripts/make_sim_figures.py`
regenerates the ESC curve; the aero residual figure is part of the same blocked
item.

## 4. Cross-checks and validation (P3.7-P3.11)

The tables below are regenerated by `notebooks/01_sim_validation.ipynb`.
Rows marked **pending** require an external dataset or a Colab runtime and are
reported explicitly rather than silently skipped.

| Check | Status | Result |
|---|---|---|
| Analytic free fall, hover, torque-free, constant torque, single-axis thrust | implemented | `tests/unit/test_sim_analytic.py` |
| Timestep convergence 100/250/500/1000/2000 Hz | implemented | fourth-order slope measured in `tests/unit/test_sim_timestep.py`; `docs/figures/convergence.png` |
| Dryden spectra/variance over 10 min | implemented | variance within 5% and spectrum shape in `tests/unit/test_dryden_wind.py`; `docs/figures/dryden_psd.png` |
| PyBullet cf2x three-level cross-check | pending | requires `pybullet`/`gym-pybullet-drones`; tests are marked and skipped |
| NanoBench one-step / rollout error | pending | external NanoBench trajectory files are not in the repository |
| Wind response plots | partly implemented | open-loop drift in `docs/figures/wind_drift.png`; closed-loop hold needs the P4.4 PID stack |
| Environment throughput | implemented | `scripts/benchmark_sim.py`; table below |

**Model limitation recorded for P3.6.** Turbulence is generated in world axes
(north, west, up) rather than in the aircraft wind axes. A quadrotor's yaw is
arbitrary and the guidance task does not define a flight-path heading, so the
wind-axis rotation is not applied. The three intensities and length scales are
still the MIL-HDBK-1797 ones, and the spectra are verified per axis.

## 5. Speed table (P3.10)

Measured on the development laptop, single process, Python 3.14 / numpy 2.5
(`python scripts/benchmark_sim.py --seconds 2`). Each RK4 step evaluates the
full motor, aero, ground-effect and rigid-body model in pure Python, so the
throughput is dominated by interpreter overhead.

| Configuration | Control steps/s | Physics steps/s |
|---|---|---|
| Legacy 100 Hz control | 1083 | 1083 |
| S8 500 Hz control / 1000 Hz physics | 549 | 1098 |
| 1000 Hz control / 1000 Hz physics | 1085 | 1085 |
| gym-pybullet-drones | pending | Colab/installed-runtime row |

The guidance environment is benchmarked separately in P5.10 (target
≥ 1000 environment steps/s, which the kinematic model meets by a wide margin).
