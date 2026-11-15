# Aerosonde values: comparison with mavsim_public and the book draft

Item 3 of the 2026-11-15 sourcing pass. Every value in `configs/airframes/aerosonde.yaml` that is
attributed to Beard & McLain was compared with

* **code**: `mavsim_python/parameters/aerosonde_parameters.py` in
  [byu-magicc/mavsim_public](https://github.com/byu-magicc/mavsim_public/blob/10363e0eaeb16a61a7463505f8e122a9eb7e2b4e/mavsim_python/parameters/aerosonde_parameters.py),
  commit `10363e0` (2026-06-13), read 2026-11-15. The file was parsed with `ast` (not executed);
  `L<n>` is the line number in that file.
* **book**: the authors' public draft of the second edition, `uavbook.pdf` dated October 2, 2025
  (linked from the mavsim_public README), Appendix E, Tables E.1 and E.2, p. 358. The 2012 print
  edition was not available; its page numbers may differ.

Comparison script: parse both, compare to a relative tolerance of 1e-9 (`match`) or 5e-3 (`rounded`).

**Result.** 52 of the 53 numbers equal the code exactly; the 53rd, the aspect ratio, is the code's own
formula (L43) rounded to 4 significant figures. The book draft agrees with the code everywhere except two
lateral derivatives (bold): `C_Y_beta` (book -0.83, code -0.98) and `C_n_p` (book -0.069, code +0.069).
We keep the code values: the mavsim README (L38) tells users to use the code's Aerosonde coefficients,
"slightly modified from the book". Neither derivative is read by the simulator (the fixed-wing model is
a point mass, EQ-FW-1), so no result depends on the choice. `epsilon` and `S_prop` exist only in the code.

| # | yaml key | ours | code name | code line | code value | book draft | verdict |
|---|---|---|---|---|---|---|---|
| 1 | `physical.mass` | 11 | `mass` | L32 | 11 | 11.0 | match |
| 2 | `physical.inertia[0]` | 0.8244 | `Jx` | L33 | 0.8244 | 0.824 | match |
| 3 | `physical.inertia[1]` | 1.135 | `Jy` | L34 | 1.135 | 1.135 | match |
| 4 | `physical.inertia[2]` | 1.759 | `Jz` | L35 | 1.759 | 1.759 | match |
| 5 | `physical.inertia_xz` | 0.1204 | `Jxz` | L36 | 0.1204 | 0.120 | match |
| 6 | `physical.wing_area` | 0.55 | `S_wing` | L37 | 0.55 | 0.55 | match |
| 7 | `physical.wing_span` | 2.8956 | `b` | L38 | 2.8956 | 2.9 | match |
| 8 | `physical.chord` | 0.18994 | `c` | L39 | 0.18994 | 0.19 | match |
| 9 | `physical.aspect_ratio` | 15.24 | `AR` | L43 | 15.2445 | (not listed) | rounded; derived b^2/S (mavsim L43 uses the same formula); 15.24 is 15.2445 rounded |
| 10 | `physical.oswald_efficiency` | 0.9 | `e` | L42 | 0.9 | 0.9 | match |
| 11 | `physical.air_density_sea_level` | 1.2682 | `rho` | L41 | 1.2682 | 1.268 | match; book density (3.5 % above ISA sea level 1.225); no altitude stated |
| 12 | `physical.propeller_area` | 0.2027 | `S_prop` | L40 | 0.2027 | (not listed) | match |
| 13 | `physical.propeller_diameter` | 0.508 | `D_prop` | L93 | 0.508 | 0.508 | match |
| 14 | `physical.motor_kv` | 145 | `KV_rpm_per_volt` | L96 | 145 | K_V 0.0659 V-s/rad | match; 145 rpm/V = 0.0659 V-s/rad (mavsim L97) |
| 15 | `physical.motor_resistance` | 0.042 | `R_motor` | L99 | 0.042 | 0.042 | match |
| 16 | `physical.motor_no_load_current` | 1.5 | `i0` | L100 | 1.5 | 1.5 | match |
| 17 | `physical.battery_cells` | 12 | `ncells` | L104 | 12 | (not listed) | match |
| 18 | `physical.battery_voltage_max` | 44.4 | `V_max` | L105 | 44.4 | 44.4 | match |
| 19 | `physical.gravity` | 9.81 | `gravity` | L44 | 9.81 | (not listed) | match |
| 20 | `aerodynamics.alpha_stall` | 0.47 | `alpha0` | L62 | 0.47 | 0.47 | match |
| 21 | `aerodynamics.CL_0` | 0.23 | `C_L_0` | L49 | 0.23 | 0.23 | match |
| 22 | `aerodynamics.CD_0` | 0.0424 | `C_D_0` | L50 | 0.0424 | 0.0424 | match |
| 23 | `aerodynamics.Cm_0` | 0.0135 | `C_m_0` | L51 | 0.0135 | 0.0135 | match |
| 24 | `aerodynamics.CL_alpha` | 5.61 | `C_L_alpha` | L52 | 5.61 | 5.61 | match |
| 25 | `aerodynamics.CD_alpha` | 0.132 | `C_D_alpha` | L53 | 0.132 | 0.132 | match |
| 26 | `aerodynamics.Cm_alpha` | -2.74 | `C_m_alpha` | L54 | -2.74 | -2.74 | match |
| 27 | `aerodynamics.CL_q` | 7.95 | `C_L_q` | L55 | 7.95 | 7.95 | match |
| 28 | `aerodynamics.Cm_q` | -38.21 | `C_m_q` | L57 | -38.21 | -38.21 | match |
| 29 | `aerodynamics.CL_delta_e` | 0.13 | `C_L_delta_e` | L58 | 0.13 | 0.13 | match |
| 30 | `aerodynamics.CD_delta_e` | 0.0135 | `C_D_delta_e` | L59 | 0.0135 | 0.0135 | match |
| 31 | `aerodynamics.Cm_delta_e` | -0.99 | `C_m_delta_e` | L60 | -0.99 | -0.99 | match |
| 32 | `aerodynamics.CD_p` | 0.043 | `C_D_p` | L64 | 0.043 | 0.043 | match |
| 33 | `aerodynamics.M` | 50 | `M` | L61 | 50 | 50 | match |
| 34 | `aerodynamics.epsilon` | 0.16 | `epsilon` | L63 | 0.16 | (not listed) | match |
| 35 | `aerodynamics.CY_beta` | -0.98 | `C_Y_beta` | L73 | -0.98 | **-0.83** | match; book differs from code; code value used (mavsim README L38 recommends the code coefficients); unused by the point-mass model |
| 36 | `aerodynamics.Cell_beta` | -0.13 | `C_ell_beta` | L74 | -0.13 | -0.13 | match |
| 37 | `aerodynamics.Cn_beta` | 0.073 | `C_n_beta` | L75 | 0.073 | 0.073 | match |
| 38 | `aerodynamics.Cell_p` | -0.51 | `C_ell_p` | L77 | -0.51 | -0.51 | match |
| 39 | `aerodynamics.Cn_p` | 0.069 | `C_n_p` | L78 | 0.069 | **-0.069** | match; book sign differs from code; code value used (README L38); unused by the point-mass model |
| 40 | `aerodynamics.Cell_r` | 0.25 | `C_ell_r` | L80 | 0.25 | 0.25 | match |
| 41 | `aerodynamics.Cn_r` | -0.095 | `C_n_r` | L81 | -0.095 | -0.095 | match |
| 42 | `aerodynamics.CY_delta_a` | 0.075 | `C_Y_delta_a` | L82 | 0.075 | 0.075 | match |
| 43 | `aerodynamics.Cell_delta_a` | 0.17 | `C_ell_delta_a` | L83 | 0.17 | 0.17 | match |
| 44 | `aerodynamics.Cn_delta_a` | -0.011 | `C_n_delta_a` | L84 | -0.011 | -0.011 | match |
| 45 | `aerodynamics.CY_delta_r` | 0.19 | `C_Y_delta_r` | L85 | 0.19 | 0.19 | match |
| 46 | `aerodynamics.Cell_delta_r` | 0.0024 | `C_ell_delta_r` | L86 | 0.0024 | 0.0024 | match |
| 47 | `aerodynamics.Cn_delta_r` | -0.069 | `C_n_delta_r` | L87 | -0.069 | -0.069 | match |
| 48 | `propeller_fit.CT_coefficients[0]` | 0.09357 | `C_T0` | L113 | 0.09357 | 0.09357 | match |
| 49 | `propeller_fit.CT_coefficients[1]` | -0.06044 | `C_T1` | L112 | -0.06044 | -0.06044 | match |
| 50 | `propeller_fit.CT_coefficients[2]` | -0.1079 | `C_T2` | L111 | -0.1079 | -0.1079 | match |
| 51 | `propeller_fit.CQ_coefficients[0]` | 0.00523 | `C_Q0` | L110 | 0.00523 | 0.005230 | match |
| 52 | `propeller_fit.CQ_coefficients[1]` | 0.00497 | `C_Q1` | L109 | 0.00497 | 0.004970 | match |
| 53 | `propeller_fit.CQ_coefficients[2]` | -0.01664 | `C_Q2` | L108 | -0.01664 | -0.01664 | match |

## Rows that cite Beard & McLain but are not values from it

**Update 2026-11-15 (owner decision, BUGS #32):** `v_max`, `a_t_max` and `a_t_min` were changed to the
model values in the "check" column (32.4 m/s, +2.47 m/s^2, -3.02 m/s^2), now computed by
`interceptor/guidance/airframes.py`. The table below records the state before that change.

The audit counted 55 rows referencing Beard & McLain: 47 in `aerosonde.yaml` (the 46 data rows in the
table above, which also lists the derived aspect ratio, plus `derived_limits.tau_accel`) and 8 in `configs/platforms.yaml`. Six of
the eight were labelled CITED only because the audit regex matched the word "aerosonde" in a file path;
the regex now matches only `Beard|McLain|mavsim`. Each was checked against the book model:

| row | value | what the source text claimed | check (2026-11-15) | label now |
|---|---|---|---|---|
| `derived_limits.tau_accel`, `platforms.fixed_wing.tau_accel` | 0.12 s | "roll/pitch first-order fit, Beard & McLain control chapter" | no 0.12 s roll/pitch constant in the book draft | PROJECT_CHOICE (v1 notebook) |
| `platforms.fixed_wing.v_min` | 15 m/s | EQ-FW-2 | `sqrt(2mg/(rho S CL_max))` with CL_max = 1.45 gives 14.6 m/s; mavsim has no CL_max and the book lift curve eqs. 4.9-4.10 peaks at 2.42 (23.6 deg; 11.3 m/s) | DERIVED, CL_max PROJECT_CHOICE |
| `platforms.fixed_wing.v_max` | 40 m/s | "thrust = drag root" (EQ-FW-3) | book eqs. 4.17-4.22 + mavsim parameters, full throttle, `C_D = C_Dp + C_L^2/(pi e AR)`: **32.4 m/s** | PROJECT_CHOICE (v1 notebook), **BUGS #32 open** |
| `platforms.fixed_wing.a_t_max` | 41.2 m/s^2 (4.2 g) | "excess thrust" (EQ-FW-4) | (T - D)/m = **2.5 m/s^2** at 25 m/s; static thrust 84.6 N gives at most 7.7 m/s^2 | PROJECT_CHOICE, **BUGS #32 open** |
| `platforms.fixed_wing.a_t_min` | -19.6 m/s^2 (-2 g) | "idle thrust/drag" (EQ-FW-4) | (T_idle - D)/m = **-3.0 m/s^2** at 25 m/s | PROJECT_CHOICE, **BUGS #32 open** |
| `platforms.fixed_wing.max_jerk` | 180 m/s^3 | EQ-FW-7 | below a_n,max/tau_a = 311 m/s^3 | PROJECT_CHOICE |
| `platforms.fixed_wing.kill_radius` | 3.0 m | half-spans | Aerosonde span 2.8956 m = mavsim L38 `b`; Orlan-10 span 3.1 m is trade press (item 4) | unchanged |
| `platforms.fixed_wing.v_target_nominal` | 30.6 m/s | Orlan-10 cruise | not a Beard value (the text mentions the Aerosonde interceptor); trade-press source (item 4) | see item 4 |

The v_max / a_t_max / a_t_min values were traced to `archive/path_planning_updated_project/apn_rl_interceptor_path_planning.ipynb` (`v_max=40.0`); no derivation produced them. Whether to replace
them with the model values is a platform decision for the project owner (BUGS #32).

Reproduce: the calculation is `interceptor`-independent and uses only the book equations and the
parameters in the table above (static thrust from eq. 4.17 at V = 0; bisection on T(V) = D(V)).
