# AIM5 values: comparison with the CADAC++ AIM5 files

Every number in `configs/airframes/aim5.yaml` that is attributed to CADAC was compared with the
text decks of the public CADAC++ AIM5 example by P. H. Zipfel (readme.asc), repository
[SwaggyTyrion/CADAC](https://github.com/SwaggyTyrion/CADAC/tree/f5405b82fedfbc9fc9bdacfb0dfd83eb0a46fa41/example/AIM5)
(MIT), commit `f5405b8` (2019-05-25), files read 2026-11-15. The decks were parsed as text.

**Result: 262 of 262 numbers match exactly.**
The 192 aerodynamic table entries, 16 grid values and 44 propulsion table entries are summarised by
file line below; scalar parameters are listed one by one. Not in CADAC (own derivations, see
`docs/mathematics.md` section 4): `flight_condition_q12` and `derived_limits`.

| yaml key | ours | CADAC | file and line | verdict |
|---|---|---|---|---|
| `physical.reference_area` | 0.01767 | 0.01767 | aim5_aero_deck.asc L3 | match |
| `physical.reference_diameter` | 0.15 | 0.15 | aim5_aero_deck.asc L4 | match |
| `physical.max_angle_of_attack` | 35.0 | 35 | input.asc L35 | match |
| `physical.nozzle_exit_area` | 0.00948 | 0.00948 | input.asc L40 | match |
| `physical.specific_impulse` | 267.0 | 267 | aim5_prop_deck.asc L2 | match |
| `physical.rate_loop_time_constant` | 0.1 | 0.1 | input.asc L47 | match |
| `physical.prop_integral_ratio` | 2.0 | 2 | input.asc L48 | match |
| `physical.accel_loop_gain` | 40.0 | 40 | input.asc L49 | match |
| `physical.mass_initial` | 63.8 | 63.8 | aim5_prop_deck.asc L3 | match |
| `physical.mass_burnout` | 38.6 | 38.6 | aim5_prop_deck.asc mass table, last row | match |
| `propulsion.mass_vs_time` (22 numbers) | table | table | aim5_prop_deck.asc L8 to L18 | match |
| `propulsion.sea_level_thrust_vs_time` (22 numbers) | table | table | aim5_prop_deck.asc L22 to L32 | match |
| `aerodynamics.alpha_grid_deg` (8 numbers) | table | table | aim5_aero_deck.asc L8 to L15 | match |
| `aerodynamics.mach_grid` (8 numbers) | table | table | aim5_aero_deck.asc L8 to L15 | match |
| `aerodynamics.cl_vs_alpha_mach` (64 numbers) | table | table | aim5_aero_deck.asc L8 to L15 | match |
| `aerodynamics.cd_powered_vs_alpha_mach` (64 numbers) | table | table | aim5_aero_deck.asc L19 to L26 | match |
| `aerodynamics.cd_unpowered_vs_alpha_mach` (64 numbers) | table | table | aim5_aero_deck.asc L30 to L37 | match |
