# Coefficient Fitting — Aerodynamic and ESC Models

## Overview

The simulation uses two sets of empirically-fitted coefficients:

1. **Aerodynamic drag coefficients** (Stage 7) — 6 polynomial models for body-frame forces and torques
2. **ESC battery polynomial** (Stage 2) — 5-coefficient model mapping motor commands to steady-state speeds

Both are fitted by least-squares regression against real PX4 flight-log data.

## Data Source

- **15 PX4 `.ulg` flight logs** from a Crazyflie 2.0 / PX4-based quadrotor
- Extracted using `pyulog` — fields: `vehicle_attitude`, `vehicle_local_position`, `actuator_outputs`, `battery_status`
- PID gains (`MC_ROLLRATE_P`, `MC_PITCHRATE_P`, etc.) read directly from the log parameter section

## ESC Model (Stage 2)

The ESC polynomial maps normalised motor command `cmd ∈ [0, 1]` and battery voltage `U_bat` to steady-state motor speed `Ω_ss`:

```
Ω_ss = c₀ + c₁·U_bat + c₂·√cmd + c₃·cmd + c₄·U_bat·√cmd
```

**Important:** The raw fitted polynomial uses an inverted command convention (cmd=0 → max speed). The simulation inverts with `cmd_esc = 1 - cmd` before evaluation.

Fitted coefficients (pooled across 4 ESC/motor channels):
```
c₀ =  4249.814
c₁ =  -194.617
c₂ = -3453.525
c₃ =    50.067
c₄ =   208.184
```

## Aerodynamic Model (Stage 7)

Six polynomial models for body-frame drag:

| Output | Regressor terms |
|--------|----------------|
| f_x | vx, vx·\|vx\|, Ω̄², vx·Ω̄² |
| f_y | vy, vy·\|vy\|, Ω̄², vy·Ω̄² |
| f_z | vz, vz·\|vz\|, v_xy², v_xy·Ω̄², vz·Ω̄², v_xy·vz·Ω̄² |
| τ_x | vy, vy·\|vy\|, Ω̄², vy·Ω̄², vy·\|vy\|·Ω̄² |
| τ_y | vx, vx·\|vx\|, Ω̄², vx·Ω̄², vx·\|vx\|·Ω̄² |
| τ_z | vx, vy |

### Unit correction

Raw fitted values are in **acceleration units** (the regression target was measured accel, not force). To convert to SI:
- Force coefficients: multiply by mass `m`
- Torque coefficients: multiply by the corresponding diagonal element of `J`

### Offset zeroing

The standalone Ω̄² terms (index 2 in fx/fy/τx/τy) encode propeller + gravity contributions already modelled by Stages 4–6. These are zeroed to avoid double-counting.

## Author

Siddh Mehta, 2025
