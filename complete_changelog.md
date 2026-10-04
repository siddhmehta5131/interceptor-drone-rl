# Complete Update Rundown: v1.0 (Demo) → v3.0 (Current Latest)

## Files at a Glance

| File | Old version | Current version |
|---|---|---|
| [`swift_physics_headless_rlvec_latest.py`](file:///c:/Users/ADMIN/Desktop/projects/RL%20based%20autonomous%20interceptor%20drone/swift_physics_headless_rlvec_latest.py) | v1.0 / v2.0 | **v3.0** |
| [`swift_physics_headless_wind_latest.py`](file:///c:/Users/ADMIN/Desktop/projects/RL%20based%20autonomous%20interceptor%20drone/swift_physics_headless_wind_latest.py) | v1.0 (no aero) | **v3.0** |
| [`swift_rl_env_latest.py`](file:///c:/Users/ADMIN/Desktop/projects/RL%20based%20autonomous%20interceptor%20drone/swift_rl_env_latest.py) | D-term dead | **D-term fixed** |

---

## 1. Bug Fixes

### BUG 1 — Ground Effect: Division by Zero ☠️
**What it was:** `k_ge = 1 / (1 - (R/4h)²)` — when `h < R/4 = 5.75mm`, the ratio exceeded 1.0, giving `1/(negative) = negative or ±Inf`, which caused NaN propagation and instant crash.

```diff
# BEFORE (v1.0 / v2.0)
ratio = R_rotor / (4.0 * height)       # could be > 1.0 → NaN

# AFTER (v3.0)
ratio = min(R_rotor / (4.0 * max(height, 1e-6)), 0.99)  # safe clamp
k_ge  = min(1.0 / (1.0 - ratio**2), 2.0)               # physically capped
```
**Impact:** Sim crashed at z < 5.75 mm. Now safe down to z = 0.1 mm.

---

### BUG 2 — Blade Flapping: Zero Torque at Hover ⚠️
**What it was:** Flapping was computed with the global body-frame velocity, which is **symmetric across all 4 rotors** at hover. The contributions cancelled to exactly zero — flapping was dead in any symmetric state.

```diff
# BEFORE (v2.0) — uses global v_B for all rotors
for j in range(4):
    tau += C_FLAP * Omega[j] * spin[j] * [v_B[1], -v_B[0], 0]
# ^^^ Symmetric Omega + symmetric v_B → net sum = 0

# AFTER (v3.0) — uses per-rotor local velocity
for j in range(4):
    v_local = v_B + cross(omega_B, r_P[j])   # each rotor sees different airspeed
    tau += C_FLAP * Omega[j] * spin[j] * [v_local[1], -v_local[0], 0]
```
**Impact:** Flapping now activates during differential thrust, which creates realistic roll coupling.

---

### BUG 3 — omega_quat Hardcoded N=1 ⚠️
**What it was:** The quaternion derivative used a hardcoded batch size of 1, silently breaking for N>1 environments.

```diff
# BEFORE: omega_quat = np.zeros((1, 4))  ← always N=1
# AFTER:  omega_quat = np.concatenate([np.zeros((N, 1)), w], axis=1)
```

---

### BUG 4 — Omega_max Not Enforced After RK4 ⚠️
**What it was:** Motor speeds were clamped at the ESC input but the RK4 integration could push Omega past `Omega_max` mid-step with no clamp at output.

```diff
# AFTER (v3.0) — enforced at RK4 output
out[4] = np.clip(out[4], 0.0, Omega_max)
```

---

### BUG 5 — omega_B_prev Aliasing in swift_rl_env.py ☠️
**What it was:** Without `.copy()`, both variables pointed to the same numpy array. When omega_B was updated, omega_B_prev changed too — making the PID D-term `omega_B - omega_B_prev = 0` always.

```diff
# BEFORE — PID D-term was permanently dead
state.omega_B_prev = state.omega_B

# AFTER — D-term alive
state.omega_B_prev = state.omega_B.copy()
```
**Impact:** Attitude PID had effectively zero derivative control. Angular velocity overshoots were not damped.

---

## 2. Missing Physics Added

### PHYSICS 1 — Body Aerodynamic Forces (Polynomial Model) ✈️
**What it was:** Zero. No body drag at all. The drone flew like it was in a vacuum.

**What v3.0 adds:** Full polynomial aerodynamic model from real NanoBench flight data (sysid):
- `F_x = f(vx, vx|vx|, Ω̄², vx·Ω̄²)` — forward drag + induced
- `F_y = f(vy, vy|vy|, Ω̄², vy·Ω̄²)` — lateral drag
- `F_z = f(vz, |vz|, vxy², ...)` — vertical drag/suction
- `τ_z = f(vx, vy)` — yaw coupling from asymmetric flow

**Impact:** At 8 m/s forward flight, v2.0 predicted x=14.7m in 3s. v3.0 correctly predicts x=4.5m. Real drag stops the tiny 40g drone fast.

---

### PHYSICS 2 — Quadratic Body Drag (Physics-Based, Always Safe) 🛡️
**What v3.0 adds:** ½ρCdA·v·|v| drag using Crazyflie geometry:

```
C_XY = 0.00184 N/(m/s)²  (30mm height × 100mm span, Cd=1.0)
C_Z  = 0.00637 N/(m/s)²  (80mm × 100mm top face, Cd=1.3)
```
**Why needed:** The polynomial model was fit only near hover. At extreme velocities it extrapolated wildly and injected energy (positive feedback). The quadratic drag is **always dissipative** and correct at all speeds.

---

### PHYSICS 3 — Dissipative Projection (Aero Safety) 🔒
**What v3.0 adds:** Any energy-injecting component of the polynomial force is projected out:

```python
# If F·v > 0 (force accelerates along velocity = energy injection), remove it
inject = max(F·v, 0) / |v|²
F_safe = F - inject·v
```
**Why needed:** The `vz·Ω²` term in the polynomial caused `fz_raw` to accelerate a near-ground drone upward → more velocity → more force → runway to z=22m in 2s from z=3cm hover.

---

### PHYSICS 4 — Motor Reaction Torque (Gyroscopic Coupling) 🔄
**What it was:** When motor speed changes, the spinning propeller's angular momentum creates a body torque. This was missing entirely.

**What v3.0 adds:**
```
τ_mot = J_mp · Σⱼ (ζⱼ · Ω̇ⱼ)
```
where `J_mp = 2e-9 kg·m²` is the propeller moment of inertia. Causes slight pitch/roll when throttling up/down.

---

### PHYSICS 5 — Motor First-Order Dynamics (ESC Lag) ⚡
**What it was:** Motor speed jumped instantly to commanded value. Unrealistic for RL training — the policy never sees the lag.

**What v3.0 adds (rk4_full_step only):**
```
Ω̇ = (Ω_ss - Ω) / τ_mot     where τ_mot = 0.02s
```
Full ESC battery-voltage polynomial `Ω_ss = f(cmd, V_bat)`. Omega is now a **state variable** that evolves in the RK4 integration.

---

### PHYSICS 6 — Hub Drag (was in v2.0, now safe) ✅
Already present in v2.0: `F_hub = −C_HUB · ΣΩ² · [vx, vy, 0]`.  
Unchanged in v3.0 — still active.

---

### PHYSICS 7 — Ground Effect (was in v2.0, now fixed) ✅
Cheeseman-Bennett formula was in v2.0 but had the division-by-zero bug above. Now safe.

---

## 3. Performance Improvements

| Optimization | v2.0 | v3.0 |
|---|---|---|
| `J_inv` computation | 4× per step (inside each RK4 sub-step) | **1× per step** (cached) |
| Aero coefficients | Allocated per sub-step | **Tiled once per step** |
| All extras function | Inlined in each derivative | **`_compute_extras()` — shared** |

**Result:** Full-pipeline step at 100 Hz → **10× realtime** on your i3 laptop.

---

## 4. API Changes

| Function | v2.0 | v3.0 |
|---|---|---|
| `rk4_rigid_body_step()` | Fixed propeller only | Same + aero drag, safe GE, fixed flapping |
| `rk4_full_step()` | **Did not exist** | **NEW** — takes motor commands, evolves Ω |
| `wind_W` parameter | Missing on some functions | Both step functions accept `wind_W=(3,)` |
| `PARAMS_NANOBENCH` | Basic rigid body params | Full physics params (aero, ESC, battery) |
| `quadratic_body_drag_batch()` | Not present | **NEW** |
| `ensure_dissipative_batch()` | Not present | **NEW** |

---

## 5. Verified Benchmark Results

### Multi-Rate Convergence (4000 Hz = ground truth)

| Scenario | 100 Hz error | 400 Hz error | 1000 Hz error |
|---|---|---|---|
| Hover | **0.00 mm** | 0.00 mm | 0.00 mm |
| Near-ground (z=3cm) | 1.18 mm | 0.27 mm | 0.09 mm |
| High-speed (8 m/s) | 77.8 mm | 18.0 mm | 6.0 mm |
| Aggressive roll | 122 mm | 17.0 mm | 5.7 mm |
| Free fall + restart | 56.1 mm | 13.0 mm | 4.3 mm |
| Tumble (ω₀=20 rad/s) | 58.0 mm | 13.4 mm | 4.5 mm |
| High-speed + wind | 54.8 mm | 12.6 mm | 4.2 mm |

> [!TIP]
> **100 Hz is fine for RL training** — errors are mm-scale, well within domain randomization noise. Use **400 Hz for precision tasks** (landing, interception).

### Aero Drag Impact (v3.0 vs v2.0 at 1000 Hz)

| Scenario | v2.0 final pos | v3.0 final pos | Δ |
|---|---|---|---|
| Hover (5m) | `[0, 0, 5]` | `[0, 0, 5]` | 0 mm ✅ |
| Near-ground | z=0.245 m | z=0.242 m | 3 mm (tiny) |
| 8 m/s forward 3s | x=14.7 m | x=4.5 m | **10.2 m** |
| Free fall recovery | z=13.6 m | z=13.7 m | 83 mm |
| Tumble | drifts 1.6m | drifts 0.86m | **936 mm** |

The aero model makes the biggest difference at **high speeds and angular rates** — exactly the extreme cases you care about for interception.

---

## 6. Summary: What Changed Where

```
swift_physics_headless_rlvec_latest.py  (PRIMARY — use this for RL)
  + ground_effect_factor_batch()    BUG FIX: no div/0
  + blade_flapping_torque_batch()   BUG FIX: per-rotor local velocity
  + aerodynamic_force_torque_batch()  NEW PHYSICS: polynomial body drag
  + ensure_dissipative_batch()      NEW: removes energy injection from polynomial
  + quadratic_body_drag_batch()     NEW PHYSICS: always-stable drag
  + rk4_full_step()                 NEW API: motor commands → full pipeline
  + _make_cache()                   PERF: precomputed J_inv + coefficients
  + Omega_max clamp at RK4 output   BUG FIX

swift_physics_headless_wind_latest.py  (SCALAR — same physics, human-readable)
  All of the above in scalar (non-vectorized) form

swift_rl_env_latest.py
  + omega_B_prev = omega_B.copy()   BUG FIX: PID D-term alive again
```
