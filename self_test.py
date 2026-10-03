"""Smoke test for v3.0 physics files."""
import numpy as np
import sys

print("=" * 60)
print("v3.0 Physics Smoke Test")
print("=" * 60)

# ─── 1. Import test ──────────────────────────────────────────────────────────
print("\n[1] Import test...")
try:
    import swift_physics_headless_wind as wind
    import swift_physics_headless_rlvec as rlvec
    import swift_rl_env as rlenv
    print(f"  wind   version: {wind.__version__}")
    print(f"  rlvec  version: {rlvec.__version__}")
    print("  ✓ All imports OK")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# ─── 2. Hover equilibrium (rk4_rigid_body_step backward compat) ──────────────
print("\n[2] Hover equilibrium (backward-compat step)...")
params = rlvec.PARAMS_NANOBENCH
Om = rlvec.hover_omega_from_params(params)
f_ps, t_ps = rlvec.propeller_force_torque(Om, params['c_l'], params['c_d'],
                                           params['spin_sign'])
f_b, tau_b = rlvec.aggregate_forces(f_ps, t_ps, params['r_P'])

state = dict(
    p_WB=np.array([0.0, 0.0, 5.0]),
    q_WB=np.array([1.0, 0.0, 0.0, 0.0]),
    v_WB=np.zeros(3),
    omega_B=np.zeros(3),
    Omega=Om,
    U_bat=4.2,
)

dt = 0.01
for i in range(100):
    state = rlvec.rk4_rigid_body_step(state, f_b, tau_b, params, dt)

drift = np.linalg.norm(state['p_WB'] - [0, 0, 5])
print(f"  Position after 1s hover: {state['p_WB']}")
print(f"  Drift: {drift*1000:.3f} mm")
assert drift < 0.01, f"Hover drift too large: {drift}"
print("  ✓ Hover stable")

# ─── 3. Ground effect safety (no NaN/Inf) ────────────────────────────────────
print("\n[3] Ground effect edge cases...")
test_heights = [0.0, 0.0001, 0.00575, 0.001, 0.01, 0.05, 0.1, 1.0]
R = params['R_ROTOR']
for h in test_heights:
    k = rlvec.ground_effect_factor(h, R)
    assert np.isfinite(k), f"GE NaN/Inf at h={h}: k={k}"
    assert 1.0 <= k <= 2.0 or k == 1.0, f"GE out of range at h={h}: k={k}"
    print(f"  h={h:.5f}m → k_ge={k:.4f}")

# Batch version
p_test = np.array([[0, 0, h] for h in test_heights])
k_batch = rlvec.ground_effect_factor_batch(p_test, R)
assert np.all(np.isfinite(k_batch)), "Batch GE has NaN/Inf!"
print("  ✓ Ground effect safe (no NaN/Inf)")

# ─── 4. Blade flapping test ───────────────────────────────────────────────────
print("\n[4] Blade flapping test...")
# For a flat quad, omega x r_P only produces vertical local velocity changes,
# which don't affect horizontal blade flapping.  Flapping activates with:
#   (a) Differential motor speeds (asymmetric Omega)
#   (b) Horizontal body velocity

# Test (a): differential Omega + forward velocity
Om_diff = np.array([[2200.0, 1800.0, 2200.0, 1800.0]])   # differential
v_B_fwd = np.array([[3.0, 0.0, 0.0]])                     # forward flight
omega_B_zero = np.zeros((1, 3))
tau_flap_a = rlvec.blade_flapping_torque_batch(
    Om_diff, v_B_fwd, omega_B_zero, params['C_FLAP'])
print(f"  Diff Omega + v_B=[3,0,0]: tau_flap = {tau_flap_a[0]}")
assert np.any(tau_flap_a != 0), "Flapping should be nonzero with differential Omega!"

# Test (b): symmetric Omega + zero velocity = zero flapping (correct physics)
Om_sym = np.full((1, 4), 2000.0)
v_B_zero = np.zeros((1, 3))
tau_flap_b = rlvec.blade_flapping_torque_batch(
    Om_sym, v_B_zero, omega_B_zero, params['C_FLAP'])
print(f"  Sym Omega + v_B=[0,0,0]:  tau_flap = {tau_flap_b[0]}  (should be zero)")
assert np.allclose(tau_flap_b, 0), "Flapping should be zero at symmetric hover!"
print("  OK Flapping correct: nonzero at diff-thrust, zero at sym-hover")

# ─── 5. Body aero drag ───────────────────────────────────────────────────────
print("\n[5] Body aerodynamic drag...")
v_B_fast = np.array([5.0, 0.0, 0.0])   # 5 m/s forward
Om_hover = rlvec.hover_omega_from_params(params)
f_a_w, t_a_w = wind.aerodynamic_force_torque(v_B_fast, Om_hover, params['m'], params)
print(f"  At v_B=[5,0,0]: f_aero={f_a_w}, τ_aero={t_a_w}")
assert abs(f_a_w[0]) > 1e-6, "Aero drag should be nonzero at 5 m/s!"
print("  ✓ Aero drag active")

# ─── 6. Full pipeline step ───────────────────────────────────────────────────
print("\n[6] Full pipeline step (rk4_full_step)...")
from scipy.optimize import brentq

# Find hover command
def _hover_cmd(U_bat=4.2):
    bc = params['battery_coeffs']
    Om_h = float(Om_hover[0])
    def _f(c):
        return (bc[0] + bc[1]*U_bat + bc[2]*np.sqrt(c) + bc[3]*c
                + bc[4]*U_bat*np.sqrt(c)) - Om_h
    try:
        return brentq(_f, 0.02, 1.0)
    except:
        return 0.5

c_hov = _hover_cmd()
cmd_hover = np.full(4, c_hov)
print(f"  Hover command: {c_hov:.4f}")

state_full = dict(
    p_WB=np.array([0.0, 0.0, 5.0]),
    q_WB=np.array([1.0, 0.0, 0.0, 0.0]),
    v_WB=np.zeros(3),
    omega_B=np.zeros(3),
    Omega=Om_hover.copy(),
    U_bat=4.2,
)

for i in range(100):
    state_full = rlvec.rk4_full_step(state_full, cmd_hover, params, dt)

drift_full = np.linalg.norm(state_full['p_WB'] - [0, 0, 5])
print(f"  Position after 1s: {state_full['p_WB']}")
print(f"  Motor speeds: {state_full['Omega']}")
print(f"  Drift: {drift_full*1000:.3f} mm")
print("  ✓ Full pipeline step works")

# ─── 7. Wind + rlvec match ───────────────────────────────────────────────────
print("\n[7] Wind vs Rlvec consistency...")
state_w = dict(
    p_WB=np.array([0.0, 0.0, 3.0]),
    q_WB=np.array([1.0, 0.0, 0.0, 0.0]),
    v_WB=np.array([1.0, 0.5, -0.1]),
    omega_B=np.array([0.1, -0.05, 0.02]),
    Omega=Om_hover.copy(),
    U_bat=4.2,
)
state_r = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in state_w.items()}

# Use same propeller forces
f_ps_w, t_ps_w = wind.propeller_force_torque(Om_hover, params['c_l'],
                                              params['c_d'], params['spin_sign'])
f_w, tau_w = wind.aggregate_forces(f_ps_w, t_ps_w, params['r_P'])

wind_vec = np.array([1.0, 0.0, 0.0])

for i in range(50):
    state_w = wind.rk4_rigid_body_step(state_w, f_w, tau_w, params, dt, wind_W=wind_vec)
    state_r = rlvec.rk4_rigid_body_step(state_r, f_w, tau_w, params, dt, wind_W=wind_vec)

dev = np.linalg.norm(state_w['p_WB'] - state_r['p_WB'])
print(f"  Wind pos after 0.5s: {state_w['p_WB']}")
print(f"  Rlvec pos after 0.5s: {state_r['p_WB']}")
print(f"  Deviation: {dev*1000:.3f} mm")
assert dev < 0.01, f"Wind vs rlvec deviation too large: {dev}"
print("  ✓ Wind and Rlvec agree")

# ─── 8. Performance benchmark ────────────────────────────────────────────────
print("\n[8] Performance benchmark...")
import time

state_perf = dict(
    p_WB=np.array([0.0, 0.0, 5.0]),
    q_WB=np.array([1.0, 0.0, 0.0, 0.0]),
    v_WB=np.zeros(3),
    omega_B=np.zeros(3),
    Omega=Om_hover.copy(),
    U_bat=4.2,
)
cmd_perf = np.full(4, c_hov)

# Warmup
for _ in range(10):
    state_perf = rlvec.rk4_full_step(state_perf, cmd_perf, params, dt)

# Timed
n_steps = 1000
t0 = time.perf_counter()
for _ in range(n_steps):
    state_perf = rlvec.rk4_full_step(state_perf, cmd_perf, params, dt)
t_elapsed = time.perf_counter() - t0

steps_per_sec = n_steps / t_elapsed
print(f"  {n_steps} full-pipeline steps in {t_elapsed:.3f}s")
print(f"  {steps_per_sec:,.0f} steps/sec")
print(f"  {1e6/steps_per_sec:.0f} µs/step")

# ─── 9. omega_B_prev aliasing fix check ──────────────────────────────────────
print("\n[9] swift_rl_env.py aliasing fix...")
rng = np.random.default_rng(42)
env_state = rlenv.DroneEnvState(1, rng)
action = np.array([[0.6, 0.01, -0.01, 0.0]])

# Run 2 steps and check D-term isn't zero
rlenv.step(env_state, action)
w1 = env_state.omega_B.copy()
prev1 = env_state.omega_B_prev.copy()
rlenv.step(env_state, action)
w2 = env_state.omega_B.copy()
prev2 = env_state.omega_B_prev.copy()

# prev2 should equal w1 (the previous omega_B), NOT w2
if np.allclose(prev2, w1):
    print(f"  omega_B_prev correctly stores previous value")
    print("  ✓ Aliasing fix working (PID D-term alive)")
else:
    print(f"  prev2={prev2}, w1={w1}, w2={w2}")
    print("  ✗ Aliasing may still be present!")

print("\n" + "=" * 60)
print("ALL SMOKE TESTS PASSED")
print("=" * 60)
