"""
validate_onestep.py  --  One-step prediction validation (fast vectorized version)

For every timestep i in the dataset:
  1. Take the REAL state at step i (position, velocity, attitude, angular rate)
  2. Apply the REAL motor command at step i
  3. Run ONE physics step (dt = 0.01s, RK4)
  4. Compare predicted state at i+1 vs real state at i+1

This is the correct way to validate a physics model. PID corrections,
wind, and real-world disturbances don't matter -- we reset to ground
truth every single step so errors never accumulate.
"""
import sys, os, glob
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
import importlib, swift_physics_headless
importlib.reload(swift_physics_headless)
phys = swift_physics_headless

# ─── Settings ────────────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    csv_files = [sys.argv[1]]
else:
    csv_files = sorted(glob.glob(r'nanobench-iros2026\datasets\dataset\*.csv'))
    csv_files = [f for f in csv_files if '_metadata' not in f]
    print(f"Running one-step validation on {len(csv_files)} trajectory files...")

params = phys.PARAMS_NANOBENCH
dt = 0.01

# ─── Helper: quaternion angle error (vectorized) ──────────────────────────────
def quat_angle_error_vec(q1, q2):
    """q1, q2: (N,4) scalar-first (w,x,y,z). Returns (N,) degrees."""
    # Relative rotation: q_rel = conj(q1) * q2
    w1, x1, y1, z1 = q1[:,0], q1[:,1], q1[:,2], q1[:,3]
    w2, x2, y2, z2 = q2[:,0], q2[:,1], q2[:,2], q2[:,3]
    rw =  w1*w2 + x1*x2 + y1*y2 + z1*z2
    rx = -x1*w2 + w1*x2 - z1*y2 + y1*z2  # note: conj flips x1,y1,z1 sign
    ry = -y1*w2 + z1*x2 + w1*y2 - x1*z2
    rz = -z1*w2 - y1*x2 + x1*y2 + w1*z2
    vec_norm = np.sqrt(rx**2 + ry**2 + rz**2)
    return np.degrees(2 * np.arctan2(vec_norm, np.abs(rw)))

# ─── One-step prediction for a single trajectory ─────────────────────────────
def one_step_errors(df):
    N = len(df)
    if N < 3:
        return None

    p   = df[['px','py','pz']].values
    q   = df[['qw','qx','qy','qz']].values   # scalar-first
    v   = df[['vx','vy','vz']].values
    om  = df[['imu_gyro_x','imu_gyro_y','imu_gyro_z']].values
    pwm = df[['motor_motor_m1','motor_motor_m2',
               'motor_motor_m3','motor_motor_m4']].values / 65535.0
    vbat = df['pwr_pm_vbat'].values
    pz   = df['pz'].values
    avg_cmd = pwm.mean(axis=1)

    # Mask: in-flight rows where we can also compare to the next row
    mask = (pz > 0.3) & (avg_cmd > 0.3)
    mask[-1] = False  # can't compare last row (no next row)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None

    # Batch predict: run each row's state through one RK4 step
    hover_omega = phys.hover_omega_from_params(params)

    p_pred_all  = np.zeros((len(idx), 3))
    v_pred_all  = np.zeros((len(idx), 3))
    q_pred_all  = np.zeros((len(idx), 4))

    for k, i in enumerate(idx):
        state = {
            "p_WB":    p[i].copy(),
            "q_WB":    q[i].copy(),
            "v_WB":    v[i].copy(),
            "omega_B": om[i].copy(),
            "Omega":   hover_omega.copy(),
            "U_bat":   float(vbat[i]),
        }
        # Single RK4 step using internal _state_derivative + rk4_step
        p_dot, q_dot, v_dot, om_dot, Omega_dot, _ = phys._state_derivative(
            state["p_WB"], state["q_WB"], state["v_WB"], state["omega_B"],
            state["Omega"], pwm[i], vbat[i], dt, params)

        p_pred_all[k]  = state["p_WB"]  + dt * p_dot
        v_pred_all[k]  = state["v_WB"]  + dt * v_dot
        q_raw          = state["q_WB"]  + dt * q_dot
        q_pred_all[k]  = q_raw / np.linalg.norm(q_raw)

    # Ground truth at i+1
    next_idx   = idx + 1
    p_next     = p[next_idx]
    v_next     = v[next_idx]
    q_next     = q[next_idx]

    pos_err = np.linalg.norm(p_pred_all - p_next, axis=1) * 1000  # mm
    vel_err = np.linalg.norm(v_pred_all - v_next, axis=1)          # m/s
    att_err = quat_angle_error_vec(q_next, q_pred_all)              # deg

    return pos_err, vel_err, att_err


# ─── Run across all files ─────────────────────────────────────────────────────
all_pos = []
all_vel = []
all_att = []
n_ok    = 0

for fpath in csv_files:
    try:
        df = pd.read_csv(fpath)
        required = ['px','py','pz','qw','qx','qy','qz',
                    'vx','vy','vz','imu_gyro_x','imu_gyro_y','imu_gyro_z',
                    'motor_motor_m1','motor_motor_m2','motor_motor_m3','motor_motor_m4',
                    'pwr_pm_vbat']
        if not all(c in df.columns for c in required):
            continue
        result = one_step_errors(df)
        if result is None:
            continue
        p_err, v_err, a_err = result
        all_pos.append(p_err)
        all_vel.append(v_err)
        all_att.append(a_err)
        n_ok += 1
        print(f"  {os.path.basename(fpath):50s}  "
              f"pos={np.median(p_err):.2f}mm  "
              f"vel={np.median(v_err):.4f}m/s  "
              f"att={np.median(a_err):.4f}deg  "
              f"({len(p_err)} steps)")
    except Exception as e:
        print(f"  SKIP {os.path.basename(fpath)}: {e}")

all_pos = np.concatenate(all_pos)
all_vel = np.concatenate(all_vel)
all_att = np.concatenate(all_att)

print()
print("=" * 68)
print(f"ONE-STEP PREDICTION RESULTS  ({n_ok} files, {len(all_pos):,} steps)")
print("=" * 68)
print(f"{'Metric':<22} {'Mean':>8} {'Median':>8} {'90th%':>8} {'99th%':>8} {'Max':>10}")
print("-" * 68)
print(f"{'Position (mm)':<22} "
      f"{all_pos.mean():>8.3f} "
      f"{np.median(all_pos):>8.3f} "
      f"{np.percentile(all_pos,90):>8.3f} "
      f"{np.percentile(all_pos,99):>8.3f} "
      f"{all_pos.max():>10.3f}")
print(f"{'Velocity (m/s)':<22} "
      f"{all_vel.mean():>8.5f} "
      f"{np.median(all_vel):>8.5f} "
      f"{np.percentile(all_vel,90):>8.5f} "
      f"{np.percentile(all_vel,99):>8.5f} "
      f"{all_vel.max():>10.5f}")
print(f"{'Attitude (deg)':<22} "
      f"{all_att.mean():>8.5f} "
      f"{np.median(all_att):>8.5f} "
      f"{np.percentile(all_att,90):>8.5f} "
      f"{np.percentile(all_att,99):>8.5f} "
      f"{all_att.max():>10.5f}")
print()
print("Good benchmarks for a Crazyflie physics model (from literature):")
print("  Position: < 5 mm median per step   (0.01s)")
print("  Velocity: < 0.05 m/s median per step")
print("  Attitude: < 0.5 deg median per step")

# ─── Plot ─────────────────────────────────────────────────────────────────────
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"One-Step Prediction Error  ({len(all_pos):,} steps, {n_ok} trajectories)")

    for ax, data, label, color, unit in [
        (axes[0], all_pos, "Position",  "steelblue",  "mm"),
        (axes[1], all_vel, "Velocity",  "seagreen",   "m/s"),
        (axes[2], all_att, "Attitude",  "darkorange",  "deg"),
    ]:
        # Clip to 99th percentile for readability
        clip = np.percentile(data, 99)
        ax.hist(data[data <= clip], bins=80, color=color, edgecolor='none', alpha=0.8)
        ax.axvline(np.median(data), color='red', linewidth=2,
                   label=f'median = {np.median(data):.4f} {unit}')
        ax.axvline(np.mean(data),   color='orange', linewidth=1.5, linestyle='--',
                   label=f'mean   = {np.mean(data):.4f} {unit}')
        ax.set_xlabel(f"{label} error ({unit})")
        ax.set_ylabel("Count")
        ax.set_title(f"{label} error per step (10ms)")
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("onestep_validation.png", dpi=150)
    print(f"\nPlot saved: onestep_validation.png")
except Exception as e:
    print(f"(Plot skipped: {e})")
