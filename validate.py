"""
validate.py  --  standalone version of phy_sim_val.ipynb

Runs the open-loop simulation against the NanoBench dataset and prints
error metrics. No Jupyter required. Run from the project directory:

    python validate.py  [path/to/nanobench_csv]

If no path is given it tries to find the CSV in the current directory.
"""

import sys, os, glob
import numpy as np

# ─── Load physics module ──────────────────────────────────────────────────────
PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)

import importlib
import swift_physics_headless
importlib.reload(swift_physics_headless)
phys = swift_physics_headless

try:
    import pandas as pd
except ImportError:
    sys.exit("ERROR: pandas not installed. Run:  pip install pandas")

# ─── Find CSV ─────────────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    csv_path = sys.argv[1]
else:
    candidates = glob.glob(os.path.join(PROJ, "*.csv"))
    if not candidates:
        sys.exit("ERROR: No CSV found. Pass path as argument: python validate.py data.csv")
    csv_path = candidates[0]
    print(f"Using CSV: {csv_path}")

df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} rows, columns: {df.columns.tolist()[:10]} ...")

# ─── Column mapping (same as notebook Cell 3) ────────────────────────────────
COLMAP = {
    "t":   "t",
    "px": "px", "py": "py", "pz": "pz",
    "vx": "vx", "vy": "vy", "vz": "vz",
    "u1": "motor_motor_m1", "u2": "motor_motor_m2",
    "u3": "motor_motor_m3", "u4": "motor_motor_m4",
    "vbat": "pwr_pm_vbat",
    "gx": "imu_gyro_x", "gy": "imu_gyro_y", "gz": "imu_gyro_z",
}

def col(name):
    c = COLMAP[name]
    if c not in df.columns:
        raise KeyError(f"Column '{c}' not found. Available: {df.columns.tolist()}")
    return df[c].to_numpy()

t = col("t")
dt = float(np.median(np.diff(t)))
print(f"dt = {dt:.5f} s  ({1/dt:.1f} Hz),  N = {len(t)} steps  ({len(t)*dt:.1f} s)")

p_gt = np.column_stack([col("px"), col("py"), col("pz")])

qx, qy, qz, qw = col("qx") if "qx" in COLMAP else (df["qx"].to_numpy(), df["qy"].to_numpy(),
                                                      df["qz"].to_numpy(), df["qw"].to_numpy())

# NanoBench: scalar-last (qx,qy,qz,qw) -> scalar-first (qw,qx,qy,qz)
q_gt = np.column_stack([df["qw"].to_numpy(), df["qx"].to_numpy(),
                         df["qy"].to_numpy(), df["qz"].to_numpy()])

v_gt = np.column_stack([col("vx"), col("vy"), col("vz")])
omega_gt = np.column_stack([col("gx"), col("gy"), col("gz")])

pwm_raw = np.column_stack([col("u1"), col("u2"), col("u3"), col("u4")])
if pwm_raw.max() > 2.0:
    cmd_sequence = pwm_raw / 65535.0
else:
    cmd_sequence = pwm_raw
vbat = col("vbat")

# ─── Parameters ───────────────────────────────────────────────────────────────
params = phys.PARAMS_NANOBENCH

# ─── Initial state (FIX: Omega from hover speed, not zero) ───────────────────
hover_omega = phys.hover_omega_from_params(params)
initial_state = {
    "p_WB":   p_gt[0].copy(),
    "q_WB":   q_gt[0].copy(),
    "v_WB":   v_gt[0].copy(),
    "omega_B": omega_gt[0].copy(),
    "Omega":  hover_omega,          # <-- hover speed, not np.zeros(4)
    "U_bat":  float(vbat[0]),
}
print(f"Initial Omega (hover): {hover_omega[0]:.1f} rad/s per motor")
print(f"Initial U_bat: {initial_state['U_bat']:.2f} V")

# ─── Sanity check: ESC at first commanded step ────────────────────────────────
first_nonzero = np.argmax(cmd_sequence.max(axis=1) > 0.02)
Omega_ss_check, _ = phys.esc_battery_model(
    cmd_sequence[first_nonzero], vbat[first_nonzero], hover_omega,
    params["eta"], params["battery_coeffs"], dt, params["c_d"])
print(f"Omega_ss at first active step (cmd={cmd_sequence[first_nonzero].mean():.3f}, "
      f"U={vbat[first_nonzero]:.2f}V): {Omega_ss_check}")

# ─── Run simulation ───────────────────────────────────────────────────────────
print("\nRunning open-loop simulation (RK4)...")
sim = phys.simulate_open_loop(
    cmd_sequence=cmd_sequence,
    dt=dt,
    initial_state=initial_state,
    params=params,
    U_bat_sequence=vbat,
    integrator="rk4",
)

p_pred = sim["p_WB"]
v_pred = sim["v_WB"]
q_pred = sim["q_WB"]

# ─── Error metrics ────────────────────────────────────────────────────────────
def quat_angle_error(q1, q2):
    def conj(q):
        return np.column_stack([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]])
    def qmul(a, b):
        w1, x1, y1, z1 = a[:,0], a[:,1], a[:,2], a[:,3]
        w2, x2, y2, z2 = b[:,0], b[:,1], b[:,2], b[:,3]
        return np.column_stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])
    rel = qmul(conj(q1), q2)
    qv_norm = np.linalg.norm(rel[:, 1:], axis=1)
    return 2 * np.arctan2(qv_norm, np.abs(rel[:, 0]))

pos_err = np.linalg.norm(p_pred - p_gt, axis=1)
vel_err = np.linalg.norm(v_pred - v_gt, axis=1)
att_err = quat_angle_error(q_gt, q_pred)

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Position error  -- mean: {pos_err.mean()*1000:.1f} mm, "
      f"median: {np.median(pos_err)*1000:.1f} mm, "
      f"final: {pos_err[-1]*1000:.1f} mm")
print(f"Velocity error  -- mean: {vel_err.mean():.3f} m/s, "
      f"final: {vel_err[-1]:.3f} m/s")
print(f"Attitude error  -- mean: {np.degrees(att_err).mean():.2f} deg, "
      f"final: {np.degrees(att_err[-1]):.2f} deg")

# Per-horizon snapshot (1s, 2s, 5s)
print()
print("Horizon snapshots:")
for t_hor in [0.5, 1.0, 2.0, 5.0]:
    idx = min(int(t_hor / dt) - 1, len(pos_err) - 1)
    if idx >= 0:
        print(f"  t={t_hor:.1f}s: pos={pos_err[idx]*1000:.1f}mm, "
              f"vel={vel_err[idx]:.3f}m/s, att={np.degrees(att_err[idx]):.2f}deg")

# ─── Optional: save a quick comparison plot ───────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    t_ax = np.arange(len(t)) * dt
    labels = ["x", "y", "z"]

    for i in range(3):
        axes[0].plot(t_ax, p_gt[:, i], "--", label=f"gt {labels[i]}")
        axes[0].plot(t_ax, p_pred[:, i], label=f"pred {labels[i]}")
    axes[0].set_ylabel("position (m)")
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].set_title("Position: predicted vs ground truth")

    for i in range(3):
        axes[1].plot(t_ax, v_gt[:, i], "--", label=f"gt v{labels[i]}")
        axes[1].plot(t_ax, v_pred[:, i], label=f"pred v{labels[i]}")
    axes[1].set_ylabel("velocity (m/s)")
    axes[1].legend(ncol=3, fontsize=8)

    axes[2].plot(t_ax, pos_err * 1000)
    axes[2].set_ylabel("position error (mm)")

    axes[3].plot(t_ax, np.degrees(att_err))
    axes[3].set_ylabel("attitude error (deg)")
    axes[3].set_xlabel("time (s)")

    plt.tight_layout()
    out_path = os.path.join(PROJ, "validation_plot.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to: {out_path}")
except Exception as e:
    print(f"\n(Plot skipped: {e})")
