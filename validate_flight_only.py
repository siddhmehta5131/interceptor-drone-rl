"""
validate_flight_only.py -- runs validation only on the actual in-flight segment,
skipping ground-sit, takeoff ramp and landing phases where an open-loop
physics model has no hope (zero thrust = free-fall for 4 seconds).
"""
import pandas as pd, numpy as np, sys, os, warnings
sys.path.insert(0, '.')
import importlib, swift_physics_headless
importlib.reload(swift_physics_headless)
phys = swift_physics_headless

csv_path = sys.argv[1] if len(sys.argv) > 1 else \
    r"nanobench-iros2026\datasets\dataset\A1b_multisine_sysid_rep2.csv"

df = pd.read_csv(csv_path)
params = phys.PARAMS_NANOBENCH

pwm_raw = df[['motor_motor_m1','motor_motor_m2','motor_motor_m3','motor_motor_m4']].values
cmd_all = pwm_raw / 65535.0
vbat_all = df['pwr_pm_vbat'].values
pz_all = df['pz'].values
dt = 0.01

# ─── Find the active flight window ───────────────────────────────────────────
# Trim: skip rows where avg cmd < 0.3 (ground sit / very slow takeoff)
# and where pz < 0.3m (still near ground)
avg_cmd = cmd_all.mean(axis=1)
in_flight_mask = (avg_cmd > 0.3) & (pz_all > 0.3)

# Find contiguous flight segment (largest block)
changes = np.diff(in_flight_mask.astype(int))
starts = np.where(changes == 1)[0] + 1
ends = np.where(changes == -1)[0] + 1
if in_flight_mask[0]: starts = np.concatenate([[0], starts])
if in_flight_mask[-1]: ends = np.concatenate([ends, [len(in_flight_mask)]])

if len(starts) == 0:
    print("No flight segment found! Using full sequence.")
    i0, i1 = 0, len(df)
else:
    longest = np.argmax(ends - starts)
    i0, i1 = starts[longest], ends[longest]

print(f"Full trajectory:  {len(df)} rows  ({len(df)*dt:.1f}s)")
print(f"Flight window:    rows {i0}..{i1}  ({(i1-i0)*dt:.1f}s,  t={i0*dt:.1f}s..{i1*dt:.1f}s)")
print(f"Flight avg_cmd:   {cmd_all[i0:i1].mean():.3f}")

cmd_seq = cmd_all[i0:i1]
vbat_seq = vbat_all[i0:i1]
qx = df['qx'].values[i0:i1]; qy = df['qy'].values[i0:i1]
qz = df['qz'].values[i0:i1]; qw = df['qw'].values[i0:i1]
q_gt = np.column_stack([qw, qx, qy, qz])
p_gt = df[['px','py','pz']].values[i0:i1]
v_gt = df[['vx','vy','vz']].values[i0:i1]
omega_gt = df[['imu_gyro_x','imu_gyro_y','imu_gyro_z']].values[i0:i1]

# ─── Initial state at start of flight window ─────────────────────────────────
# Use real Omega from ESC model at the first commanded value rather than hover
first_cmd = cmd_seq[0]
first_U   = vbat_seq[0]
Oss_init, _ = phys.esc_battery_model(first_cmd, first_U, phys.hover_omega_from_params(params),
                                      params['eta'], params['battery_coeffs'], dt, params['c_d'])
Omega_init = np.maximum(Oss_init, 0.0)

initial_state = {
    "p_WB":   p_gt[0].copy(),
    "q_WB":   q_gt[0].copy(),
    "v_WB":   v_gt[0].copy(),
    "omega_B": omega_gt[0].copy(),
    "Omega":  Omega_init,
    "U_bat":  float(vbat_seq[0]),
}
print(f"Initial Omega:    {Omega_init.mean():.1f} rad/s  (hover need: {phys.hover_omega_from_params(params)[0]:.1f})")

# ─── Simulate ────────────────────────────────────────────────────────────────
with warnings.catch_warnings(record=True) as wlist:
    warnings.simplefilter("always")
    sim = phys.simulate_open_loop(cmd_seq, dt, initial_state, params,
                                   U_bat_sequence=vbat_seq, integrator="rk4")
ovf = [w for w in wlist if 'overflow' in str(w.message).lower() or 'invalid' in str(w.message).lower()]
print(f"Overflow warnings: {len(ovf)}")

p_pred = sim["p_WB"]; v_pred = sim["v_WB"]; q_pred = sim["q_WB"]

# ─── Metrics ─────────────────────────────────────────────────────────────────
def quat_angle_error(q1, q2):
    def conj(q): return np.column_stack([q[:,0], -q[:,1], -q[:,2], -q[:,3]])
    def qmul(a,b):
        w1,x1,y1,z1=a[:,0],a[:,1],a[:,2],a[:,3]
        w2,x2,y2,z2=b[:,0],b[:,1],b[:,2],b[:,3]
        return np.column_stack([w1*w2-x1*x2-y1*y2-z1*z2,w1*x2+x1*w2+y1*z2-z1*y2,
                                w1*y2-x1*z2+y1*w2+z1*x2,w1*z2+x1*y2-y1*x2+z1*w2])
    rel = qmul(conj(q1), q2)
    return 2 * np.arctan2(np.linalg.norm(rel[:,1:],axis=1), np.abs(rel[:,0]))

pos_err = np.linalg.norm(p_pred - p_gt, axis=1)
vel_err = np.linalg.norm(v_pred - v_gt, axis=1)
att_err = quat_angle_error(q_gt, q_pred)

print()
print("=" * 60)
print("RESULTS (flight window only)")
print("=" * 60)
print(f"Position error  -- mean: {pos_err.mean()*1000:.1f} mm, "
      f"median: {np.median(pos_err)*1000:.1f} mm, final: {pos_err[-1]*1000:.1f} mm")
print(f"Velocity error  -- mean: {vel_err.mean():.3f} m/s,  final: {vel_err[-1]:.3f} m/s")
print(f"Attitude error  -- mean: {np.degrees(att_err).mean():.2f} deg, final: {np.degrees(att_err[-1]):.2f} deg")
print()
print("Horizon snapshots:")
for t_h in [0.1, 0.5, 1.0, 2.0, 5.0]:
    idx = min(int(t_h/dt)-1, len(pos_err)-1)
    if idx >= 0:
        print(f"  t={t_h:.1f}s: pos={pos_err[idx]*1000:.1f}mm  vel={vel_err[idx]:.3f}m/s  att={np.degrees(att_err[idx]):.2f}deg")

# ─── Plot ────────────────────────────────────────────────────────────────────
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    t_ax = np.arange(len(p_gt)) * dt
    fig, axes = plt.subplots(4, 1, figsize=(12,10), sharex=True)
    labels = ["x","y","z"]
    for i in range(3):
        axes[0].plot(t_ax, p_gt[:,i],'--',label=f"gt {labels[i]}")
        axes[0].plot(t_ax, p_pred[:,i],label=f"pred {labels[i]}")
    axes[0].set_ylabel("position (m)"); axes[0].legend(ncol=3,fontsize=8)
    axes[0].set_title("Position: predicted vs ground truth (flight window)")
    for i in range(3):
        axes[1].plot(t_ax, v_gt[:,i],'--',label=f"gt v{labels[i]}")
        axes[1].plot(t_ax, v_pred[:,i],label=f"pred v{labels[i]}")
    axes[1].set_ylabel("velocity (m/s)"); axes[1].legend(ncol=3,fontsize=8)
    axes[2].plot(t_ax, pos_err*1000); axes[2].set_ylabel("position error (mm)")
    axes[3].plot(t_ax, np.degrees(att_err)); axes[3].set_ylabel("attitude error (deg)")
    axes[3].set_xlabel("time (s)")
    plt.tight_layout()
    out = os.path.join('.', 'validation_flight_only.png')
    plt.savefig(out, dpi=150); print(f"Plot saved: {out}")
except Exception as e:
    print(f"(Plot skipped: {e})")
