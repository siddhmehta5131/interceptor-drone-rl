"""
identify_params.py
Identifies c_l, c_d, and J from the NanoBench dataset directly:

1. c_l: from hover segments where vz~0, pz stable -- T = m*g = 4*c_l*Omega^2
2. c_d: from yaw rate change vs net motor torque -- tau_z = sum(sign_i * c_d * Omega_i^2)
3. J check: compare computed omega_dot to measured omega_dot
   to check if J needs scaling
"""
import pandas as pd, numpy as np, sys, glob
sys.path.insert(0, '.')
import swift_physics_headless as phys

# Use multiple files for better statistics
csv_files = sorted(glob.glob(r'nanobench-iros2026\datasets\dataset\*.csv'))
csv_files = [f for f in csv_files if '_metadata' not in f]
print(f"Found {len(csv_files)} trajectory CSVs")

all_hover_Omega = []
all_hover_Omega_sq = []
all_tau_z = []
all_omega_z_dot = []
all_tau_rp = []     # roll/pitch torques from motor imbalance
all_omega_rp_dot = [] # actual roll/pitch angular accel

params = phys.PARAMS_NANOBENCH
m  = params['m']
g  = 9.81
dt = 0.01

for fpath in csv_files[:10]:   # use first 10 trajectories
    try:
        df = pd.read_csv(fpath)
        if not all(c in df.columns for c in ['motor_motor_m1','pwr_pm_vbat','imu_gyro_x']):
            continue

        pwm = df[['motor_motor_m1','motor_motor_m2','motor_motor_m3','motor_motor_m4']].values / 65535.0
        vbat = df['pwr_pm_vbat'].values
        pz   = df['pz'].values
        vz   = df['vz'].values
        omega_x = df['imu_gyro_x'].values
        omega_y = df['imu_gyro_y'].values
        omega_z = df['imu_gyro_z'].values

        N = len(df)
        # Get Omega_ss from ESC model
        Omega_ss_all = np.zeros((N, 4))
        Omega_prev = phys.hover_omega_from_params(params)
        for i in range(N):
            Oss, _ = phys.esc_battery_model(pwm[i], vbat[i], Omega_prev,
                                             params['eta'], params['battery_coeffs'], dt, params['c_d'])
            Omega_ss_all[i] = np.maximum(Oss, 0.0)
            Omega_prev = Omega_ss_all[i]

        # ── 1. Hover segments: T = m*g, find c_l ─────────────────────────
        # "hover" = pz > 0.3m, |vz| < 0.1 m/s, |vx|+|vy| < 0.3 m/s
        vx = df['vx'].values; vy = df['vy'].values
        hover_mask = (pz > 0.3) & (np.abs(vz) < 0.1) & (np.abs(vx) + np.abs(vy) < 0.3)
        if hover_mask.sum() > 50:
            # T_total = 4 * c_l * mean(Omega^2) = m*g
            # c_l = m*g / (4 * mean(Omega^2))
            Omega_sq_hover = (Omega_ss_all[hover_mask]**2).mean(axis=1)  # mean across 4 motors per timestep
            c_l_estimates = (m * g) / (4 * Omega_sq_hover)
            all_hover_Omega_sq.extend(Omega_sq_hover.tolist())
            all_hover_Omega.extend(Omega_ss_all[hover_mask].mean(axis=1).tolist())

        # ── 2. Yaw dynamics: tau_z = c_d * sum(sign_i * Omega_i^2) = J_zz * omega_z_dot
        spin_sign = params['spin_sign']  # [+1,-1,+1,-1]
        for i in range(1, N-1):
            if pz[i] < 0.3:
                continue
            net_Omega_sq = np.sum(spin_sign * Omega_ss_all[i]**2)
            oz_dot = (omega_z[i+1] - omega_z[i-1]) / (2*dt)
            all_tau_z.append(net_Omega_sq)
            all_omega_z_dot.append(oz_dot)

        # ── 3. Roll/pitch torque vs angular accel (to check J_xx)
        # r_P for roll torque: tau_roll = sum(sign(r_P_y) * c_l * Omega_i^2)
        # i.e. motors 1,2 (y>0) push up, motors 3,4 (y<0) push down
        roll_sign = np.sign(params['r_P'][:, 1])  # [+1,+1,-1,-1]
        pitch_sign = -np.sign(params['r_P'][:, 0]) # [-1,+1,+1,-1]
        arm = 0.0397
        for i in range(1, N-1):
            if pz[i] < 0.5:
                continue
            # Roll torque from motor forces (using current c_l)
            F_i = params['c_l'] * Omega_ss_all[i]**2
            tau_roll_cl  = arm * np.sum(roll_sign  * F_i)
            tau_pitch_cl = arm * np.sum(pitch_sign * F_i)
            ox_dot = (omega_x[i+1] - omega_x[i-1]) / (2*dt)
            oy_dot = (omega_y[i+1] - omega_y[i-1]) / (2*dt)
            all_tau_rp.append([tau_roll_cl, tau_pitch_cl])
            all_omega_rp_dot.append([ox_dot, oy_dot])

    except Exception as e:
        print(f"  Skipped {fpath}: {e}")

all_hover_Omega_sq = np.array(all_hover_Omega_sq)
all_hover_Omega    = np.array(all_hover_Omega)
all_tau_z          = np.array(all_tau_z)
all_omega_z_dot    = np.array(all_omega_z_dot)
all_tau_rp         = np.array(all_tau_rp)
all_omega_rp_dot   = np.array(all_omega_rp_dot)

print(f"\nHover samples: {len(all_hover_Omega_sq)}")
print(f"Yaw samples:   {len(all_tau_z)}")
print(f"Roll/pitch samples: {len(all_tau_rp)}")

# ── c_l from hover ───────────────────────────────────────────────────────────
if len(all_hover_Omega_sq) > 0:
    c_l_hover = (m * g) / (4 * all_hover_Omega_sq)
    c_l_mean  = np.median(c_l_hover)
    c_l_std   = c_l_hover.std()
    print(f"\n=== c_l FROM HOVER ===")
    print(f"  median c_l = {c_l_mean:.4e}  std={c_l_std:.4e}")
    print(f"  mean   c_l = {c_l_hover.mean():.4e}")
    print(f"  Current in model: {params['c_l']:.4e}")
    print(f"  Ratio (data/model): {c_l_mean/params['c_l']:.3f}")
    print(f"  Mean hover Omega:   {all_hover_Omega.mean():.1f} rad/s  (model hover: {phys.hover_omega_from_params(params)[0]:.1f} rad/s)")

# ── c_d from yaw dynamics ────────────────────────────────────────────────────
if len(all_tau_z) > 50:
    # J_zz * omega_z_dot = c_d * net_Omega_sq + noise
    # Filter out outliers
    good = np.abs(all_omega_z_dot) < 50
    tau_z_g = all_tau_z[good]
    odz_g   = all_omega_z_dot[good]
    # Least squares: c_d = (J_zz * omega_z_dot) / net_Omega_sq
    J_zz = params['J'][2,2]
    c_d_estimates = (J_zz * odz_g) / tau_z_g
    c_d_median = np.median(c_d_estimates)
    print(f"\n=== c_d FROM YAW DYNAMICS ===")
    print(f"  median c_d = {c_d_median:.4e}")
    print(f"  mean   c_d = {c_d_estimates.mean():.4e}  std={c_d_estimates.std():.4e}")
    print(f"  Current in model: {params['c_d']:.4e}")
    print(f"  Ratio (data/model): {c_d_median/params['c_d']:.3f}")

# ── J_xx from roll/pitch dynamics ────────────────────────────────────────────
if len(all_tau_rp) > 50:
    # J_xx * omega_x_dot = tau_roll  =>  J_xx = tau_roll / omega_x_dot
    good_x = np.abs(all_omega_rp_dot[:,0]) > 0.5   # need nonzero accel for good SNR
    good_y = np.abs(all_omega_rp_dot[:,1]) > 0.5
    if good_x.sum() > 20:
        J_xx_est = all_tau_rp[good_x,0] / all_omega_rp_dot[good_x,0]
        J_xx_med = np.median(J_xx_est)
        J_xx_q1  = np.percentile(J_xx_est, 25)
        J_xx_q3  = np.percentile(J_xx_est, 75)
        print(f"\n=== J_xx FROM ROLL DYNAMICS ===")
        print(f"  median J_xx = {J_xx_med:.4e}  IQR=[{J_xx_q1:.4e}, {J_xx_q3:.4e}]")
        print(f"  Current in model: {params['J'][0,0]:.4e}")
        print(f"  Ratio (data/model): {J_xx_med/params['J'][0,0]:.3f}")
        print(f"  -> If ratio >> 1: simulated torque is too small relative to accel (c_l too small)")
        print(f"  -> If ratio << 1: simulated torque is too large relative to accel (c_l too large or J too small)")
    if good_y.sum() > 20:
        J_yy_est = all_tau_rp[good_y,1] / all_omega_rp_dot[good_y,1]
        J_yy_med = np.median(J_yy_est)
        print(f"\n=== J_yy FROM PITCH DYNAMICS ===")
        print(f"  median J_yy = {J_yy_med:.4e}")
        print(f"  Current in model: {params['J'][1,1]:.4e}")
        print(f"  Ratio (data/model): {J_yy_med/params['J'][1,1]:.3f}")
