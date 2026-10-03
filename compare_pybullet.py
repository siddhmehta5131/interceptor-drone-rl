"""
compare_pybullet.py
====================
Compares swift_physics_headless.py against PyBullet rigid-body dynamics
on synthetically generated test cases.

DESIGN: Both models receive IDENTICAL inputs:
  - Same initial state (position, velocity, quaternion, angular rate)
  - Same motor angular velocities (Omega in rad/s)
  - Same timestep (dt = 0.01 s)

The forces/torques are computed by OUR motor model (propeller_force_torque,
aggregate_forces) and applied to BOTH models identically.
This isolates the rigid-body integration (Euler/RK4 vs PyBullet) from any
motor model differences.

Test cases cover:
  1. Pure hover (symmetric motors, no wind)
  2. Hover + uniform noise on all motors (small imbalance)
  3. Aggressive pitch/roll maneuver (large motor asymmetry)
  4. Full random (random state + random commands)
"""
import sys, os
import numpy as np

sys.path.insert(0, '.')
import importlib, swift_physics_headless
importlib.reload(swift_physics_headless)
phys = swift_physics_headless

# ── Try importing PyBullet ────────────────────────────────────────────────────
try:
    import pybullet as p
    import pybullet_data
    HAS_PYBULLET = True
except ImportError:
    HAS_PYBULLET = False
    print("WARNING: pybullet not installed. Only swift model will run.")
    print("Install with: pip install pybullet")

params  = phys.PARAMS_NANOBENCH
dt      = 0.01
m       = params['m']
J       = params['J']
g       = 9.81
rng     = np.random.default_rng(42)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  PyBullet setup (headless / DIRECT mode, no GUI, no gravity from PyBullet
#     since we apply gravity explicitly via our force model, same as swift)
# ─────────────────────────────────────────────────────────────────────────────
def setup_pybullet():
    """Create a minimal pybullet world: a point-mass sphere with correct
    mass and diagonal inertia matching PARAMS_NANOBENCH."""
    client = p.connect(p.DIRECT)
    p.resetSimulation(physicsClientId=client)
    # Turn OFF pybullet's built-in gravity -- we apply it ourselves via forces
    p.setGravity(0, 0, 0, physicsClientId=client)
    p.setTimeStep(dt, physicsClientId=client)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)

    # Create a sphere with the right mass.  Inertia is overridden below.
    col  = p.createCollisionShape(p.GEOM_SPHERE, radius=0.05, physicsClientId=client)
    vis  = p.createVisualShape(p.GEOM_SPHERE, radius=0.05,
                                rgbaColor=[0,0,1,0.5], physicsClientId=client)
    body = p.createMultiBody(baseMass=m,
                              baseCollisionShapeIndex=col,
                              baseVisualShapeIndex=vis,
                              basePosition=[0, 0, 1],
                              physicsClientId=client)

    # Override diagonal inertia to match our J matrix exactly
    Jxx, Jyy, Jzz = J[0,0], J[1,1], J[2,2]
    p.changeDynamics(body, -1,
                     localInertiaDiagonal=(Jxx, Jyy, Jzz),
                     physicsClientId=client)
    # Disable all damping (we don't have it in swift either)
    p.changeDynamics(body, -1, linearDamping=0.0, angularDamping=0.0,
                     physicsClientId=client)
    return client, body


def pybullet_onestep(client, body, pos, quat_wxyz, vel, omega_B, forces_world, torques_world):
    """
    Set state, apply forces/torques, step once, return new state.
    
    forces_world:  (3,) net force in WORLD frame (N)
    torques_world: (3,) net torque in WORLD frame (N*m)
    omega_B:       (3,) angular velocity in BODY frame
    quat_wxyz:     (4,) quaternion (w,x,y,z)  -- PyBullet wants (x,y,z,w)
    """
    # PyBullet quaternion convention: (x,y,z,w)
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])

    # Set position & orientation
    p.resetBasePositionAndOrientation(body, pos.tolist(), quat_xyzw.tolist(),
                                      physicsClientId=client)
    # Convert omega from body to world frame
    R = np.array(p.getMatrixFromQuaternion(quat_xyzw, physicsClientId=client)).reshape(3,3)
    omega_W = R @ omega_B
    p.resetBaseVelocity(body, vel.tolist(), omega_W.tolist(), physicsClientId=client)

    # Apply forces and torques in world frame at center of mass
    p.applyExternalForce(body, -1, forces_world.tolist(),
                         [0,0,0], p.LINK_FRAME,      # actually WORLD_FRAME for base
                         physicsClientId=client)
    # Note: applyExternalForce with LINK_FRAME on base = WORLD_FRAME for pos [0,0,0]
    # For torque use WORLD_FRAME explicitly
    p.applyExternalTorque(body, -1, torques_world.tolist(),
                          p.WORLD_FRAME, physicsClientId=client)

    p.stepSimulation(physicsClientId=client)

    pos_new, quat_new_xyzw = p.getBasePositionAndOrientation(body, physicsClientId=client)
    vel_new, omega_W_new   = p.getBaseVelocity(body, physicsClientId=client)

    pos_new   = np.array(pos_new)
    vel_new   = np.array(vel_new)
    # Convert omega back to body frame
    R_new     = np.array(p.getMatrixFromQuaternion(quat_new_xyzw,
                          physicsClientId=client)).reshape(3,3)
    omega_B_new = R_new.T @ np.array(omega_W_new)
    # Quaternion back to (w,x,y,z)
    q = np.array(quat_new_xyzw)
    quat_new_wxyz = np.array([q[3], q[0], q[1], q[2]])

    return pos_new, quat_new_wxyz, vel_new, omega_B_new


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Shared force/torque computation (identical for both models)
# ─────────────────────────────────────────────────────────────────────────────
def compute_forces_and_torques(Omega, q_wxyz):
    """
    Given motor speeds Omega (rad/s, 4-vector) and attitude quaternion,
    return net force and torque in WORLD and BODY frames.
    These are passed to BOTH models so they see identical physics inputs.
    """
    # Propeller forces in body frame
    f_props, tau_props = phys.propeller_force_torque(
        Omega, params['c_l'], params['c_d'], params['spin_sign'])
    f_body, tau_body = phys.aggregate_forces(f_props, tau_props, params['r_P'])

    # Add gravity in body frame (swift convention: g_W = [0,0,-9.81])
    R = phys.quat2rotm_manual(q_wxyz)          # 3x3 rotation matrix world<-body
    g_W = params['g_W']                         # [0, 0, -9.81]
    f_grav_body = R.T @ (m * g_W)              # gravity in body frame
    f_total_body = f_body + f_grav_body

    # Convert total force to world frame (for PyBullet which uses world frame)
    f_total_world  = R @ f_total_body
    tau_total_world = R @ tau_body              # torque world frame (no gravity torque)

    return f_total_body, tau_body, f_total_world, tau_total_world


# ─────────────────────────────────────────────────────────────────────────────
# 3.  swift one-step (Euler integration of rigid body)
# ─────────────────────────────────────────────────────────────────────────────
def swift_onestep(pos, q, vel, omega_B, Omega):
    """One Euler step using swift's rigid_body_dynamics."""
    f_body, tau_body, _, _ = compute_forces_and_torques(Omega, q)

    # rigid_body_dynamics expects separate force contributions -- we pass
    # the full total as f_prop and zero for aero (we already computed everything)
    p_dot, q_dot, v_dot, om_dot, _ = phys.rigid_body_dynamics(
        f_prop   = f_body,
        f_aero   = np.zeros(3),
        tau_prop = tau_body,
        tau_mot  = np.zeros(3),
        tau_aero = np.zeros(3),
        tau_iner = np.zeros(3),
        q_WB     = q,
        v_WB     = vel,
        omega_B  = omega_B,
        Omega_ss = Omega,
        Omega    = Omega,
        m        = m,
        J        = J,
        g_W      = np.zeros(3),   # already included in f_body above
        k_mot    = params['k_mot'],
    )
    pos_new = pos  + dt * p_dot
    q_raw   = q    + dt * q_dot
    q_new   = q_raw / np.linalg.norm(q_raw)
    vel_new = vel  + dt * v_dot
    om_new  = omega_B + dt * om_dot
    return pos_new, q_new, vel_new, om_new


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Synthetic test case generator
# ─────────────────────────────────────────────────────────────────────────────
hover_omega = phys.hover_omega_from_params(params)   # ~1956 rad/s

def gen_test_cases():
    cases = []

    # ── A. Pure hover (symmetric) ──────────────────────────────────────────
    cases.append(dict(
        label="hover_symmetric",
        pos   = np.array([0.0, 0.0, 1.0]),
        q     = np.array([1.0, 0.0, 0.0, 0.0]),   # identity rotation
        vel   = np.zeros(3),
        omega = np.zeros(3),
        Omega = hover_omega.copy(),
    ))

    # ── B. Hover + small motor noise (realistic PID jitter ±2%) ───────────
    noise_small = rng.uniform(-0.02, 0.02, 4) * hover_omega[0]
    cases.append(dict(
        label="hover_small_noise",
        pos   = np.array([0.0, 0.0, 1.0]),
        q     = np.array([1.0, 0.0, 0.0, 0.0]),
        vel   = np.zeros(3),
        omega = np.zeros(3),
        Omega = hover_omega + noise_small,
    ))

    # ── C. Aggressive pitch maneuver (±15% motor spread, like fast trefoil) ─
    pitch_delta = np.array([0.15, 0.15, -0.15, -0.15]) * hover_omega[0]
    cases.append(dict(
        label="aggressive_pitch",
        pos   = np.array([0.5, 0.0, 1.2]),
        q     = np.array([0.9962, 0.0872, 0.0, 0.0]),  # ~10° roll
        vel   = np.array([1.0, 0.0, 0.2]),
        omega = np.array([2.0, 0.5, 0.1]),
        Omega = hover_omega + pitch_delta,
    ))

    # ── D. Aggressive yaw + climb ──────────────────────────────────────────
    yaw_delta = np.array([0.1, -0.1, 0.1, -0.1]) * hover_omega[0]
    climb_delta = np.array([0.2, 0.2, 0.2, 0.2]) * hover_omega[0]
    cases.append(dict(
        label="yaw_and_climb",
        pos   = np.array([0.0, 0.5, 0.8]),
        q     = np.array([0.9659, 0.0, 0.0, 0.2588]),  # ~30° yaw
        vel   = np.array([0.0, 0.5, 0.8]),
        omega = np.array([0.0, 0.0, 3.0]),
        Omega = hover_omega + yaw_delta + climb_delta,
    ))

    # ── E. 500 random cases (random state + random motor speeds) ──────────
    for i in range(500):
        # Random attitude (small angles -- keep physically reasonable)
        angle = rng.uniform(0, np.radians(30))
        axis  = rng.standard_normal(3); axis /= np.linalg.norm(axis)
        q = np.array([np.cos(angle/2), *(np.sin(angle/2)*axis)])
        Omega_rand = hover_omega + rng.uniform(-0.25, 0.35, 4) * hover_omega[0]
        Omega_rand = np.clip(Omega_rand, 100, params['Omega_max'])
        cases.append(dict(
            label = f"random_{i:04d}",
            pos   = rng.uniform(-1, 1, 3) * np.array([1,1,0.5]) + np.array([0,0,1]),
            q     = q,
            vel   = rng.uniform(-2, 2, 3),
            omega = rng.uniform(-5, 5, 3),
            Omega = Omega_rand,
        ))
    return cases


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Run comparison
# ─────────────────────────────────────────────────────────────────────────────
def quat_angle_deg(q1, q2):
    """Angle between two quaternions (w,x,y,z), in degrees."""
    w1,x1,y1,z1 = q1; w2,x2,y2,z2 = q2
    # conj(q1) * q2
    rw =  w1*w2+x1*x2+y1*y2+z1*z2
    rx = -x1*w2+w1*x2-z1*y2+y1*z2
    ry = -y1*w2+z1*x2+w1*y2-x1*z2
    rz = -z1*w2-y1*x2+x1*y2+w1*z2
    return np.degrees(2 * np.arctan2(np.sqrt(rx**2+ry**2+rz**2), abs(rw)))


test_cases = gen_test_cases()
print(f"Generated {len(test_cases)} test cases")

if not HAS_PYBULLET:
    print("PyBullet not available. Exiting.")
    sys.exit(1)

client, body = setup_pybullet()
print(f"PyBullet initialized (client={client}, body={body})")
print(f"Drone mass: {m*1000:.1f}g  Jxx={J[0,0]:.4e}  hover_omega={hover_omega[0]:.1f} rad/s")
print()

results = []

for tc in test_cases:
    pos   = np.array(tc['pos'],   dtype=float)
    q     = np.array(tc['q'],     dtype=float); q /= np.linalg.norm(q)
    vel   = np.array(tc['vel'],   dtype=float)
    omega = np.array(tc['omega'], dtype=float)
    Omega = np.array(tc['Omega'], dtype=float)

    # Compute forces identically for both models
    _, tau_body, f_world, tau_world = compute_forces_and_torques(Omega, q)

    # ── Swift ──────────────────────────────────────────────────────────────
    try:
        pos_s, q_s, vel_s, om_s = swift_onestep(pos, q, vel, omega, Omega)
        swift_ok = True
    except Exception as e:
        swift_ok = False; pos_s = q_s = vel_s = om_s = None

    # ── PyBullet ───────────────────────────────────────────────────────────
    try:
        pos_b, q_b, vel_b, om_b = pybullet_onestep(
            client, body, pos, q, vel, omega, f_world, tau_world)
        bullet_ok = True
    except Exception as e:
        bullet_ok = False; pos_b = q_b = vel_b = om_b = None

    if swift_ok and bullet_ok:
        pos_err = np.linalg.norm(pos_s - pos_b) * 1000   # mm
        vel_err = np.linalg.norm(vel_s - vel_b)            # m/s
        att_err = quat_angle_deg(q_s, q_b)                 # deg
        om_err  = np.linalg.norm(om_s - om_b)              # rad/s
        results.append(dict(
            label=tc['label'],
            pos_err=pos_err, vel_err=vel_err, att_err=att_err, om_err=om_err,
            pos_s=pos_s, pos_b=pos_b, vel_s=vel_s, vel_b=vel_b,
            q_s=q_s, q_b=q_b,
        ))

p.disconnect(client)
print(f"Completed {len(results)} / {len(test_cases)} test cases")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Report
# ─────────────────────────────────────────────────────────────────────────────
pos_errs = np.array([r['pos_err'] for r in results])
vel_errs = np.array([r['vel_err'] for r in results])
att_errs = np.array([r['att_err'] for r in results])
om_errs  = np.array([r['om_err']  for r in results])

print()
print("=" * 65)
print("SWIFT vs PYBULLET  -- one-step comparison (same forces, dt=0.01s)")
print("=" * 65)
print(f"{'Metric':<22} {'Mean':>10} {'Median':>10} {'Max':>10}")
print("-" * 55)
print(f"{'Position (mm)':<22} {pos_errs.mean():>10.5f} {np.median(pos_errs):>10.5f} {pos_errs.max():>10.5f}")
print(f"{'Velocity (m/s)':<22} {vel_errs.mean():>10.6f} {np.median(vel_errs):>10.6f} {vel_errs.max():>10.6f}")
print(f"{'Attitude (deg)':<22} {att_errs.mean():>10.6f} {np.median(att_errs):>10.6f} {att_errs.max():>10.6f}")
print(f"{'Ang. velocity (rad/s)':<22} {om_errs.mean():>10.6f} {np.median(om_errs):>10.6f} {om_errs.max():>10.6f}")
print()

# Named test cases
print("Named test cases:")
print(f"  {'Label':<25} {'pos(mm)':>10} {'vel(m/s)':>10} {'att(deg)':>10} {'om(r/s)':>10}")
for r in results[:4]:
    print(f"  {r['label']:<25} {r['pos_err']:>10.5f} {r['vel_err']:>10.6f} {r['att_err']:>10.6f} {r['om_err']:>10.6f}")

# What level of agreement is "good"?
print()
print("What these numbers mean:")
print(f"  Floating point equality would be ~1e-15. Machine precision is ~1e-10.")
print(f"  Values < 1e-6 mm / 1e-6 deg: effectively identical (just integration order)")
print(f"  Values < 0.1 mm / 0.01 deg:  very good agreement")
print(f"  Values > 1 mm / 1 deg:        significant difference (integration method matters)")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Plot
# ─────────────────────────────────────────────────────────────────────────────
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Swift vs PyBullet — one-step prediction difference\n"
                 "(same forces applied to both — only integration method differs)")

    datasets = [
        (pos_errs, "Position diff (mm)", "steelblue"),
        (vel_errs, "Velocity diff (m/s)", "seagreen"),
        (att_errs, "Attitude diff (deg)", "darkorange"),
        (om_errs,  "Angular rate diff (rad/s)", "mediumpurple"),
    ]
    for ax, (data, label, color) in zip(axes, datasets):
        clip = np.percentile(data, 99)
        ax.hist(data[data <= clip], bins=60, color=color, edgecolor='none', alpha=0.85)
        ax.axvline(np.median(data), color='red', linewidth=2,
                   label=f'median={np.median(data):.3e}')
        ax.set_xlabel(label); ax.set_title(label)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("pybullet_comparison.png", dpi=150)
    print(f"\nPlot saved: pybullet_comparison.png")
except Exception as e:
    print(f"(Plot skipped: {e})")
