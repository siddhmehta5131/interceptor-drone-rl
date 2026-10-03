"""
compare_v3_vs_pybullet.py  --  Swift v3.0 vs PyBullet Extreme Comparison
==========================================================================

Tests at realistic 400 Hz and 1000 Hz physics rates with extreme scenarios:
  S1:  Standard hover  (baseline sanity)
  S2:  Near-ground hover (z=3cm, deep ground effect)
  S3:  High-speed forward flight (8 m/s, open-loop)
  S4:  Aggressive pitch maneuver (differential thrust)
  S5:  Extreme crosswind (10 m/s gust)
  S6:  Free fall + motor restart
"""

import numpy as np
import time
import sys

# ─── Import Swift v3.0 ───────────────────────────────────────────────────────
import swift_physics_headless_wind as wind
import swift_physics_headless_rlvec as rlvec

# ─── Import PyBullet ──────────────────────────────────────────────────────────
try:
    import pybullet as p
    import pybullet_data
except ImportError:
    print("ERROR: pybullet not installed. Run: pip install pybullet")
    sys.exit(1)

print("=" * 70)
print("  Swift v3.0 vs PyBullet  --  Extreme Case Comparison")
print("  Physics rates: 400 Hz and 1000 Hz")
print("=" * 70)


# ─── Crazyflie constants (shared) ────────────────────────────────────────────
MASS       = 0.04085     # kg
ARM        = 0.0397      # m
C_L        = 2.618e-8    # N/(rad/s)^2
C_D        = 5.45e-11    # N*m/(rad/s)^2
J_XX       = 2.3951e-5
J_YY       = 2.3951e-5
J_ZZ       = 3.2347e-5
R_PROP     = 0.023       # m

OMEGA_HOVER = float(np.sqrt(MASS * 9.81 / (4.0 * C_L)))
THRUST_HOVER_PER_MOTOR = C_L * OMEGA_HOVER**2

# Motor positions (body frame)
MOTOR_POS = np.array([
    [ ARM,  ARM, 0.0],
    [-ARM,  ARM, 0.0],
    [-ARM, -ARM, 0.0],
    [ ARM, -ARM, 0.0],
])
SPIN_DIR = np.array([1, -1, 1, -1])   # CCW, CW, CCW, CW


# ─── PyBullet Crazyflie wrapper ───────────────────────────────────────────────

class PyBulletCrazyflie:
    """Thin PyBullet wrapper that applies per-motor thrust + reaction torque."""

    def __init__(self, dt=0.001, gui=False):
        self.dt = dt
        mode = p.GUI if gui else p.DIRECT
        self.client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(dt, physicsClientId=self.client)

        # Create drone as a box (Crazyflie-scale)
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[ARM, ARM, 0.005],
                                     physicsClientId=self.client)
        vis = -1
        self.body = p.createMultiBody(
            baseMass=MASS,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[0, 0, 5],
            baseOrientation=[0, 0, 0, 1],
            physicsClientId=self.client,
        )
        # Set inertia tensor
        p.changeDynamics(self.body, -1,
                         localInertiaDiagonal=[J_XX, J_YY, J_ZZ],
                         linearDamping=0, angularDamping=0,
                         physicsClientId=self.client)
        # Create ground plane
        self.ground = p.loadURDF("plane.urdf", physicsClientId=self.client)

    def reset(self, pos, quat_xyzw, vel, omega):
        """Reset drone state. quat in [x,y,z,w] for PyBullet."""
        p.resetBasePositionAndOrientation(self.body, pos.tolist(),
                                          quat_xyzw.tolist(),
                                          physicsClientId=self.client)
        p.resetBaseVelocity(self.body, vel.tolist(), omega.tolist(),
                            physicsClientId=self.client)

    def apply_motor_forces(self, Omega):
        """Apply thrust + reaction torque from 4 motors at given speeds."""
        for j in range(4):
            thrust = C_L * Omega[j]**2
            torque_z = SPIN_DIR[j] * C_D * Omega[j]**2

            # Thrust force at motor position (body frame z-up)
            p.applyExternalForce(
                self.body, -1,
                forceObj=[0, 0, thrust],
                posObj=MOTOR_POS[j].tolist(),
                flags=p.LINK_FRAME,
                physicsClientId=self.client,
            )
            # Reaction torque (yaw)
            p.applyExternalTorque(
                self.body, -1,
                torqueObj=[0, 0, torque_z],
                flags=p.LINK_FRAME,
                physicsClientId=self.client,
            )

    def apply_external_force_world(self, force_W):
        """Apply external force in world frame (e.g. wind drag proxy)."""
        p.applyExternalForce(
            self.body, -1,
            forceObj=force_W.tolist(),
            posObj=[0, 0, 0],
            flags=p.WORLD_FRAME,
            physicsClientId=self.client,
        )

    def step(self):
        p.stepSimulation(physicsClientId=self.client)

    def get_state(self):
        pos, orn = p.getBasePositionAndOrientation(
            self.body, physicsClientId=self.client)
        vel, omega = p.getBaseVelocity(self.body, physicsClientId=self.client)
        # PyBullet quaternion is [x,y,z,w], Swift is [w,x,y,z]
        q_wxyz = np.array([orn[3], orn[0], orn[1], orn[2]])
        return {
            'p_WB': np.array(pos),
            'q_WB': q_wxyz,
            'v_WB': np.array(vel),
            'omega_B': np.array(omega),
        }

    def close(self):
        p.disconnect(self.client)


def wxyz_to_xyzw(q):
    """Convert [w,x,y,z] -> [x,y,z,w] for PyBullet."""
    return np.array([q[1], q[2], q[3], q[0]])


# ─── Run comparison ──────────────────────────────────────────────────────────

def run_scenario(name, init_state, motor_speeds_fn, duration_s,
                 wind_force_W=None, sim_rates=[400, 1000]):
    """Run a scenario at multiple sim rates, compare Swift v3.0 vs PyBullet.

    motor_speeds_fn(t) -> (4,) array of Omega per motor
    wind_force_W: (3,) constant world-frame wind force or None
    """
    print(f"\n{'─'*70}")
    print(f"  {name}")
    print(f"{'─'*70}")

    params = rlvec.PARAMS_NANOBENCH

    for rate in sim_rates:
        dt = 1.0 / rate
        n_steps = int(duration_s / dt)

        # ── Swift v3.0 ────────────────────────────────────────────────────
        state_s = {k: v.copy() if isinstance(v, np.ndarray) else v
                   for k, v in init_state.items()}

        t0_s = time.perf_counter()
        swift_traj = []
        for i in range(n_steps):
            t = i * dt
            Om = motor_speeds_fn(t)
            state_s['Omega'] = Om

            f_ps, t_ps = rlvec.propeller_force_torque(
                Om, params['c_l'], params['c_d'], params['spin_sign'])
            f_b, tau_b = rlvec.aggregate_forces(f_ps, t_ps, params['r_P'])

            w_W = wind_force_W / MASS if wind_force_W is not None else None
            # For wind: convert force to velocity-equivalent for the aero model
            # Actually, pass as wind velocity (rough: F=m*a -> a=F/m is accel not vel)
            # Better: just pass zero wind and apply force separately

            state_s = rlvec.rk4_rigid_body_step(
                state_s, f_b, tau_b, params, dt, wind_W=None)

            # Apply wind as velocity-equivalent impulse (simple Euler kick)
            if wind_force_W is not None:
                state_s['v_WB'] = state_s['v_WB'] + (wind_force_W / MASS) * dt

            swift_traj.append(state_s['p_WB'].copy())
        t_swift = time.perf_counter() - t0_s
        swift_traj = np.array(swift_traj)

        # ── PyBullet ──────────────────────────────────────────────────────
        pb = PyBulletCrazyflie(dt=dt)
        q_xyzw = wxyz_to_xyzw(init_state['q_WB'])
        pb.reset(init_state['p_WB'], q_xyzw,
                 init_state['v_WB'], init_state['omega_B'])

        t0_p = time.perf_counter()
        pb_traj = []
        for i in range(n_steps):
            t = i * dt
            Om = motor_speeds_fn(t)
            pb.apply_motor_forces(Om)
            if wind_force_W is not None:
                pb.apply_external_force_world(wind_force_W)
            pb.step()
            st = pb.get_state()
            pb_traj.append(st['p_WB'].copy())
        t_pb = time.perf_counter() - t0_p
        pb_traj = np.array(pb_traj)
        pb.close()

        # ── Compare ───────────────────────────────────────────────────────
        dev = np.linalg.norm(swift_traj - pb_traj, axis=1)
        max_dev = np.max(dev) * 1000   # mm
        mean_dev = np.mean(dev) * 1000
        final_dev = dev[-1] * 1000

        print(f"\n  @ {rate} Hz  ({n_steps} steps, {duration_s}s)")
        print(f"    Swift v3.0  : final pos = {swift_traj[-1]}  ({t_swift:.3f}s wall)")
        print(f"    PyBullet    : final pos = {pb_traj[-1]}  ({t_pb:.3f}s wall)")
        print(f"    Max dev     : {max_dev:.2f} mm")
        print(f"    Mean dev    : {mean_dev:.2f} mm")
        print(f"    Final dev   : {final_dev:.2f} mm")
        print(f"    Speed ratio : Swift {t_pb/t_swift:.1f}x {'faster' if t_swift<t_pb else 'slower'} than PyBullet")


# ─── Scenario definitions ────────────────────────────────────────────────────

def make_init(pos=[0,0,5], vel=[0,0,0], omega=[0,0,0]):
    return dict(
        p_WB=np.array(pos, dtype=float),
        q_WB=np.array([1.0, 0.0, 0.0, 0.0]),
        v_WB=np.array(vel, dtype=float),
        omega_B=np.array(omega, dtype=float),
        Omega=np.full(4, OMEGA_HOVER),
        U_bat=4.2,
    )


# S1: Standard hover
def s1_hover_omega(t):
    return np.full(4, OMEGA_HOVER)

# S2: Near-ground hover (z=3cm)
def s2_ground_omega(t):
    return np.full(4, OMEGA_HOVER)

# S3: High-speed forward (pitch forward by reducing front motors)
def s3_highspeed_omega(t):
    Om = np.full(4, OMEGA_HOVER)
    # Front motors (M1, M4) reduced → pitch forward
    Om[0] -= 200   # M1 front-right
    Om[3] -= 200   # M4 front-left (actually rear... let me check)
    # In our frame: M1=[+arm,+arm], M2=[-arm,+arm], M3=[-arm,-arm], M4=[+arm,-arm]
    # Front = +x direction = M1, M4.  Reduce these to pitch nose-down.
    return Om

# S4: Aggressive yaw + roll (differential thrust)
def s4_aggressive_omega(t):
    Om = np.full(4, OMEGA_HOVER)
    if t < 0.5:
        # Roll: boost M1,M4 (+y side), reduce M2,M3 (-y side)
        Om[0] += 300
        Om[3] += 300
        Om[1] -= 300
        Om[2] -= 300
    else:
        # Reverse
        Om[0] -= 300
        Om[3] -= 300
        Om[1] += 300
        Om[2] += 300
    return np.maximum(Om, 0)

# S5: Hover with extreme crosswind (force equivalent)
def s5_wind_omega(t):
    return np.full(4, OMEGA_HOVER)

# S6: Free fall (motors off for 0.3s, then restart)
def s6_freefall_omega(t):
    if t < 0.3:
        return np.zeros(4)   # motors OFF
    else:
        return np.full(4, OMEGA_HOVER * 1.3)  # 30% extra thrust to recover


# ─── Run all scenarios ───────────────────────────────────────────────────────

if __name__ == "__main__":
    RATES = [400, 1000]

    run_scenario(
        "S1: Standard Hover (5m altitude, 3s)",
        make_init(pos=[0, 0, 5]),
        s1_hover_omega,
        duration_s=3.0,
        sim_rates=RATES,
    )

    run_scenario(
        "S2: Near-Ground Hover (z=3cm, deep ground effect, 2s)",
        make_init(pos=[0, 0, 0.03]),
        s2_ground_omega,
        duration_s=2.0,
        sim_rates=RATES,
    )

    run_scenario(
        "S3: High-Speed Forward Flight (pitch-down differential, 3s)",
        make_init(pos=[0, 0, 5], vel=[5, 0, 0]),
        s3_highspeed_omega,
        duration_s=3.0,
        sim_rates=RATES,
    )

    run_scenario(
        "S4: Aggressive Roll Reversal (differential thrust, 2s)",
        make_init(pos=[0, 0, 5]),
        s4_aggressive_omega,
        duration_s=2.0,
        sim_rates=RATES,
    )

    run_scenario(
        "S5: Hover + 10 m/s Crosswind (as force, 3s)",
        make_init(pos=[0, 0, 5]),
        s5_wind_omega,
        duration_s=3.0,
        wind_force_W=np.array([0.05, 0.0, 0.0]),   # ~10 m/s wind drag force proxy
        sim_rates=RATES,
    )

    run_scenario(
        "S6: Free Fall + Motor Restart (0.3s off, then recover, 2s)",
        make_init(pos=[0, 0, 10]),
        s6_freefall_omega,
        duration_s=2.0,
        sim_rates=RATES,
    )

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  COMPARISON COMPLETE")
    print("=" * 70)
    print("\nNotes:")
    print("  - PyBullet does NOT model ground effect, blade flapping, or hub drag")
    print("  - PyBullet does NOT model body aerodynamic drag (polynomial model)")
    print("  - Deviations in S2 (near-ground) are expected: Swift has GE, PyBullet doesn't")
    print("  - Deviations in S3/S5 are partly due to Swift's aero drag model")
    print("  - At 1000 Hz both engines should converge to similar results")
    print("  - Speed ratio shows Swift's advantage for RL training (no C++ overhead)")
