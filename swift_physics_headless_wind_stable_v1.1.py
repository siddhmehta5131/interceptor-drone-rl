"""
swift_physics_headless_wind.py

Headless, importable physics core extracted from swift_demo_wind_only.py --
no pygame, no matplotlib, no top-level execution (the live gamepad loop is
gone). Exposes the rigid-body RK4 integrator directly so it can be driven
with externally-supplied constant forces/torques, for apples-to-apples
comparison against PyBullet (PyBullet has no notion of our ESC/motor-lag
or wind models, so those stages are deliberately excluded from this
comparison -- only the RK4 rigid-body integration itself is being checked).
"""

import numpy as np

__version__ = "wind-only-headless-1.1"

_arm = 0.0397
r_P = np.array([
    [ _arm,  _arm, 0.0],
    [-_arm,  _arm, 0.0],
    [-_arm, -_arm, 0.0],
    [ _arm, -_arm, 0.0],
])
zeta = np.array([
    [0.0, 0.0,  1.0],
    [0.0, 0.0, -1.0],
    [0.0, 0.0,  1.0],
    [0.0, 0.0, -1.0],
])
spin_sign = zeta[:, 2]

PARAMS_NANOBENCH = dict(
    m=0.04085,
    J=np.diag([2.3951e-5, 2.3951e-5, 3.2347e-5]),
    J_mp=2.0e-9,
    g_W=np.array([0.0, 0.0, -9.81]),
    r_P=r_P,
    zeta=zeta,
    spin_sign=spin_sign,
    c_l=2.618e-8,
    c_d=5.45e-11,
    k_mot=0.02,
    Omega_max=2800.0,
)


def hover_omega_from_params(params):
    omega = float(np.sqrt(params["m"] * 9.81 / (4.0 * params["c_l"])))
    return np.full(4, omega)


def quat2rotm_manual(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),   1 - 2 * (x**2 + z**2),   2 * (y * z - w * x)],
        [2 * (x * z - w * y),   2 * (y * z + w * x),     1 - 2 * (x**2 + y**2)],
    ])


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def propeller_force_torque(Omega, c_l, c_d, spin_sign):
    f_props = np.zeros((4, 3))
    tau_props = np.zeros((4, 3))
    for j in range(4):
        f_props[j] = np.array([0.0, 0.0, c_l * Omega[j]**2])
        tau_props[j] = np.array([0.0, 0.0, spin_sign[j] * c_d * Omega[j]**2])
    return f_props, tau_props


def aggregate_forces(f_props, tau_props, r_P):
    f_body = np.sum(f_props, axis=0)
    tau_body = np.zeros(3)
    for j in range(4):
        tau_body += tau_props[j] + np.cross(r_P[j], f_props[j])
    return f_body, tau_body


def _rigid_body_deriv_fixed_force(p, q, v, w, f_body, tau_body, m, J, g_W):
    """Rigid-body derivative under a CONSTANT (held-fixed) body-frame
    force/torque -- no ESC, no motor lag, no wind, no aero regression.
    This isolates exactly what PyBullet is also doing: integrate a free
    body forward under one fixed external wrench for one dt.

    v1.1: gyroscopic term -J^-1(w x Jw) added to omega_dot so that
    the angular dynamics match PyBullet (which includes it). Without
    this term trajectories diverge by ~286 mm / 3.25 deg over 6 s
    of aggressive manoeuvring (see trajectory comparison results)."""
    R = quat2rotm_manual(q)
    p_dot = v
    omega_quat = np.array([0.0, w[0], w[1], w[2]])
    q_dot = 0.5 * quat_mult(q, omega_quat)
    v_dot = (1.0 / m) * (R @ f_body) + g_W
    J_inv = np.linalg.inv(J)
    gyro = np.cross(w, J @ w)           # gyroscopic reaction torque
    omega_dot = J_inv @ (tau_body - gyro)
    return p_dot, q_dot, v_dot, omega_dot


def rk4_rigid_body_step(state0, f_body, tau_body, params, dt):
    """state0: dict with p_WB, q_WB, v_WB, omega_B (Omega/U_bat ignored --
    this function only integrates the rigid body under a fixed wrench).
    Returns a new state dict with the same keys, RK4-integrated by dt."""
    m, J, g_W = params["m"], params["J"], params["g_W"]
    p, q, v, w = state0["p_WB"], state0["q_WB"], state0["v_WB"], state0["omega_B"]

    def deriv(p_, q_, v_, w_):
        return _rigid_body_deriv_fixed_force(p_, q_, v_, w_, f_body, tau_body, m, J, g_W)

    s0 = [p, q, v, w]
    k1 = list(deriv(*s0))
    s1 = [s0[i] + 0.5 * dt * k1[i] for i in range(4)]
    k2 = list(deriv(*s1))
    s2 = [s0[i] + 0.5 * dt * k2[i] for i in range(4)]
    k3 = list(deriv(*s2))
    s3 = [s0[i] + dt * k3[i] for i in range(4)]
    k4 = list(deriv(*s3))
    p_n, q_n, v_n, w_n = [s0[i] + (dt / 6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) for i in range(4)]
    q_n = q_n / np.linalg.norm(q_n)
    return dict(p_WB=p_n, q_WB=q_n, v_WB=v_n, omega_B=w_n,
                Omega=state0.get("Omega"), U_bat=state0.get("U_bat"))
