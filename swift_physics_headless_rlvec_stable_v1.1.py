"""
swift_physics_headless_rlvec.py

Headless wrapper around the vectorized functions in swift_rl_env.py.
Requires swift_rl_env.py to also be uploaded/importable (this module does
not duplicate its physics, it re-uses the same quat2rotm_batch /
quat_mult_batch that the full vectorized env uses, so the comparison
actually exercises the vectorized code path).

Like swift_physics_headless_wind.py, this isolates the rigid-body RK4
integrator under a fixed, externally-supplied force/torque -- no ESC,
motor lag, wind, aero regression, or domain randomization -- so PyBullet
and Swift are integrating the exact same physical problem for one dt.
N=1 is used throughout so results are directly comparable to the
single-drone PyBullet reference.
"""

import numpy as np
import swift_rl_env as rlenv

__version__ = "rlvec-headless-1.1"

PARAMS_NANOBENCH = dict(
    m=rlenv._m_nom,
    J=rlenv._J_nom,
    J_mp=rlenv.J_mp,
    g_W=rlenv.g_W,
    r_P=rlenv.r_P,
    zeta=rlenv.zeta,
    spin_sign=rlenv.spin_sign,
    c_l=rlenv._c_l_nom,
    c_d=rlenv._c_d_nom,
    k_mot=rlenv.k_mot,
    Omega_max=rlenv.Omega_max,
)


def hover_omega_from_params(params):
    omega = float(np.sqrt(params["m"] * 9.81 / (4.0 * params["c_l"])))
    return np.full(4, omega)


def propeller_force_torque(Omega, c_l, c_d, spin_sign):
    """Single-drone convenience wrapper around the batched version (N=1)."""
    f_props, tau_props = rlenv.propeller_force_torque_batch(
        Omega[None, :], np.array([c_l]), np.array([c_d]))
    return f_props[0], tau_props[0]


def aggregate_forces(f_props, tau_props, r_P):
    f_body, tau_body = rlenv.aggregate_forces_batch(f_props[None, :, :], tau_props[None, :, :])
    return f_body[0], tau_body[0]


def _rigid_body_deriv_fixed_force_batch(p, q, v, w, f_body, tau_body, m, J_inv, g_W):
    """N=1 batched rigid-body derivative under a fixed wrench (no ESC,
    motor lag, wind, or aero) -- the same isolation swift_physics_headless_wind
    makes, but going through the vectorized quat2rotm_batch/quat_mult_batch
    machinery so the RL env's actual code path is what's being tested.

    v1.1: gyroscopic term -J^-1(w x Jw) added to omega_dot so that
    the angular dynamics match PyBullet (which includes it). Without
    this term trajectories diverge by ~286 mm / 3.25 deg over 6 s
    of aggressive manoeuvring (see trajectory comparison results)."""
    R = rlenv.quat2rotm_batch(q)
    p_dot = v
    omega_quat = np.concatenate([np.zeros((1, 1)), w], axis=1)
    q_dot = 0.5 * rlenv.quat_mult_batch(q, omega_quat)
    v_dot = np.einsum('nij,nj->ni', R, f_body) / m[:, None] + g_W
    # gyroscopic term: w x (J w) per drone, then rotate into J_inv
    # J = inv(J_inv); for diagonal J we compute J_w directly via J_inv inverse
    J_diag = 1.0 / J_inv[:, [0,1,2], [0,1,2]]          # (N,3) diagonal entries of J
    Jw = J_diag * w                                       # (N,3) = J @ w  (diagonal J)
    gyro = np.cross(w, Jw)                               # (N,3) = w x Jw
    omega_dot = np.einsum('nij,nj->ni', J_inv, tau_body - gyro)
    return p_dot, q_dot, v_dot, omega_dot


def rk4_rigid_body_step(state0, f_body, tau_body, params, dt):
    """state0: dict with p_WB, q_WB, v_WB, omega_B as plain (3,)/(4,) arrays
    (single drone). f_body, tau_body: (3,) constant body-frame wrench.
    Returns a new state dict, same shapes, RK4-integrated by dt, run
    through the N=1 vectorized code path from swift_rl_env.py."""
    m = np.array([params["m"]])
    J_inv = np.zeros((1, 3, 3))
    idx = np.arange(3)
    J_inv[0, idx, idx] = 1.0 / np.diag(params["J"]) if params["J"].ndim == 2 else 1.0 / params["J"]
    g_W = params["g_W"]

    p = state0["p_WB"][None, :]
    q = state0["q_WB"][None, :]
    v = state0["v_WB"][None, :]
    w = state0["omega_B"][None, :]
    f = f_body[None, :]
    tau = tau_body[None, :]

    def deriv(p_, q_, v_, w_):
        return _rigid_body_deriv_fixed_force_batch(p_, q_, v_, w_, f, tau, m, J_inv, g_W)

    s0 = [p, q, v, w]
    k1 = list(deriv(*s0))
    s1 = [s0[i] + 0.5 * dt * k1[i] for i in range(4)]
    k2 = list(deriv(*s1))
    s2 = [s0[i] + 0.5 * dt * k2[i] for i in range(4)]
    k3 = list(deriv(*s2))
    s3 = [s0[i] + dt * k3[i] for i in range(4)]
    k4 = list(deriv(*s3))
    p_n, q_n, v_n, w_n = [s0[i] + (dt / 6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) for i in range(4)]
    q_n = q_n / np.linalg.norm(q_n, axis=1, keepdims=True)
    return dict(p_WB=p_n[0], q_WB=q_n[0], v_WB=v_n[0], omega_B=w_n[0],
                Omega=state0.get("Omega"), U_bat=state0.get("U_bat"))
